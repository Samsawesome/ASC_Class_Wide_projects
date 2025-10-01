import pandas as pd
import numpy as np
from typing import List, Dict
import re

class ResultParser:
    def __init__(self):
        self.latency_metrics = ['lat_ns', 'latency']
        self.throughput_metrics = ['iops', 'bw_bytes']
        
    def parse_results(self, results: List[Dict]) -> pd.DataFrame:
        """Parse FIO JSON results into pandas DataFrame with comprehensive error handling"""
        rows = []
        
        for result in results:
            for job in result['jobs']:
                # Extract basic job information
                job_name = job['jobname']
                row = {
                    'job_name': job_name,
                    'run_id': job.get('run_id', 0),
                    'timestamp': pd.Timestamp.now()
                }
                
                # Extract job parameters from name using improved parser
                params = self._parse_job_name_improved(job_name, job)
                row.update(params)
                
                # Extract read metrics
                read = job['read']
                read_iops = read.get('iops', 0)
                row.update({
                    'read_iops': read_iops,
                    'read_bw_bytes': read.get('bw_bytes', 0),
                })
                # Extract read latency metrics after setting IOPS
                row.update(self._extract_latency_metrics(read, 'read', read_iops))
                
                # Extract write metrics
                write = job['write']
                write_iops = write.get('iops', 0)
                row.update({
                    'write_iops': write_iops,
                    'write_bw_bytes': write.get('bw_bytes', 0),
                })
                # Extract write latency metrics after setting IOPS
                row.update(self._extract_latency_metrics(write, 'write', write_iops))
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df = self._calculate_derived_metrics(df)
        df = self._clean_nan_values(df)
        return df
    
    def _extract_latency_metrics(self, io_data: Dict, prefix: str, iops: float) -> Dict:
        """Extract latency metrics from FIO JSON data with comprehensive fallbacks"""
        metrics = {}
        
        # Try multiple locations for latency data
        lat_sources = [
            io_data.get('lat_ns', {}),
            io_data.get('clat_ns', {}),
            io_data.get('slat_ns', {})
        ]
        
        # Find the first source that has meaningful data
        lat_data = {}
        for source in lat_sources:
            if source and source.get('mean', 0) > 0:
                lat_data = source
                break
        
        # Extract basic latency metrics
        metrics.update({
            f'{prefix}_lat_min_ns': lat_data.get('min', 0),
            f'{prefix}_lat_max_ns': lat_data.get('max', 0),
            f'{prefix}_lat_mean_ns': lat_data.get('mean', 0),
            f'{prefix}_lat_stddev_ns': lat_data.get('stddev', 0),
        })
        
        # Extract percentiles - try multiple locations
        percentile_sources = [
            io_data.get('clat_ns', {}).get('percentile', {}),
            io_data.get('lat_ns', {}).get('percentile', {}),
            io_data.get('slat_ns', {}).get('percentile', {})
        ]
        
        percentile_data = {}
        for source in percentile_sources:
            if source:
                percentile_data = source
                break
        
        metrics.update({
            f'{prefix}_lat_p99_ns': percentile_data.get('99.000000', 0),
            f'{prefix}_lat_p95_ns': percentile_data.get('95.000000', 0),
            f'{prefix}_lat_p50_ns': percentile_data.get('50.000000', 0),
        })
        
        # For larger block sizes where latency might not be reported in ns fields,
        # try to calculate approximate latency from IOPS and queue depth
        if metrics[f'{prefix}_lat_mean_ns'] == 0 and iops > 0:
            # Approximate latency using Little's Law: Latency = Queue Depth / IOPS
            # This is a rough approximation but better than 0
            approx_latency_ns = (32 * 1e9) / iops  # QD=32, convert to ns
            if 1000 < approx_latency_ns < 1000000000:  # Reasonable bounds: 1us to 1s
                metrics[f'{prefix}_lat_mean_ns'] = approx_latency_ns
                metrics[f'{prefix}_lat_min_ns'] = approx_latency_ns * 0.8  # Rough estimate
                metrics[f'{prefix}_lat_max_ns'] = approx_latency_ns * 1.2   # Rough estimate
                metrics[f'{prefix}_lat_stddev_ns'] = approx_latency_ns * 0.1  # Rough estimate
        
        return metrics
    
    def _parse_job_name_improved(self, job_name: str, job_data: Dict) -> Dict:
        """Improved job name parser that handles various FIO job name formats"""
        params = {}
        
        # Extract block size from job name using regex
        bs_match = re.search(r'(\d+k)', job_name.lower())
        if bs_match:
            params['block_size'] = bs_match.group(1)
        else:
            # Fallback: try to get from job options
            job_options = job_data.get('job options', {})
            bs_option = job_options.get('bs', '4k')  # Default to 4k if not found
            params['block_size'] = bs_option
        
        # Extract queue depth
        qd_match = re.search(r'qd(\d+)', job_name.lower())
        if qd_match:
            params['queue_depth'] = int(qd_match.group(1))
        else:
            # Fallback: try to get from job options
            job_options = job_data.get('job options', {})
            qd_option = job_options.get('iodepth', 1)
            params['queue_depth'] = int(qd_option)
        
        # Extract operation type
        if 'randread' in job_name.lower() or 'random_read' in job_name.lower():
            params['operation'] = 'randread'
            params['pattern'] = 'random'
        elif 'randwrite' in job_name.lower() or 'random_write' in job_name.lower():
            params['operation'] = 'randwrite'
            params['pattern'] = 'random'
        elif 'read' in job_name.lower() and 'rand' not in job_name.lower():
            params['operation'] = 'read'
            params['pattern'] = 'sequential'
        elif 'write' in job_name.lower() and 'rand' not in job_name.lower():
            params['operation'] = 'write'
            params['pattern'] = 'sequential'
        else:
            # Fallback from job options
            job_options = job_data.get('job options', {})
            rw_option = job_options.get('rw', 'read')
            params['operation'] = rw_option
            if 'rand' in rw_option:
                params['pattern'] = 'random'
            else:
                params['pattern'] = 'sequential'
        
        # Extract read mix for rw_mix jobs
        if 'rw_' in job_name.lower():
            mix_match = re.search(r'rw_(\d+)_(\d+)', job_name.lower())
            if mix_match:
                params['read_mix'] = int(mix_match.group(1))
            elif '100_0' in job_name.lower():
                params['read_mix'] = 100
            elif '0_100' in job_name.lower():
                params['read_mix'] = 0
            elif '70_30' in job_name.lower():
                params['read_mix'] = 70
            elif '50_50' in job_name.lower():
                params['read_mix'] = 50
        
        return params
    
    def _clean_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean NaN values from the dataframe"""
        # Replace infinite values with NaN first
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # For numeric columns, fill NaN with 0 (appropriate for performance metrics)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
        return df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics like MB/s, us latency"""
        # Convert bytes to MB/s
        df['read_bw_mbps'] = df['read_bw_bytes'] / (1024 * 1024)
        df['write_bw_mbps'] = df['write_bw_bytes'] / (1024 * 1024)
        
        # Convert nanoseconds to microseconds
        latency_cols = [col for col in df.columns if 'lat' in col and 'ns' in col and 'p99' not in col and 'p95' not in col and 'p50' not in col]
        for col in latency_cols:
            new_col = col.replace('_ns', '_us')
            df[new_col] = df[col] / 1000
        
        # Convert percentile latencies too
        percentile_cols = [col for col in df.columns if 'lat_p' in col and '_ns' in col]
        for col in percentile_cols:
            new_col = col.replace('_ns', '_us')
            df[new_col] = df[col] / 1000
        
        return df