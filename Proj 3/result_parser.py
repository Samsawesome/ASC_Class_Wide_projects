import pandas as pd
import numpy as np
from typing import List, Dict
import re

class ResultParser:
    def __init__(self):
        self.latency_metrics = ['lat_ns', 'latency']
        self.throughput_metrics = ['iops', 'bw_bytes']
        
    def parse_results(self, results: List[Dict]) -> pd.DataFrame:
        """Parse FIO JSON results into pandas DataFrame"""
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
                
                # Extract read metrics with robust error handling
                read = job['read']
                
                # Handle different JSON structures for latency data
                read_lat_data = read.get('lat_ns', {})
                if not read_lat_data or read_lat_data.get('min', 0) == 0:
                    # Try completion latency instead
                    read_lat_data = read.get('clat_ns', {})
                
                row.update({
                    'read_iops': read.get('iops', 0),
                    'read_bw_bytes': read.get('bw_bytes', 0),
                    'read_lat_min_ns': read_lat_data.get('min', 0),
                    'read_lat_max_ns': read_lat_data.get('max', 0),
                    'read_lat_mean_ns': read_lat_data.get('mean', 0),
                    'read_lat_stddev_ns': read_lat_data.get('stddev', 0),
                })
                
                # Extract percentiles if they exist - check multiple locations
                clat_percentiles = read.get('clat_ns', {}).get('percentile', {})
                if not clat_percentiles:
                    clat_percentiles = read.get('lat_ns', {}).get('percentile', {})
                
                row.update({
                    'read_lat_p99_ns': clat_percentiles.get('99.000000', 0),
                    'read_lat_p95_ns': clat_percentiles.get('95.000000', 0),
                    'read_lat_p50_ns': clat_percentiles.get('50.000000', 0),
                })
                
                # Extract write metrics
                write = job['write']
                
                # Handle different JSON structures for write latency data
                write_lat_data = write.get('lat_ns', {})
                if not write_lat_data or write_lat_data.get('min', 0) == 0:
                    # Try completion latency instead
                    write_lat_data = write.get('clat_ns', {})
                
                row.update({
                    'write_iops': write.get('iops', 0),
                    'write_bw_bytes': write.get('bw_bytes', 0),
                    'write_lat_min_ns': write_lat_data.get('min', 0),
                    'write_lat_max_ns': write_lat_data.get('max', 0),
                    'write_lat_mean_ns': write_lat_data.get('mean', 0),
                    'write_lat_stddev_ns': write_lat_data.get('stddev', 0),
                })
                
                # Write percentiles
                write_clat_percentiles = write.get('clat_ns', {}).get('percentile', {})
                if not write_clat_percentiles:
                    write_clat_percentiles = write.get('lat_ns', {}).get('percentile', {})
                
                row.update({
                    'write_lat_p99_ns': write_clat_percentiles.get('99.000000', 0),
                    'write_lat_p95_ns': write_clat_percentiles.get('95.000000', 0),
                    'write_lat_p50_ns': write_clat_percentiles.get('50.000000', 0),
                })
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df = self._calculate_derived_metrics(df)
        df = self._clean_nan_values(df)
        return df
    
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