import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

class PlotGenerator:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = sns.color_palette("husl", 8)
        Path('plots').mkdir(exist_ok=True)
        
    def generate_plots(self, df: pd.DataFrame, config_name: str):
        """Generate plots based on benchmark type with error handling"""
        try:
            if config_name == 'zero_queue':
                self._plot_zero_queue(df)
            elif config_name == 'block_size_sweep':
                self._plot_block_size_sweep(df)
            elif config_name == 'rw_mix_sweep':
                self._plot_rw_mix(df)
            elif config_name == 'queue_depth_sweep':
                self._plot_queue_depth_sweep(df)
            elif config_name == 'tail_latency':
                self._plot_tail_latency(df)
            else:
                print(f"Warning: No plot generator for config '{config_name}'")
        except Exception as e:
            print(f"Error generating plots for {config_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_zero_queue(self, df):
        """Plot zero queue depth results with safe column access"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        try:
            # Safely check for required columns
            required_cols = ['queue_depth', 'block_size', 'operation', 
                           'read_lat_mean_us', 'write_lat_mean_us', 
                           'read_bw_mbps', 'write_bw_mbps']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns for zero_queue plot: {missing_cols}")
                # Create a simple summary table instead
                ax1.axis('off')
                ax1.text(0.5, 0.5, f"Missing data columns:\n{', '.join(missing_cols)}", 
                        ha='center', va='center', transform=ax1.transAxes)
                return
            
            # Filter QD=1 results
            qd1_df = df[df['queue_depth'] == 1]
            
            if len(qd1_df) == 0:
                ax1.axis('off')
                ax1.text(0.5, 0.5, "No QD=1 data found", 
                        ha='center', va='center', transform=ax1.transAxes)
                return
            
            # 4K Random Read/Write Latency
            random_4k = qd1_df[qd1_df['block_size'] == '4k']
            if len(random_4k) > 0:
                operations = ['randread', 'randwrite']
                latencies = []
                for op in operations:
                    op_data = random_4k[random_4k['operation'] == op]
                    if len(op_data) > 0:
                        if 'read' in op:
                            latencies.append(op_data['read_lat_mean_us'].mean())
                        else:
                            latencies.append(op_data['write_lat_mean_us'].mean())
                    else:
                        latencies.append(0)
                
                ax1.bar(operations, latencies, color=self.colors[:2])
                ax1.set_ylabel('Latency (μs)')
                ax1.set_title('4K Random QD=1 Latency')
            else:
                ax1.axis('off')
                ax1.text(0.5, 0.5, "No 4K random data", 
                        ha='center', va='center', transform=ax1.transAxes)
            
            # 128K Sequential Read/Write Bandwidth
            seq_128k = qd1_df[qd1_df['block_size'] == '128k']
            if len(seq_128k) > 0:
                operations = ['read', 'write']
                bandwidths = []
                for op in operations:
                    op_data = seq_128k[seq_128k['operation'] == op]
                    if len(op_data) > 0:
                        if 'read' in op:
                            bandwidths.append(op_data['read_bw_mbps'].mean())
                        else:
                            bandwidths.append(op_data['write_bw_mbps'].mean())
                    else:
                        bandwidths.append(0)
                
                ax2.bar(operations, bandwidths, color=self.colors[2:4])
                ax2.set_ylabel('Bandwidth (MB/s)')
                ax2.set_title('128K Sequential QD=1 Bandwidth')
            else:
                ax2.axis('off')
                ax2.text(0.5, 0.5, "No 128K sequential data", 
                        ha='center', va='center', transform=ax2.transAxes)
            
            # Create a simple summary table
            ax3.axis('off')
            ax4.axis('off')
            
            summary_data = [['Workload', 'Avg Latency (μs)', 'Bandwidth (MB/s)']]
            
            # Collect available metrics
            for workload_name, bs, op in [('4K Random Read', '4k', 'randread'),
                                        ('4K Random Write', '4k', 'randwrite'),
                                        ('128K Seq Read', '128k', 'read'),
                                        ('128K Seq Write', '128k', 'write')]:
                workload_data = qd1_df[(qd1_df['block_size'] == bs) & (qd1_df['operation'] == op)]
                if len(workload_data) > 0:
                    if 'read' in op:
                        lat = workload_data['read_lat_mean_us'].mean()
                        bw = workload_data['read_bw_mbps'].mean()
                    else:
                        lat = workload_data['write_lat_mean_us'].mean()
                        bw = workload_data['write_bw_mbps'].mean()
                    summary_data.append([workload_name, f"{lat:.1f}", f"{bw:.1f}"])
                else:
                    summary_data.append([workload_name, "N/A", "N/A"])
            
            ax3.table(cellText=summary_data, cellLoc='center', loc='center', 
                     colWidths=[0.3, 0.3, 0.3])
            ax3.set_title('QD=1 Performance Summary')
            
            plt.tight_layout()
            plt.savefig('plots/zero_queue_baseline.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in _plot_zero_queue: {e}")
            # Create error plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('off')
            ax.text(0.5, 0.5, f"Plot Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            plt.savefig('plots/error_zero_queue.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_block_size_sweep(self, df):
        """Plot block size sweep results"""
        try:
            # Check required columns
            if 'block_size' not in df.columns:
                print("Warning: No 'block_size' column found for block_size_sweep plot")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Convert block size to numeric
            block_size_map = {'4k': 4, '16k': 16, '32k': 32, '64k': 64, 
                             '128k': 128, '256k': 256, '512k': 512, '1m': 1024}
            df['block_size_kb'] = df['block_size'].map(block_size_map)
            
            # Separate random and sequential
            random_mask = (df['pattern'] == 'random') | (df['job_name'].str.contains('random', na=False))
            sequential_mask = (df['pattern'] == 'sequential') | (df['job_name'].str.contains('sequential', na=False))
            
            random_df = df[random_mask]
            sequential_df = df[sequential_mask]
            
            # Random access - IOPS
            if len(random_df) > 0:
                random_grouped = random_df.groupby('block_size_kb').mean()
                ax1.plot(random_grouped.index, random_grouped['read_iops'] / 1000, 'o-', 
                        label='Random Read', linewidth=2)
                ax1.set_xlabel('Block Size (KB)')
                ax1.set_ylabel('IOPS (Thousands)')
                ax1.set_xscale('log')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_title('Random Access - IOPS vs Block Size')
            
            # Sequential access - Bandwidth
            if len(sequential_df) > 0:
                seq_grouped = sequential_df.groupby('block_size_kb').mean()
                ax2.plot(seq_grouped.index, seq_grouped['read_bw_mbps'], 's-', 
                        label='Sequential Read', linewidth=2, color='red')
                ax2.set_xlabel('Block Size (KB)')
                ax2.set_ylabel('Bandwidth (MB/s)')
                ax2.set_xscale('log')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_title('Sequential Access - Bandwidth vs Block Size')
            
            plt.tight_layout()
            plt.savefig('plots/block_size_sweep.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in _plot_block_size_sweep: {e}")
    
    def _plot_rw_mix(self, df):
        """Plot read/write mix results"""
        try:
            # Check required columns
            required_cols = ['read_mix', 'read_iops', 'write_iops', 'read_lat_mean_us', 'write_lat_mean_us']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns for rw_mix plot: {missing_cols}")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Group by read mix
            grouped = df.groupby('read_mix').mean()
            
            # Throughput
            mixes = grouped.index
            read_iops = grouped['read_iops'] / 1000
            write_iops = grouped['write_iops'] / 1000
            
            width = 10
            ax1.bar(mixes - width/2, read_iops, width=width, label='Read IOPS', alpha=0.7)
            ax1.bar(mixes + width/2, write_iops, width=width, label='Write IOPS', alpha=0.7)
            ax1.set_xlabel('Read Percentage (%)')
            ax1.set_ylabel('IOPS (Thousands)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title('IOPS by Read/Write Mix')
            
            # Latency
            read_lat = grouped['read_lat_mean_us']
            write_lat = grouped['write_lat_mean_us']
            
            ax2.plot(mixes, read_lat, 'o-', label='Read Latency', linewidth=2)
            ax2.plot(mixes, write_lat, 's-', label='Write Latency', linewidth=2)
            ax2.set_xlabel('Read Percentage (%)')
            ax2.set_ylabel('Latency (μs)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Latency by Read/Write Mix')
            
            plt.tight_layout()
            plt.savefig('plots/rw_mix_sweep.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in _plot_rw_mix: {e}")
    
    def _plot_queue_depth_sweep(self, df):
        """Plot queue depth scalability results"""
        try:
            # Check required columns
            required_cols = ['queue_depth', 'read_iops', 'read_lat_mean_us']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns for queue_depth_sweep plot: {missing_cols}")
                return
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            grouped = df.groupby('queue_depth').mean()
            
            # Throughput vs Queue Depth
            ax1.plot(grouped.index, grouped['read_iops'] / 1000, 'o-', linewidth=2)
            ax1.set_xlabel('Queue Depth')
            ax1.set_ylabel('IOPS (Thousands)')
            ax1.set_xscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Throughput Scalability')
            
            # Latency vs Queue Depth
            ax2.plot(grouped.index, grouped['read_lat_mean_us'], 's-', linewidth=2, color='red')
            ax2.set_xlabel('Queue Depth')
            ax2.set_ylabel('Latency (μs)')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Latency vs Queue Depth')
            
            # Throughput-Latency Trade-off
            ax3.plot(grouped['read_iops'] / 1000, grouped['read_lat_mean_us'], 'D-', 
                    linewidth=2, color='green')
            ax3.set_xlabel('Throughput (KIOPS)')
            ax3.set_ylabel('Latency (μs)')
            ax3.grid(True, alpha=0.3)
            ax3.set_title('Throughput-Latency Trade-off')
            
            # Annotate key points
            for qd in [1, 8, 32, 64]:
                if qd in grouped.index:
                    idx = grouped.index.get_loc(qd)
                    ax3.annotate(f'QD={qd}', 
                                (grouped['read_iops'].iloc[idx] / 1000, 
                                 grouped['read_lat_mean_us'].iloc[idx]),
                                xytext=(10, 10), textcoords='offset points')
            
            plt.tight_layout()
            plt.savefig('plots/queue_depth_sweep.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in _plot_queue_depth_sweep: {e}")
    
    def _plot_tail_latency(self, df):
        """Plot tail latency characterization results"""
        try:
            # Check required columns
            required_cols = ['queue_depth', 'read_lat_p50_us', 'read_lat_p95_us', 'read_lat_p99_us']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns for tail_latency plot: {missing_cols}")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Filter read workloads
            read_workloads = df[df['operation'].str.contains('read', na=False)]
            if len(read_workloads) == 0:
                print("No read workloads found for tail latency analysis")
                return
            
            qd_groups = read_workloads.groupby('queue_depth')
            
            # Latency percentiles vs queue depth
            percentiles = ['p50', 'p95', 'p99']
            colors = ['blue', 'orange', 'red']
            
            for i, percentile in enumerate(percentiles):
                latencies = []
                qds = []
                
                for qd, group in qd_groups:
                    if not group.empty:
                        lat_col = f'read_lat_{percentile}_us'
                        if lat_col in group.columns:
                            latencies.append(group[lat_col].mean())
                            qds.append(qd)
                
                if latencies:
                    ax1.plot(qds, latencies, 'o-', label=percentile, color=colors[i], linewidth=2)
            
            ax1.set_xlabel('Queue Depth')
            ax1.set_ylabel('Latency (μs)')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Tail Latency vs Queue Depth')
            
            # CDF-like distribution
            sample_qds = [1, 8, 32, 64]
            for qd in sample_qds:
                if qd in qd_groups.groups:
                    group = qd_groups.get_group(qd)
                    if not group.empty:
                        latencies = []
                        percentiles_to_plot = [50, 95, 99]
                        for p in percentiles_to_plot:
                            lat_col = f'read_lat_p{p}_us'
                            if lat_col in group.columns:
                                latencies.append(group[lat_col].mean())
                        
                        if latencies:
                            ax2.plot(percentiles_to_plot, latencies, 's-', label=f'QD={qd}', linewidth=2)
            
            ax2.set_xlabel('Percentile')
            ax2.set_ylabel('Latency (μs)')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Latency Distribution by Percentile')
            
            plt.tight_layout()
            plt.savefig('plots/tail_latency.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in _plot_tail_latency: {e}")