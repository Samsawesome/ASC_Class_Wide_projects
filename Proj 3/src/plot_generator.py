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
                self._plot_block_size_sweep_detailed(df)
            elif config_name == 'rw_mix_sweep':
                self._plot_rw_mix(df)
                self._plot_rw_mix_detailed(df)
            elif config_name == 'queue_depth_sweep':
                self._plot_queue_depth_sweep(df)
                self._plot_queue_depth_tradeoff(df)
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
                # Ensure numeric columns for grouping
                numeric_cols = random_df.select_dtypes(include=[np.number]).columns
                random_grouped = random_df.groupby('block_size_kb')[numeric_cols].mean()
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
                # Ensure numeric columns for grouping
                numeric_cols = sequential_df.select_dtypes(include=[np.number]).columns
                seq_grouped = sequential_df.groupby('block_size_kb')[numeric_cols].mean()
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
    
    def _extract_read_mix_from_jobnames(self, df):
        """Extract read mix from job names as fallback"""
        df_copy = df.copy()
        
        def extract_mix(job_name):
            if job_name is None:
                return None
            job_lower = job_name.lower()
            if 'rw_100_0' in job_lower:
                return 100
            elif 'rw_0_100' in job_lower:
                return 0
            elif 'rw_70_30' in job_lower:
                return 70
            elif 'rw_50_50' in job_lower:
                return 50
            elif 'seq_rw_100_0' in job_lower:
                return 100
            elif 'seq_rw_0_100' in job_lower:
                return 0
            elif 'seq_rw_70_30' in job_lower:
                return 70
            elif 'seq_rw_50_50' in job_lower:
                return 50
            return None
        
        df_copy['read_mix'] = df_copy['job_name'].apply(extract_mix)
        return df_copy

    def _plot_rw_mix(self, df):
        """Plot read/write mix results with NaN handling"""
        try:
            # Check required columns
            required_cols = ['read_iops', 'write_iops', 'read_lat_mean_us', 'write_lat_mean_us']
            
            # If read_mix column exists, use it, otherwise try to extract from job names
            if 'read_mix' not in df.columns:
                # Try to extract read_mix from job names
                df_with_mix = self._extract_read_mix_from_jobnames(df)
                if 'read_mix' in df_with_mix.columns:
                    df = df_with_mix
                else:
                    print("Warning: No 'read_mix' column found for rw_mix plot")
                    return
            
            # Clean data - remove rows with NaN in required columns
            df_clean = df.dropna(subset=required_cols)
            if len(df_clean) == 0:
                print("Warning: No valid data after cleaning NaN values")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Group by read mix - ensure numeric columns and handle NaN
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            grouped = df_clean.groupby('read_mix')[numeric_cols].mean()
            
            # Clean grouped data - replace any remaining NaN with 0 for plotting
            grouped = grouped.fillna(0)
            
            # Throughput
            mixes = grouped.index
            read_iops = grouped['read_iops'] / 1000
            write_iops = grouped['write_iops'] / 1000
            
            # Check if we have valid data to plot
            if len(mixes) == 0 or (read_iops.sum() == 0 and write_iops.sum() == 0):
                print("Warning: No valid throughput data for RW mix plot")
                return
            
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
            
            # Only plot if we have valid latency data
            valid_latency = (read_lat.sum() > 0 or write_lat.sum() > 0)
            if valid_latency:
                ax2.plot(mixes, read_lat, 'o-', label='Read Latency', linewidth=2)
                ax2.plot(mixes, write_lat, 's-', label='Write Latency', linewidth=2)
                
                ax2.set_xlabel('Read Percentage (%)')
                ax2.set_ylabel('Latency (μs)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_title('Latency by Read/Write Mix')
            else:
                ax2.text(0.5, 0.5, 'No latency data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Latency by Read/Write Mix (No Data)')
            
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
            
            # Group by queue depth - ensure numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            grouped = df.groupby('queue_depth')[numeric_cols].mean()
            
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

    def _plot_block_size_sweep_detailed(self, df):
        """Enhanced block size sweep with IOPS/MB/s and latency on dual axes"""
        try:
            if 'block_size' not in df.columns:
                print("Warning: No 'block_size' column found for detailed block_size_sweep plot")
                return
            
            # Convert block size to numeric
            block_size_map = {'4k': 4, '16k': 16, '32k': 32, '64k': 64, 
                             '128k': 128, '256k': 256, '512k': 512, '1m': 1024}
            df['block_size_kb'] = df['block_size'].map(block_size_map)
            
            # Create separate figures for random and sequential
            patterns = ['random', 'sequential']
            
            for pattern in patterns:
                if pattern == 'random':
                    pattern_mask = (df['pattern'] == 'random') | (df['job_name'].str.contains('random', na=False))
                    perf_metric = 'read_iops'
                    perf_label = 'IOPS'
                    perf_scale = 1  # Keep as IOPS
                    title_suffix = 'Random'
                else:
                    pattern_mask = (df['pattern'] == 'sequential') | (df['job_name'].str.contains('sequential', na=False))
                    perf_metric = 'read_bw_mbps'
                    perf_label = 'Bandwidth (MB/s)'
                    perf_scale = 1  # Keep as MB/s
                    title_suffix = 'Sequential'
                
                pattern_df = df[pattern_mask]
                if len(pattern_df) == 0:
                    continue
                
                # Group by block size - ensure numeric columns
                numeric_cols = pattern_df.select_dtypes(include=[np.number]).columns
                grouped = pattern_df.groupby('block_size_kb')[numeric_cols].mean()
                
                # Create figure with dual y-axes
                fig, ax1 = plt.subplots(figsize=(12, 8))
                
                # Plot throughput (IOPS or MB/s)
                color1 = 'tab:blue'
                if pattern == 'random':
                    throughput = grouped[perf_metric] / 1000  # Convert to KIOPS
                    ax1.set_ylabel('IOPS (Thousands)', color=color1)
                else:
                    throughput = grouped[perf_metric]
                    ax1.set_ylabel('Bandwidth (MB/s)', color=color1)
                
                ax1.plot(grouped.index, throughput, 'o-', color=color1, linewidth=3, 
                        markersize=8, label=f'{title_suffix} {perf_label}')
                ax1.tick_params(axis='y', labelcolor=color1)
                ax1.set_xlabel('Block Size (KB)')
                ax1.set_xscale('log')
                ax1.grid(True, alpha=0.3)
                
                # Create second y-axis for latency
                ax2 = ax1.twinx()
                color2 = 'tab:red'
                latency = grouped['read_lat_mean_us']
                ax2.plot(grouped.index, latency, 's--', color=color2, linewidth=2, 
                        markersize=6, label='Latency')
                ax2.set_ylabel('Latency (μs)', color=color2)
                ax2.tick_params(axis='y', labelcolor=color2)
                
                # Mark the transition point (64KB)
                transition_point = 64
                if transition_point in grouped.index:
                    ax1.axvline(x=transition_point, color='gray', linestyle=':', alpha=0.7, 
                               label=f'IOPS/BW Transition ({transition_point}KB)')
                    # Annotate the transition
                    y_max = max(throughput.max(), latency.max()/100)
                    ax1.annotate('IOPS-dominated', xy=(transition_point/2, y_max*0.8), 
                                xytext=(transition_point/3, y_max*0.9),
                                arrowprops=dict(arrowstyle='->', alpha=0.7),
                                ha='center')
                    ax1.annotate('BW-dominated', xy=(transition_point*2, y_max*0.8), 
                                xytext=(transition_point*3, y_max*0.9),
                                arrowprops=dict(arrowstyle='->', alpha=0.7),
                                ha='center')
                
                # Add legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                plt.title(f'Block Size Sweep - {title_suffix} Access\nThroughput vs Latency Trade-off')
                plt.tight_layout()
                plt.savefig(f'plots/block_size_sweep_{pattern}_detailed.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Error in _plot_block_size_sweep_detailed: {e}")

    def _plot_rw_mix_detailed(self, df):
        """Enhanced read/write mix analysis with comprehensive metrics and NaN handling"""
        try:
            # Check required columns
            required_cols = ['read_iops', 'write_iops', 'read_lat_mean_us', 'write_lat_mean_us']
            
            # If read_mix column exists, use it, otherwise try to extract from job names
            if 'read_mix' not in df.columns:
                # Try to extract read_mix from job names
                df_with_mix = self._extract_read_mix_from_jobnames(df)
                if 'read_mix' in df_with_mix.columns:
                    df = df_with_mix
                else:
                    print("Warning: No 'read_mix' column found for detailed rw_mix plot")
                    return
            
            # Clean data - remove rows with NaN in required columns
            df_clean = df.dropna(subset=required_cols)
            if len(df_clean) == 0:
                print("Warning: No valid data after cleaning NaN values")
                return
            
            # Group by read mix - ensure numeric columns and handle NaN
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            grouped = df_clean.groupby('read_mix')[numeric_cols].mean()
            
            # Clean grouped data - replace any remaining NaN with 0 for plotting
            grouped = grouped.fillna(0)
            mixes = grouped.index
            
            # Check if we have enough valid data
            if len(mixes) == 0:
                print("Warning: No valid data points for detailed RW mix plot")
                return
            
            # Create comprehensive plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Panel 1: Throughput breakdown
            read_iops = grouped['read_iops'] / 1000
            write_iops = grouped['write_iops'] / 1000
            total_iops = (grouped['read_iops'] + grouped['write_iops']) / 1000
            
            # Check if we have valid throughput data
            has_throughput_data = (read_iops.sum() > 0 or write_iops.sum() > 0)
            
            if has_throughput_data and len(mixes) > 0:
                width = 0.25
                x = np.arange(len(mixes))
                
                ax1.bar(x - width, read_iops, width, label='Read IOPS', alpha=0.8, color='blue')
                ax1.bar(x, write_iops, width, label='Write IOPS', alpha=0.8, color='red')
                ax1.bar(x + width, total_iops, width, label='Total IOPS', alpha=0.8, color='green')
                
                ax1.set_xlabel('Read Percentage (%)')
                ax1.set_ylabel('IOPS (Thousands)')
                ax1.set_xticks(x)
                ax1.set_xticklabels(mixes)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_title('Throughput by Read/Write Mix')
            else:
                ax1.text(0.5, 0.5, 'No throughput data available', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Throughput by Read/Write Mix (No Data)')
            
            # Panel 2: Latency comparison
            read_lat = grouped['read_lat_mean_us']
            write_lat = grouped['write_lat_mean_us']
            
            # Check if we have valid latency data
            has_latency_data = (read_lat.sum() > 0 or write_lat.sum() > 0)
            
            if has_latency_data and len(mixes) > 0:
                ax2.plot(mixes, read_lat, 'o-', label='Read Latency', linewidth=2, markersize=8, color='blue')
                ax2.plot(mixes, write_lat, 's-', label='Write Latency', linewidth=2, markersize=8, color='red')
                
                ax2.set_xlabel('Read Percentage (%)')
                ax2.set_ylabel('Latency (μs)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_title('Latency by Read/Write Mix')
            else:
                ax2.text(0.5, 0.5, 'No latency data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Latency by Read/Write Mix (No Data)')
            
            # Panel 3: Performance asymmetry (only if we have both read and write data)
            has_both_rw_data = has_throughput_data and has_latency_data and len(mixes) > 0
            
            if has_both_rw_data:
                # Calculate ratios, avoiding division by zero
                read_write_ratio = np.divide(grouped['read_iops'], grouped['write_iops'], 
                                           out=np.zeros_like(grouped['read_iops']), 
                                           where=grouped['write_iops'] != 0)
                
                latency_ratio = np.divide(grouped['write_lat_mean_us'], grouped['read_lat_mean_us'], 
                                        out=np.zeros_like(grouped['write_lat_mean_us']), 
                                        where=grouped['read_lat_mean_us'] != 0)
                
                ax3.bar(mixes - 0.2, read_write_ratio, 0.4, label='R/W Throughput Ratio', alpha=0.7, color='orange')
                ax3.bar(mixes + 0.2, latency_ratio, 0.4, label='W/R Latency Ratio', alpha=0.7, color='purple')
                
                ax3.set_xlabel('Read Percentage (%)')
                ax3.set_ylabel('Ratio')
                ax3.set_xticks(mixes)
                ax3.set_xticklabels(mixes)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.set_title('Read/Write Performance Asymmetry')
                ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            else:
                ax3.text(0.5, 0.5, 'Insufficient data for\nperformance asymmetry analysis', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Read/Write Performance Asymmetry (No Data)')
            
            # Panel 4: Summary table
            ax4.axis('off')
            table_data = [['Mix', 'Read IOPS', 'Write IOPS', 'Read Lat(μs)', 'Write Lat(μs)']]
            
            if has_both_rw_data:
                for mix in mixes:
                    row_data = [
                        f"{mix}%/{100-mix}%",
                        f"{read_iops.loc[mix]:.1f}K" if read_iops.loc[mix] > 0 else "N/A",
                        f"{write_iops.loc[mix]:.1f}K" if write_iops.loc[mix] > 0 else "N/A",
                        f"{read_lat.loc[mix]:.1f}" if read_lat.loc[mix] > 0 else "N/A",
                        f"{write_lat.loc[mix]:.1f}" if write_lat.loc[mix] > 0 else "N/A"
                    ]
                    table_data.append(row_data)
                
                table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                                 colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                ax4.set_title('Performance Summary')
            else:
                ax4.text(0.5, 0.5, 'No summary data available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Performance Summary (No Data)')
            
            plt.tight_layout()
            plt.savefig('plots/rw_mix_sweep_detailed.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in _plot_rw_mix_detailed: {e}")

    def _plot_queue_depth_tradeoff(self, df):
        """Throughput-latency trade-off curve with knee point identification"""
        try:
            # Check required columns
            required_cols = ['queue_depth', 'read_iops', 'read_lat_mean_us']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns for queue_depth trade-off plot: {missing_cols}")
                return
            
            # Group by queue depth - ensure numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            grouped = df.groupby('queue_depth')[numeric_cols].mean()
            
            if len(grouped) < 3:
                print("Insufficient data points for trade-off analysis")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Panel 1: Throughput vs Latency scatter with curve
            throughput = grouped['read_iops'] / 1000  # Convert to KIOPS
            latency = grouped['read_lat_mean_us']
            
            # Plot the trade-off curve
            ax1.plot(throughput, latency, 'o-', linewidth=3, markersize=10, 
                    color='green', alpha=0.7, label='Trade-off Curve')
            
            # Annotate each point with QD value
            for qd, (tp, lat) in zip(grouped.index, zip(throughput, latency)):
                ax1.annotate(f'QD={qd}', (tp, lat), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.8)
            
            ax1.set_xlabel('Throughput (KIOPS)')
            ax1.set_ylabel('Latency (μs)')
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Throughput-Latency Trade-off Curve')
            ax1.legend()
            
            # Panel 2: Identify knee point and show analysis
            knee_point = None
            if len(throughput) >= 5:
                knee_point = self._find_knee_point(throughput.values, latency.values)
                if knee_point is not None:
                    knee_tp, knee_lat = knee_point
                    
                    # Plot with knee point highlighted
                    ax2.plot(throughput, latency, 'o-', linewidth=2, markersize=8, 
                            color='blue', alpha=0.7, label='Performance Curve')
                    ax2.plot(knee_tp, knee_lat, 's', markersize=12, color='red', 
                            label=f'Knee Point (~{knee_tp:.1f} KIOPS)')
                    
                    # Add regions
                    low_qd_mask = throughput <= knee_tp
                    high_qd_mask = throughput > knee_tp
                    
                    if any(low_qd_mask):
                        ax2.fill_between(throughput[low_qd_mask], latency[low_qd_mask], 
                                       alpha=0.2, color='green', label='Optimal Region')
                    if any(high_qd_mask):
                        ax2.fill_between(throughput[high_qd_mask], latency[high_qd_mask], 
                                       alpha=0.2, color='orange', label='Saturated Region')
                    
                    # Little's Law analysis
                    optimal_qd_idx = throughput[low_qd_mask].index[-1] if any(low_qd_mask) else None
                    if optimal_qd_idx is not None:
                        optimal_throughput = throughput.loc[optimal_qd_idx] * 1000  # Convert back to IOPS
                        optimal_latency = latency.loc[optimal_qd_idx] / 1e6  # Convert to seconds
                        littles_law_qd = optimal_throughput * optimal_latency
                        
                        # Add Little's Law information
                        ax2.text(0.05, 0.95, f'Knee Point Analysis:\n'
                                f'Throughput: {knee_tp:.1f} KIOPS\n'
                                f'Latency: {knee_lat:.1f} μs\n'
                                f"Little's Law QD: {littles_law_qd:.1f}",
                                transform=ax2.transAxes, fontsize=10,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    ax2.set_xlabel('Throughput (KIOPS)')
                    ax2.set_ylabel('Latency (μs)')
                    ax2.grid(True, alpha=0.3)
                    ax2.set_title('Knee Point Analysis')
                    ax2.legend()
            
            plt.tight_layout()
            plt.savefig('plots/queue_depth_tradeoff_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in _plot_queue_depth_tradeoff: {e}")

    def _find_knee_point(self, throughput, latency):
        """Find the knee point in throughput-latency curve using curvature method"""
        try:
            if len(throughput) < 3:
                return None
            
            # Simple method: find point where latency starts increasing rapidly
            # Calculate the slope of latency increase
            if len(throughput) >= 3:
                # Find the point where the latency increase is maximum relative to throughput
                lat_increases = np.diff(latency) / np.diff(throughput)
                if len(lat_increases) > 0:
                    knee_idx = np.argmax(lat_increases) + 1
                    if knee_idx < len(throughput):
                        return throughput[knee_idx], latency[knee_idx]
            return None
        except:
            return None