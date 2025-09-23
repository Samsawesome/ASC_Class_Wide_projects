import matplotlib.pyplot as plt
import numpy as np
import re
import os
import pandas as pd
from tabulate import tabulate
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import statistics
from scipy import stats

def get_distinct_colors(n_colors):
    """Generate distinct colors for plotting many lines"""
    # Use a colormap with good perceptual properties
    cmap = mpl.colormaps['tab20']
    colors = [cmap(i) for i in range(min(n_colors, 20))]
    
    # If we need more colors, use another colormap
    if n_colors > 20:
        cmap2 = mpl.colormaps['Set3']
        colors.extend([cmap2(i) for i in range(min(n_colors - 20, 12))])
    
    # If we still need more, generate colors manually
    if len(colors) < n_colors:
        additional_colors = [
            '#8B0000', '#006400', '#4B0082', '#FF8C00', '#8FBC8F',
            '#483D8B', '#2F4F4F', '#9400D3', '#FF1493', '#00CED1',
            '#696969', '#556B2F', '#9932CC', '#8B4513', '#2E8B57'
        ]
        colors.extend(additional_colors[:n_colors - len(colors)])
    
    return colors

def parse_latency():
    """Parse latency measurements with enhanced error handling"""
    levels, sizes, latencies = [], [], []
    
    try:
        with open('../results/raw_data/latency.txt') as f:
            for line in f:
                line = line.strip()
                if not line or 'Level:' not in line:
                    continue
                    
                try:
                    parts = line.split(',')
                    if len(parts) < 3:
                        continue
                        
                    level = parts[0].split(': ')[1].strip()
                    size_str = parts[1].split(': ')[1].replace('bytes', '').strip()
                    latency_str = parts[2].split(': ')[1].replace('ns', '').strip()
                    
                    # Convert size with proper handling
                    size = int(size_str)
                    latency = float(latency_str)
                    
                    # Validate data
                    if latency < 0 or size < 0:
                        print(f"Warning: Invalid data for {level}: latency={latency} ns, size={size} bytes")
                        continue
                    
                    levels.append(level)
                    sizes.append(size)
                    latencies.append(latency)
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line} - {e}")
                    continue
                    
    except FileNotFoundError:
        print("Error: latency.txt file not found")
        return [], [], []
    
    if not levels:
        print("Warning: No valid latency data found")
        
    return levels, sizes, latencies

def create_latency_table(levels, sizes, latencies_ns):
    """Create comprehensive latency table with proper formatting"""
    if not levels:
        return "No data available"
    
    # CPU frequency in GHz (adjust based on your system)
    cpu_freq_ghz = 3.69
    
    # Calculate latencies in cycles and add relative comparisons
    latencies_cycles = [lat_ns * cpu_freq_ghz for lat_ns in latencies_ns]
    
    # Calculate relative latencies (normalized to L1)
    if latencies_ns:
        l1_latency = latencies_ns[0] if 'L1' in levels[0] else min(latencies_ns)
        if l1_latency == 0:
            l1_latency = 0.0001 #outside of timer resolution
        relative_latencies = [lat_ns / l1_latency for lat_ns in latencies_ns]
    else:
        relative_latencies = []
    
    # Create detailed table data
    table_data = []
    for i, (level, size, latency_ns, latency_cycles, relative) in enumerate(
            zip(levels, sizes, latencies_ns, latencies_cycles, relative_latencies)):
        
        # Format size appropriately
        if size >= 1024 * 1024:
            size_str = f"{size/1024/1024:.1f} MB"
        elif size >= 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
            
        table_data.append([
            level, 
            size_str,
            f"{latency_ns:.2f} ns",
            f"{latency_cycles:.1f} cycles",
            f"{relative:.1f}x"
        ])
    
    # Create markdown table
    markdown_table = tabulate(table_data, 
                            headers=['Memory Level', 'Size', 'Latency (ns)', 
                                   'Latency (cycles)', 'Relative to L1'], 
                            tablefmt='pipe')
    
    # Save table to file
    try:
        with open('../results/plots/latency_table.md', 'w') as f:
            f.write("# Memory Hierarchy Latency Measurements\n\n")
            f.write(f"CPU Frequency: {cpu_freq_ghz} GHz\n\n")
            f.write(markdown_table)
        
        # Save CSV for analysis
        df = pd.DataFrame({
            'Memory_Level': levels,
            'Size_bytes': sizes,
            'Latency_ns': latencies_ns,
            'Latency_cycles': latencies_cycles,
            'Relative_to_L1': relative_latencies
        })
        df.to_csv('../results/plots/latency_table.csv', index=False)
        
    except Exception as e:
        print(f"Error saving table files: {e}")
    
    return markdown_table

def plot_latency():
    """Create enhanced latency visualization"""
    levels, sizes, latencies = parse_latency()
    
    if not levels:
        print("No data to plot")
        return
    
    # Create figure with subplots - FIX: Use constrained_layout and larger figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), constrained_layout=True)
    
    # FIX: Check for zero/negative latencies that can break plots
    valid_indices = [i for i, lat in enumerate(latencies) if lat > 0]
    if not valid_indices:
        print("Error: All latencies are zero or negative - cannot create plot")
        return
        
    # Filter out zero/negative latencies for plotting
    levels_plot = [levels[i] for i in valid_indices]
    sizes_plot = [sizes[i] for i in valid_indices]
    latencies_plot = [latencies[i] for i in valid_indices]
    
    # Plot 1: Regular bar plot with proper scaling
    bars = ax1.bar(levels_plot, latencies_plot, color=plt.cm.viridis(np.linspace(0, 1, len(levels_plot))))
    ax1.set_ylabel('Latency (ns)', fontsize=12, fontweight='bold')
    ax1.set_title('Memory Hierarchy Latency Measurements', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars with automatic positioning
    max_latency = max(latencies_plot) if latencies_plot else 1
    min_latency = min(latencies_plot) if latencies_plot else 0
    
    for i, (bar, latency) in enumerate(zip(bars, latencies_plot)):
        height = bar.get_height()
        # Position text above bar
        vertical_offset = max_latency * 0.02
        ax1.text(bar.get_x() + bar.get_width()/2., height + vertical_offset,
                f'{latency:.1f} ns', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add size information below bars
        size_kb = sizes_plot[i] / 1024
        if size_kb >= 1024:
            size_str = f'{size_kb/1024:.0f} MB'
        else:
            size_str = f'{size_kb:.0f} KB'
            
        ax1.text(bar.get_x() + bar.get_width()/2., -max_latency * 0.08,  # Reduced negative offset
                size_str, ha='center', va='top', fontsize=9, color='gray', style='italic')
    
    # Set y-axis limits with proper margins
    ax1.set_ylim(0, max_latency * 1.25)
    
    # Plot 2: Logarithmic scale to show hierarchy clearly
    # FIX: Ensure no zero values for log scale
    if min_latency > 0:
        bars2 = ax2.bar(levels_plot, latencies_plot, color=plt.cm.plasma(np.linspace(0, 1, len(levels_plot))))
        ax2.set_ylabel('Latency (ns) - Log Scale', fontsize=12, fontweight='bold')
        ax2.set_yscale('log')
        ax2.set_title('Memory Hierarchy Latency (Logarithmic Scale)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on log plot with better positioning
        for i, (level, latency) in enumerate(zip(levels_plot, latencies_plot)):
            ax2.text(i, latency * 1.5, f'{latency:.1f} ns', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Log scale not available\n(zero latencies present)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Memory Hierarchy Latency - Log Scale (Not Available)', fontsize=14)
    
    # FIX: Remove tight_layout when using constrained_layout
    # plt.tight_layout()  # COMMENTED OUT - using constrained_layout instead
    
    # Save plot with proper error handling
    try:
        # FIX: Ensure directory exists
        os.makedirs('../results/plots', exist_ok=True)
        
        # FIX: Save with different parameters
        plt.savefig('../results/plots/zero_latency_benchmark.png', 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        
        print("Plot saved successfully as '../results/plots/zero_latency_benchmark.png'")
        
    except Exception as e:
        print(f"Error saving plot: {e}")
        # Try alternative save method
        try:
            plt.savefig('../results/plots/zero_latency_benchmark.png', dpi=150)
            print("Plot saved with reduced quality due to initial error")
        except Exception as e2:
            print(f"Complete failure to save plot: {e2}")
    
    plt.close()
    
    # Create and display the table using original data (including zeros)
    table = create_latency_table(levels, sizes, latencies)
    
    print("\n" + "="*80)
    print("MEMORY HIERARCHY LATENCY ANALYSIS")
    print("="*80)
    print(f"Number of levels measured: {len(levels)}")
    print(f"Number of levels plotted: {len(levels_plot)}")
    print(f"Ratio (slowest/fastest): {max(latencies)/min(latencies):.1f}x")
    print("="*80)
    print(table)
    print("="*80)

def parse_bandwidth():
    """Parse bandwidth data with support for multiple runs and error bars"""
    data = {}
    current_stride = None
    current_ratio = None
    run_data = []
    
    with open('../results/raw_data/bandwidth.txt') as f:
        for line in f:
            if 'Stride:' in line:
                parts = line.split(',')
                stride = int(parts[0].split(': ')[1].replace('B', '').strip())
                read_ratio = float(parts[1].split(': ')[1].strip())
                bandwidth_str = parts[2].split(': ')[1].split(' ')[0].strip()
                bandwidth = float(bandwidth_str)
                
                # Create unique key for this configuration
                config_key = (stride, read_ratio)
                
                if config_key not in data:
                    data[config_key] = []
                
                data[config_key].append(bandwidth)
    
    # Reorganize data by read ratio with error bar information
    organized_data = {}
    for (stride, read_ratio), bandwidths in data.items():
        if read_ratio not in organized_data:
            organized_data[read_ratio] = {'strides': [], 'bandwidth_mean': [], 'bandwidth_std': []}
        
        # Calculate mean and standard deviation
        mean_bw = statistics.mean(bandwidths)
        std_bw = statistics.stdev(bandwidths) if len(bandwidths) > 1 else 0
        
        organized_data[read_ratio]['strides'].append(stride)
        organized_data[read_ratio]['bandwidth_mean'].append(mean_bw)
        organized_data[read_ratio]['bandwidth_std'].append(std_bw)
    
    return organized_data

def plot_bandwidth():
    data = parse_bandwidth()
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, (read_ratio, values) in enumerate(data.items()):
        # Sort by stride for proper x-axis ordering
        sorted_data = sorted(zip(values['strides'], values['bandwidth_mean'], values['bandwidth_std']))
        strides_sorted, mean_sorted, std_sorted = zip(*sorted_data)
        
        plt.errorbar(strides_sorted, mean_sorted, yerr=std_sorted,
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)], 
                    linewidth=2, markersize=8, capsize=5, capthick=2,
                    label=f'{int(read_ratio*100)}% Read')
    
    plt.xlabel('Stride (Bytes)')
    plt.ylabel('Bandwidth (MB/s)')
    plt.title('Memory Bandwidth vs Access Stride and Read/Write Ratio\n(with Error Bars from 3 Runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks([64, 256, 1024], ['64', '256', '1024'])
    
    plt.tight_layout()
    plt.savefig('../results/plots/pattern_stride_bandwidth.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional plot: Bandwidth vs Read Ratio for different strides
    plt.figure(figsize=(12, 8))
    
    # Reorganize data by stride
    stride_data = {}
    for read_ratio, values in data.items():
        for stride, mean_bw, std_bw in zip(values['strides'], values['bandwidth_mean'], values['bandwidth_std']):
            if stride not in stride_data:
                stride_data[stride] = {'read_ratios': [], 'bandwidth_mean': [], 'bandwidth_std': []}
            stride_data[stride]['read_ratios'].append(read_ratio)
            stride_data[stride]['bandwidth_mean'].append(mean_bw)
            stride_data[stride]['bandwidth_std'].append(std_bw)
    
    for i, (stride, values) in enumerate(stride_data.items()):
        # Sort by read ratio for proper plotting
        sorted_data = sorted(zip(values['read_ratios'], values['bandwidth_mean'], values['bandwidth_std']))
        read_ratios_sorted, mean_sorted, std_sorted = zip(*sorted_data)
        
        plt.errorbar(read_ratios_sorted, mean_sorted, yerr=std_sorted,
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)], 
                    linewidth=2, markersize=8, capsize=5, capthick=2,
                    label=f'Stride {stride}B')
    
    plt.xlabel('Read Ratio')
    plt.ylabel('Bandwidth (MB/s)')
    plt.title('Memory Bandwidth vs Read/Write Ratio for Different Strides\n(with Error Bars from 3 Runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks([0.0, 0.5, 0.7, 1.0], ['0% (100% Write)', '50%', '70%', '100% (100% Read)'])
    
    plt.tight_layout()
    plt.savefig('../results/plots/read_write_sweep_bandwidth.png', dpi=300, bbox_inches='tight')
    plt.close()

def parse_loaded_latency():
    """Parse loaded latency data with support for multiple runs and error bars"""
    data = {}
    current_config = None
    run_data = []
    
    with open('../results/raw_data/loaded_latency.txt') as f:
        for line in f:
            if 'Threads:' in line:
                parts = line.split(',')
                threads = int(parts[0].split(': ')[1].strip())
                stride = int(parts[1].split(': ')[1].replace('B', '').strip())
                throughput_str = parts[2].split(': ')[1].split(' ')[0].strip()
                latency_str = parts[3].split(': ')[1].split(' ')[0].strip()
                
                throughput = float(throughput_str)
                latency = float(latency_str)
                
                # Create unique key for this configuration
                config_key = (threads, stride)
                
                if config_key not in data:
                    data[config_key] = {'throughput': [], 'latency': []}
                
                data[config_key]['throughput'].append(throughput)
                data[config_key]['latency'].append(latency)
    
    # Reorganize data by stride
    organized_data = {}
    for (threads, stride), values in data.items():
        if stride not in organized_data:
            organized_data[stride] = {
                'threads': [], 
                'throughput_mean': [], 'throughput_std': [],
                'latency_mean': [], 'latency_std': []
            }
        
        # Calculate means and standard deviations
        throughput_mean = statistics.mean(values['throughput'])
        throughput_std = statistics.stdev(values['throughput']) if len(values['throughput']) > 1 else 0
        latency_mean = statistics.mean(values['latency'])
        latency_std = statistics.stdev(values['latency']) if len(values['latency']) > 1 else 0
        
        organized_data[stride]['threads'].append(threads)
        organized_data[stride]['throughput_mean'].append(throughput_mean)
        organized_data[stride]['throughput_std'].append(throughput_std)
        organized_data[stride]['latency_mean'].append(latency_mean)
        organized_data[stride]['latency_std'].append(latency_std)
    
    return organized_data

def plot_loaded_latency():
    data = parse_loaded_latency()
    
    plt.figure(figsize=(18, 5))
    
    # Throughput vs Threads
    plt.subplot(1, 3, 1)
    for stride, values in data.items():
        # Sort by threads for proper x-axis ordering
        sorted_data = sorted(zip(values['threads'], values['throughput_mean'], values['throughput_std']))
        threads_sorted, mean_sorted, std_sorted = zip(*sorted_data)
        
        plt.errorbar(threads_sorted, mean_sorted, yerr=std_sorted, 
                    marker='o', linewidth=2, markersize=6, capsize=5, capthick=2,
                    label=f'Stride {stride}B')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Throughput (MB/s)')
    plt.title('Throughput vs Concurrency\n(with Error Bars from 3 Runs)')
    plt.legend()
    plt.grid(True)
    
    # Latency vs Threads
    plt.subplot(1, 3, 2)
    for stride, values in data.items():
        # Sort by threads for proper x-axis ordering
        sorted_data = sorted(zip(values['threads'], values['latency_mean'], values['latency_std']))
        threads_sorted, mean_sorted, std_sorted = zip(*sorted_data)
        
        plt.errorbar(threads_sorted, mean_sorted, yerr=std_sorted,
                    marker='s', linewidth=2, markersize=6, capsize=5, capthick=2,
                    label=f'Stride {stride}B')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Latency (ns/op)')
    plt.title('Latency vs Concurrency\n(with Error Bars from 3 Runs)')
    plt.legend()
    plt.grid(True)
    
    # Throughput vs Latency curve for intensity sweep
    plt.subplot(1, 3, 3)
    for stride, values in data.items():
        # Sort by threads to maintain proper progression
        sorted_data = sorted(zip(values['threads'], values['throughput_mean'], values['latency_mean']))
        _, throughput_sorted, latency_sorted = zip(*sorted_data)
        
        plt.plot(latency_sorted, throughput_sorted, 'o-', linewidth=2, markersize=6,
                label=f'Stride {stride}B')
    
    plt.xlabel('Latency (ns/op)')
    plt.ylabel('Throughput (MB/s)')
    plt.title('Throughput vs Latency (Intensity Sweep)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../results/plots/intensity_sweep.png', dpi=300, bbox_inches='tight')
    plt.close()

def parse_pattern_latency():
    """Parse pattern latency data with support for multiple runs and error bars"""
    data = {}
    current_config = None
    run_data = []
    
    with open('../results/raw_data/pattern_latency.txt') as f:
        for line in f:
            if 'Stride:' in line:
                parts = line.split(',')
                stride = int(parts[0].split(': ')[1].replace('B', '').strip())
                read_ratio = float(parts[1].split(': ')[1].strip())
                latency_str = parts[2].split(': ')[1].split(' ')[0].strip()
                latency = float(latency_str)
                
                # Create unique key for this configuration
                config_key = (stride, read_ratio)
                
                if config_key not in data:
                    data[config_key] = []
                
                data[config_key].append(latency)
    
    # Reorganize data by read ratio with error bar information
    organized_data = {}
    for (stride, read_ratio), latencies in data.items():
        if read_ratio not in organized_data:
            organized_data[read_ratio] = {'strides': [], 'latency_mean': [], 'latency_std': []}
        
        # Calculate mean and standard deviation
        mean_lat = statistics.mean(latencies)
        std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        organized_data[read_ratio]['strides'].append(stride)
        organized_data[read_ratio]['latency_mean'].append(mean_lat)
        organized_data[read_ratio]['latency_std'].append(std_lat)
    
    return organized_data

def plot_pattern_latency():
    data = parse_pattern_latency()
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, (read_ratio, values) in enumerate(data.items()):
        # Sort by stride for proper x-axis ordering
        sorted_data = sorted(zip(values['strides'], values['latency_mean'], values['latency_std']))
        strides_sorted, mean_sorted, std_sorted = zip(*sorted_data)
        
        plt.errorbar(strides_sorted, mean_sorted, yerr=std_sorted,
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)], 
                    linewidth=2, markersize=8, capsize=5, capthick=2,
                    label=f'{int(read_ratio*100)}% Read')
    
    plt.xlabel('Stride (Bytes)')
    plt.ylabel('Latency (ns)')
    plt.title('Memory Latency vs Access Stride and Read/Write Ratio\n(with Error Bars from 3 Runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks([64, 256, 1024], ['64', '256', '1024'])
    
    plt.tight_layout()
    plt.savefig('../results/plots/pattern_stride_latency.png', dpi=300, bbox_inches='tight')
    plt.close()

def parse_kernel_bench():
    """Parse kernel benchmark data with support for multiple runs"""
    cache_data = {'size': [], 'stride': [], 'random': [], 'time': [], 'perf': []}
    tlb_data = {'size': [], 'huge_pages': [], 'time': [], 'perf': []}
    
    # Dictionary to collect multiple runs for each configuration
    cache_runs = {}
    tlb_runs = {}
    
    with open('../results/raw_data/kernel_bench.txt') as f:
        current_section = None
        for line in f:
            line = line.strip()
            if line.startswith('==='):
                if 'Cache Miss' in line:
                    current_section = 'cache'
                elif 'TLB Impact' in line:
                    current_section = 'tlb'
            elif line and current_section == 'cache' and 'Size:' in line:
                parts = line.split(',')
                size = int(parts[0].split(': ')[1].replace('B', '').strip())
                stride = int(parts[1].split(': ')[1].strip())
                random = parts[2].split(': ')[1].strip() == 'Yes'
                time = float(parts[3].split(': ')[1].replace('ns', '').strip())
                perf = float(parts[4].split(': ')[1].split(' ')[0].strip())
                
                # Create unique key for this configuration
                config_key = (size, stride, random)
                if config_key not in cache_runs:
                    cache_runs[config_key] = {'time': [], 'perf': []}
                
                cache_runs[config_key]['time'].append(time)
                cache_runs[config_key]['perf'].append(perf)
                
            elif line and current_section == 'tlb' and 'Size:' in line:
                parts = line.split(',')
                size = int(parts[0].split(': ')[1].replace('B', '').strip())
                huge_pages = parts[1].split(': ')[1].strip() == 'Yes'
                time = float(parts[2].split(': ')[1].replace('ns', '').strip())
                perf = float(parts[3].split(': ')[1].split(' ')[0].strip())
                
                # Create unique key for this configuration
                config_key = (size, huge_pages)
                if config_key not in tlb_runs:
                    tlb_runs[config_key] = {'time': [], 'perf': []}
                
                tlb_runs[config_key]['time'].append(time)
                tlb_runs[config_key]['perf'].append(perf)
    
    # Calculate means for cache data
    for (size, stride, random), runs in cache_runs.items():
        cache_data['size'].append(size)
        cache_data['stride'].append(stride)
        cache_data['random'].append(random)
        cache_data['time'].append(statistics.mean(runs['time']))
        cache_data['perf'].append(statistics.mean(runs['perf']))
    
    # Calculate means for TLB data
    for (size, huge_pages), runs in tlb_runs.items():
        tlb_data['size'].append(size)
        tlb_data['huge_pages'].append(huge_pages)
        tlb_data['time'].append(statistics.mean(runs['time']))
        tlb_data['perf'].append(statistics.mean(runs['perf']))
    
    return cache_data, tlb_data

def plot_kernel_bench():
    cache_data, tlb_data = parse_kernel_bench()
    
    if not cache_data['size']:
        print("No cache miss impact data found!")
        return
    
    # Plot cache miss impact
    plt.figure(figsize=(14, 8))
    
    # Group by size and access pattern
    unique_combinations = set()
    for i in range(len(cache_data['size'])):
        combo = (cache_data['size'][i], cache_data['random'][i])
        unique_combinations.add(combo)
    
    unique_combinations = sorted(unique_combinations, key=lambda x: (x[1], x[0]))
    
    # Get distinct colors for each line
    n_lines = len(unique_combinations)
    colors = get_distinct_colors(n_lines)
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_']
    line_styles = ['-', '--', '-.', ':']
    
    legend_handles = []
    legend_labels = []
    
    for i, (size, random_access) in enumerate(unique_combinations):
        indices = [idx for idx in range(len(cache_data['size'])) 
                  if cache_data['size'][idx] == size and cache_data['random'][idx] == random_access]
        
        if indices:
            strides = [cache_data['stride'][idx] for idx in indices]
            perf = [cache_data['perf'][idx] for idx in indices]
            
            # Sort by stride for proper line plotting
            sorted_data = sorted(zip(strides, perf))
            strides_sorted, perf_sorted = zip(*sorted_data)
            
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            line_style = line_styles[(i // len(markers)) % len(line_styles)]
            
            line, = plt.plot(strides_sorted, perf_sorted, 
                           marker=marker, 
                           color=color,
                           linestyle=line_style,
                           linewidth=2, 
                           markersize=6,
                           markeredgecolor='white',
                           markeredgewidth=0.5)
            
            legend_handles.append(line)
            access_type = "Random" if random_access else "Sequential"
            legend_labels.append(f"{size/1024:.0f}KB {access_type}")
    
    plt.xlabel('Stride (elements)', fontsize=12)
    plt.ylabel('Performance (operations/second)', fontsize=12)
    plt.title('Cache Miss Impact on SAXPY Performance', fontsize=14, fontweight='bold')
    
    # Create legend outside the plot if there are many lines
    if n_lines > 8:
        plt.legend(legend_handles, legend_labels, 
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left', 
                  fontsize=9,
                  ncol=2 if n_lines > 16 else 1)
        plt.subplots_adjust(right=0.7)
    else:
        plt.legend(legend_handles, legend_labels, fontsize=10)
    
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/plots/cache_miss_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional subplots for better analysis
    create_cache_subplots(cache_data)
    
    # Plot TLB impact
    plot_tlb_impact_individual()

def create_cache_subplots(cache_data):
    """Create additional subplots for cache analysis"""
    if not cache_data['size']:
        return
    
    # Get cache sizes from latency measurements for annotations
    levels, sizes, latencies = parse_latency()
    cache_sizes = {}
    for level, size in zip(levels, sizes):
        cache_sizes[level] = size
    
    plt.figure(figsize=(16, 12))
    
    # Subplot 1: Sequential access
    plt.subplot(2, 2, 1)
    plot_sequential_vs_random(cache_data, random=False, title='Sequential Access', cache_sizes=cache_sizes)
    
    # Subplot 2: Random access
    plt.subplot(2, 2, 2)
    plot_sequential_vs_random(cache_data, random=True, title='Random Access', cache_sizes=cache_sizes)
    
    # Subplot 3: Performance ratio (Random/Sequential)
    plt.subplot(2, 2, 3)
    plot_performance_ratio(cache_data, cache_sizes=cache_sizes)
    
    # Subplot 4: Best case performance for each size
    plt.subplot(2, 2, 4)
    plot_best_case_performance(cache_data, cache_sizes=cache_sizes)
    
    plt.tight_layout()
    plt.savefig('../results/plots/working-set_size_sweep.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sequential_vs_random(cache_data, random=False, title="", cache_sizes=None):
    """Plot performance for sequential or random access"""
    unique_sizes = sorted(set(cache_data['size']))
    unique_strides = sorted(set(cache_data['stride']))
    
    colors = get_distinct_colors(len(unique_strides))
    
    for i, stride in enumerate(unique_strides):
        sizes_for_stride = []
        perf_for_stride = []
        
        for size in unique_sizes:
            indices = [idx for idx in range(len(cache_data['size'])) 
                      if cache_data['size'][idx] == size and 
                         cache_data['stride'][idx] == stride and
                         cache_data['random'][idx] == random]
            if indices:
                sizes_for_stride.append(size)
                perf_for_stride.append(np.mean([cache_data['perf'][idx] for idx in indices]))
        
        if sizes_for_stride:
            # Sort by size for proper x-axis ordering
            sorted_data = sorted(zip(sizes_for_stride, perf_for_stride))
            sizes_sorted, perf_sorted = zip(*sorted_data)
            
            plt.plot(sizes_sorted, perf_sorted, 'o-', 
                    color=colors[i % len(colors)],
                    label=f'Stride {stride}',
                    linewidth=2, markersize=6)
    
    plt.xscale('log')
    plt.xlabel('Working Set Size (bytes)')
    plt.ylabel('Performance (ops/s)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    
    # Add cache boundaries if sizes are provided
    if cache_sizes:
        for level, size in cache_sizes.items():
            plt.axvline(x=size, color='gray', linestyle='--', alpha=0.7, label=f'{level} Cache')
        plt.legend(fontsize=8)

def plot_performance_ratio(cache_data, cache_sizes=None):
    """Plot performance ratio between random and sequential access"""
    unique_sizes = sorted(set(cache_data['size']))
    unique_strides = sorted(set(cache_data['stride']))
    
    colors = get_distinct_colors(len(unique_strides))
    
    for i, stride in enumerate(unique_strides):
        sizes_for_stride = []
        ratios = []
        
        for size in unique_sizes:
            # Get sequential performance
            seq_indices = [idx for idx in range(len(cache_data['size'])) 
                          if cache_data['size'][idx] == size and 
                             cache_data['stride'][idx] == stride and
                             not cache_data['random'][idx]]
            # Get random performance
            rand_indices = [idx for idx in range(len(cache_data['size'])) 
                           if cache_data['size'][idx] == size and 
                              cache_data['stride'][idx] == stride and
                              cache_data['random'][idx]]
            
            if seq_indices and rand_indices:
                seq_perf = np.mean([cache_data['perf'][idx] for idx in seq_indices])
                rand_perf = np.mean([cache_data['perf'][idx] for idx in rand_indices])
                if seq_perf > 0:
                    sizes_for_stride.append(size)
                    ratios.append(rand_perf / seq_perf)
        
        if sizes_for_stride:
            # Sort by size for proper x-axis ordering
            sorted_data = sorted(zip(sizes_for_stride, ratios))
            sizes_sorted, ratios_sorted = zip(*sorted_data)
            
            plt.plot(sizes_sorted, ratios_sorted, 's-', 
                    color=colors[i % len(colors)],
                    label=f'Stride {stride}',
                    linewidth=2, markersize=6)
    
    plt.xscale('log')
    plt.xlabel('Working Set Size (bytes)')
    plt.ylabel('Random/Sequential Performance Ratio')
    plt.title('Performance Degradation from Random Access')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    plt.legend(fontsize=8)
    
    # Add cache boundaries if sizes are provided
    if cache_sizes:
        for level, size in cache_sizes.items():
            plt.axvline(x=size, color='gray', linestyle='--', alpha=0.7, label=f'{level} Cache')
        plt.legend(fontsize=8)

def plot_best_case_performance(cache_data, cache_sizes=None):
    """Plot best-case performance for each working set size"""
    unique_sizes = sorted(set(cache_data['size']))
    
    best_seq_perf = []
    best_rand_perf = []
    
    for size in unique_sizes:
        # Best sequential performance (min stride)
        seq_indices = [idx for idx in range(len(cache_data['size'])) 
                      if cache_data['size'][idx] == size and not cache_data['random'][idx]]
        if seq_indices:
            best_seq = max([cache_data['perf'][idx] for idx in seq_indices])
            best_seq_perf.append(best_seq)
        else:
            best_seq_perf.append(0)
        
        # Best random performance (min stride)
        rand_indices = [idx for idx in range(len(cache_data['size'])) 
                       if cache_data['size'][idx] == size and cache_data['random'][idx]]
        if rand_indices:
            best_rand = max([cache_data['perf'][idx] for idx in rand_indices])
            best_rand_perf.append(best_rand)
        else:
            best_rand_perf.append(0)
    
    plt.plot(unique_sizes, best_seq_perf, 'o-', label='Best Sequential', linewidth=2)
    plt.plot(unique_sizes, best_rand_perf, 's-', label='Best Random', linewidth=2)
    
    plt.xscale('log')
    plt.xlabel('Working Set Size (bytes)')
    plt.ylabel('Best Performance (ops/s)')
    plt.title('Best-Case Performance by Working Set Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add cache boundaries if sizes are provided
    if cache_sizes:
        for level, size in cache_sizes.items():
            plt.axvline(x=size, color='gray', linestyle='--', alpha=0.7, label=f'{level} Cache')
        plt.legend(fontsize=8)

def parse_tlb_impact():
    sizes, huge_pages, times, perfs = [], [], [], []
    with open('../results/raw_data/kernel_bench.txt') as f:
        current_section = None
        for line in f:
            line = line.strip()
            if '=== TLB Impact Tests ===' in line:
                current_section = 'tlb'
            elif current_section == 'tlb' and 'Size:' in line and 'Huge Pages:' in line:
                parts = line.split(',')
                size = int(parts[0].split(': ')[1].replace('B', '').strip())
                huge_page = parts[1].split(': ')[1].strip() == 'Yes'
                time = float(parts[2].split(': ')[1].replace('ns', '').strip())
                perf = float(parts[3].split(': ')[1].split(' ')[0].strip())
                
                sizes.append(size)
                huge_pages.append(huge_page)
                times.append(time)
                perfs.append(perf)
    return sizes, huge_pages, times, perfs

def plot_tlb_impact_individual():
    sizes, huge_pages, times, perfs = parse_tlb_impact()
    
    if not sizes:
        print("No TLB impact data found!")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Group by page type and sort by size
    regular_data = sorted([(size, perf) for size, perf, hp in zip(sizes, perfs, huge_pages) if not hp])
    huge_data = sorted([(size, perf) for size, perf, hp in zip(sizes, perfs, huge_pages) if hp])
    
    if regular_data:
        regular_sizes, regular_perfs = zip(*regular_data)
        plt.plot(regular_sizes, regular_perfs, 'o-', linewidth=2, markersize=8, 
                label='Regular Pages (4KB)', color='#E74C3C')
    if huge_data:
        huge_sizes, huge_perfs = zip(*huge_data)
        plt.plot(huge_sizes, huge_perfs, 's-', linewidth=2, markersize=8,
                label='Huge Pages (2MB)', color='#3498DB')
    
    plt.xlabel('Working Set Size (bytes)', fontsize=12)
    plt.ylabel('Performance (operations/second)', fontsize=12)
    plt.title('TLB Impact on SAXPY Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for cache boundaries
    plt.axvline(x=32*1024, color='gray', linestyle='--', alpha=0.7, label='L1 Cache (~32KB)')
    plt.axvline(x=256*1024, color='gray', linestyle='-.', alpha=0.7, label='L2 Cache (~256KB)')
    plt.axvline(x=8*1024*1024, color='gray', linestyle=':', alpha=0.7, label='L3 Cache (~8MB)')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('../results/plots/tlb_miss_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Create plots directory if it doesn't exist
    os.makedirs('../results/plots', exist_ok=True)
    
    # Install tabulate if not present
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate for table formatting...")
        os.system("pip install tabulate")
        from tabulate import tabulate
    
    try:
        plot_latency()
        print("Latency plot and table created successfully")
    except Exception as e:
        print(f"Error creating latency plot: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        plot_bandwidth()
        print("Bandwidth plot created successfully")
    except Exception as e:
        print(f"Error creating bandwidth plot: {e}")
    
    try:
        plot_loaded_latency()
        print("Loaded latency plot created successfully")
    except Exception as e:
        print(f"Error creating loaded latency plot: {e}")
    
    try:
        plot_pattern_latency()
        print("Pattern latency plot created successfully")
    except Exception as e:
        print(f"Error creating pattern latency plot: {e}")
    
    try:
        plot_tlb_impact_individual()
        print("TLB impact plot created successfully")
    except Exception as e:
        print(f"Error creating TLB impact plot: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        plot_kernel_bench()
        print("Kernel benchmark plots created successfully")
    except Exception as e:
        print(f"Error creating kernel benchmark plots: {e}")
    
    print("All plots completed!")