import matplotlib.pyplot as plt
import numpy as np
import re
import os

def parse_latency():
    sizes, latencies = [], []
    with open('results/raw_data/latency.txt') as f:
        for line in f:
            if 'Size:' in line:
                parts = line.split(',')
                # Extract size number and remove "bytes" text
                size_str = parts[0].split(': ')[1].replace(' bytes', '').strip()
                latency_str = parts[1].split(': ')[1].replace(' ns', '').strip()
                
                sizes.append(int(size_str))
                latencies.append(float(latency_str))
    return sizes, latencies

def plot_latency():
    sizes, latencies = parse_latency()
    labels = ['L1', 'L2', 'L3', 'DRAM']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, latencies)
    plt.ylabel('Latency (ns)')
    plt.title('Memory Hierarchy Latency')
    
    # Add value labels on top of bars
    for i, (label, latency) in enumerate(zip(labels, latencies)):
        plt.text(i, latency + 0.1, f'{latency:.2f} ns', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.savefig('results/plots/latency.png')
    plt.close()

def parse_bandwidth():
    data = {}
    with open('results/raw_data/bandwidth.txt') as f:
        for line in f:
            if 'Stride:' in line:
                parts = line.split(',')
                stride = int(parts[0].split(': ')[1].replace('B', '').strip())
                read_ratio = float(parts[1].split(': ')[1].strip())
                bandwidth_str = parts[2].split(': ')[1].split(' ')[0].strip()
                
                if read_ratio not in data:
                    data[read_ratio] = {'strides': [], 'bandwidth': []}
                
                data[read_ratio]['strides'].append(stride)
                data[read_ratio]['bandwidth'].append(float(bandwidth_str))
    return data

def plot_bandwidth():
    data = parse_bandwidth()
    
    plt.figure(figsize=(12, 8))
    
    # Bandwidth vs Stride for different read ratios
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, (read_ratio, values) in enumerate(data.items()):
        plt.plot(values['strides'], values['bandwidth'], 
                marker=markers[i % len(markers)], 
                color=colors[i % len(colors)], 
                linewidth=2, markersize=8,
                label=f'{int(read_ratio*100)}% Read')
    
    plt.xlabel('Stride (Bytes)')
    plt.ylabel('Bandwidth (MB/s)')
    plt.title('Memory Bandwidth vs Access Stride and Read/Write Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks([64, 256, 1024], ['64', '256', '1024'])
    
    plt.tight_layout()
    plt.savefig('results/plots/bandwidth.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional plot: Bandwidth vs Read Ratio for different strides
    plt.figure(figsize=(12, 8))
    
    # Reorganize data by stride
    stride_data = {}
    for read_ratio, values in data.items():
        for stride, bandwidth in zip(values['strides'], values['bandwidth']):
            if stride not in stride_data:
                stride_data[stride] = {'read_ratios': [], 'bandwidth': []}
            stride_data[stride]['read_ratios'].append(read_ratio)
            stride_data[stride]['bandwidth'].append(bandwidth)
    
    for i, (stride, values) in enumerate(stride_data.items()):
        # Sort by read ratio for proper plotting
        sorted_data = sorted(zip(values['read_ratios'], values['bandwidth']))
        read_ratios_sorted, bandwidth_sorted = zip(*sorted_data)
        
        plt.plot(read_ratios_sorted, bandwidth_sorted, 
                marker=markers[i % len(markers)], 
                color=colors[i % len(colors)], 
                linewidth=2, markersize=8,
                label=f'Stride {stride}B')
    
    plt.xlabel('Read Ratio')
    plt.ylabel('Bandwidth (MB/s)')
    plt.title('Memory Bandwidth vs Read/Write Ratio for Different Strides')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks([0.0, 0.5, 0.7, 1.0], ['0% (100% Write)', '50%', '70%', '100% (100% Read)'])
    
    plt.tight_layout()
    plt.savefig('results/plots/bandwidth_read_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()

def parse_loaded_latency():
    data = {}
    with open('results/raw_data/loaded_latency.txt') as f:
        for line in f:
            if 'Threads:' in line:
                parts = line.split(',')
                threads = int(parts[0].split(': ')[1].strip())
                stride = int(parts[1].split(': ')[1].replace('B', '').strip())
                throughput_str = parts[2].split(': ')[1].split(' ')[0].strip()
                latency_str = parts[3].split(': ')[1].split(' ')[0].strip()
                
                if stride not in data:
                    data[stride] = {'threads': [], 'throughput': [], 'latency': []}
                
                data[stride]['threads'].append(threads)
                data[stride]['throughput'].append(float(throughput_str))
                data[stride]['latency'].append(float(latency_str))
    return data

def plot_loaded_latency():
    data = parse_loaded_latency()
    
    plt.figure(figsize=(12, 5))
    
    # Throughput vs Threads
    plt.subplot(1, 2, 1)
    for stride, values in data.items():
        plt.plot(values['threads'], values['throughput'], 'o-', label=f'Stride {stride}B')
    plt.xlabel('Number of Threads')
    plt.ylabel('Throughput (MB/s)')
    plt.title('Throughput vs Concurrency')
    plt.legend()
    plt.grid(True)
    
    # Latency vs Threads
    plt.subplot(1, 2, 2)
    for stride, values in data.items():
        plt.plot(values['threads'], values['latency'], 'o-', label=f'Stride {stride}B')
    plt.xlabel('Number of Threads')
    plt.ylabel('Latency (ns/op)')
    plt.title('Latency vs Concurrency')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/plots/loaded_latency.png')
    plt.close()

def parse_kernel_bench():
    cache_data = {'size': [], 'stride': [], 'random': [], 'time': [], 'perf': []}
    tlb_data = {'size': [], 'huge_pages': [], 'time': [], 'perf': []}
    
    with open('results/raw_data/kernel_bench.txt') as f:
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
                
                cache_data['size'].append(size)
                cache_data['stride'].append(stride)
                cache_data['random'].append(random)
                cache_data['time'].append(time)
                cache_data['perf'].append(perf)
            elif line and current_section == 'tlb' and 'Size:' in line:
                parts = line.split(',')
                size = int(parts[0].split(': ')[1].replace('B', '').strip())
                huge_pages = parts[1].split(': ')[1].strip() == 'Yes'
                time = float(parts[2].split(': ')[1].replace('ns', '').strip())
                perf = float(parts[3].split(': ')[1].split(' ')[0].strip())
                
                tlb_data['size'].append(size)
                tlb_data['huge_pages'].append(huge_pages)
                tlb_data['time'].append(time)
                tlb_data['perf'].append(perf)
    
    return cache_data, tlb_data

def plot_kernel_bench():
    cache_data, tlb_data = parse_kernel_bench()
    
    # Plot cache miss impact
    plt.figure(figsize=(12, 6))
    
    # Group by size and access pattern
    unique_sizes = sorted(set(cache_data['size']))
    for random in [False, True]:
        for size in unique_sizes:
            indices = [i for i in range(len(cache_data['size'])) 
                      if cache_data['size'][i] == size and cache_data['random'][i] == random]
            if indices:
                strides = [cache_data['stride'][i] for i in indices]
                perf = [cache_data['perf'][i] for i in indices]
                label = f"{size}B {'Random' if random else 'Sequential'}"
                plt.plot(strides, perf, 'o-', label=label)
    
    plt.xlabel('Stride')
    plt.ylabel('Performance (ops/s)')
    plt.title('Cache Miss Impact on SAXPY Performance')
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('results/plots/cache_miss_impact.png')
    plt.close()
    
    # Plot TLB impact
    plt.figure(figsize=(12, 6))
    
    for huge_pages in [False, True]:
        indices = [i for i in range(len(tlb_data['size'])) 
                  if tlb_data['huge_pages'][i] == huge_pages]
        if indices:
            sizes = [tlb_data['size'][i] for i in indices]
            perf = [tlb_data['perf'][i] for i in indices]
            label = 'Huge Pages' if huge_pages else 'Regular Pages'
            plt.plot(sizes, perf, 'o-', label=label)
    
    plt.xlabel('Working Set Size (bytes)')
    plt.ylabel('Performance (ops/s)')
    plt.title('TLB Impact on SAXPY Performance')
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('results/plots/tlb_impact.png')
    plt.close()

if __name__ == '__main__':
    # Create plots directory if it doesn't exist
    os.makedirs('results/plots', exist_ok=True)
    
    try:
        plot_latency()
        print("Latency plot created successfully")
    except Exception as e:
        print(f"Error creating latency plot: {e}")
    
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
        plot_kernel_bench()
        print("Kernel benchmark plots created successfully")
    except Exception as e:
        print(f"Error creating kernel benchmark plots: {e}")
    
    print("All plots completed!")