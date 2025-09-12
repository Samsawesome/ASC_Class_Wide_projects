import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_arithmetic_intensity(kernel, data_type):
    """Calculate arithmetic intensity for different kernels"""
    elem_size = 4 if data_type == 'f32' else 8
    
    if kernel == 'axpy':
        # 2 FLOPs, 3 memory accesses (2 loads, 1 store)
        flops = 2
        bytes_accessed = 3 * elem_size
    elif kernel == 'dot_product':
        # 2 FLOPs, 2 memory accesses (2 loads)
        flops = 2
        bytes_accessed = 2 * elem_size
    elif kernel == 'elementwise_multiply':
        # 1 FLOP, 3 memory accesses (2 loads, 1 store)
        flops = 1
        bytes_accessed = 3 * elem_size
    elif kernel == 'stencil_3point':
        # 5 FLOPs, 3 memory accesses (3 loads)
        flops = 5
        bytes_accessed = 3 * elem_size
    elif kernel == 'memory_bandwidth':
        # 0 FLOPs, 2 memory accesses (1 load, 1 store)
        flops = 0
        bytes_accessed = 2 * elem_size
    else:
        return 0
    
    return flops / bytes_accessed if bytes_accessed > 0 else 0

def plot_roofline_model(df, peak_flops, peak_bandwidth, output_dir):
    """Plot roofline model for performance analysis"""
    plt.figure(figsize=(12, 8))
    
    # Filter for relevant data
    filtered_df = df[(df['aligned'] == 1) & (df['stride'] == 1) & 
                    (df['implementation'] == 'vectorized')]
    
    # Calculate arithmetic intensity for each measurement
    ai_values = []
    gflops_values = []
    kernel_types = []
    
    for _, row in filtered_df.iterrows():
        ai = calculate_arithmetic_intensity(row['kernel'], row['data_type'])
        ai_values.append(ai)
        gflops_values.append(row['gflops'])
        kernel_types.append(row['kernel'])
    
    # Create roofline model
    ai_range = np.logspace(-3, 2, 100)
    compute_bound = np.full_like(ai_range, peak_flops)
    memory_bound = peak_bandwidth * ai_range
    
    roofline = np.minimum(compute_bound, memory_bound)
    
    # Plot roofline
    plt.loglog(ai_range, roofline, 'k-', label='Roofline')
    plt.loglog(ai_range, compute_bound, 'r--', label='Compute Bound')
    plt.loglog(ai_range, memory_bound, 'b--', label='Memory Bound')
    
    # Plot actual measurements
    unique_kernels = list(set(kernel_types))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_kernels)))
    
    for i, kernel in enumerate(unique_kernels):
        kernel_ai = [ai for ai, k in zip(ai_values, kernel_types) if k == kernel]
        kernel_gflops = [gf for gf, k in zip(gflops_values, kernel_types) if k == kernel]
        
        if kernel_ai and kernel_gflops:
            plt.scatter(kernel_ai, kernel_gflops, color=colors[i], label=kernel, alpha=0.7)
    
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (GFLOP/s)')
    plt.title('Roofline Model Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roofline_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print("Roofline Analysis:")
    print("==================")
    for kernel in unique_kernels:
        kernel_ai = [ai for ai, k in zip(ai_values, kernel_types) if k == kernel]
        kernel_gflops = [gf for gf, k in zip(gflops_values, kernel_types) if k == kernel]
        
        if kernel_ai and kernel_gflops:
            avg_ai = np.mean(kernel_ai)
            avg_gflops = np.mean(kernel_gflops)
            
            # Determine if compute-bound or memory-bound
            expected_perf = min(peak_flops, peak_bandwidth * avg_ai)
            efficiency = avg_gflops / expected_perf * 100
            
            bound_type = "Compute-bound" if peak_flops < peak_bandwidth * avg_ai else "Memory-bound"
            
            print(f"{kernel}: AI={avg_ai:.4f}, Performance={avg_gflops:.2f} GFLOP/s, "
                  f"Expected={expected_perf:.2f} GFLOP/s, Efficiency={efficiency:.1f}%, {bound_type}")

def main():
    # Set up directories
    results_dir = "../results"
    plots_dir = os.path.join(results_dir, "plots")
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Load results
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No results files found!")
        return
    
    df = pd.read_csv(os.path.join(results_dir, csv_files[0]))
    
    # Estimate peak performance (adjust based on your CPU)
    # These are example values - you should research your specific CPU's capabilities
    peak_flops = 100.0  # GFLOP/s (example: 4 cores × 2.5 GHz × 8 FLOPs/cycle = 80 GFLOP/s)
    peak_bandwidth = 25.0  # GB/s (example: DDR4-2400 = ~19 GB/s per channel, dual channel = ~38 GB/s)
    
    # Run roofline analysis
    plot_roofline_model(df, peak_flops, peak_bandwidth, plots_dir)

if __name__ == "__main__":
    main()