import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

def get_project_root():
    """Get the absolute path to the project root directory"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    return project_root

def get_results_dir():
    """Get the absolute path to the results directory"""
    project_root = get_project_root()
    results_dir = project_root / "results"
    return results_dir

def get_plots_dir():
    """Get the absolute path to the plots directory"""
    results_dir = get_results_dir()
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

def load_results_with_metadata(results_dir):
    """Load CSV files that contain metadata comments"""
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No results files found!")
        return None
    
    all_dfs = []
    
    for csv_file in csv_files:
        file_path = os.path.join(results_dir, csv_file)
        print(f"Processing: {csv_file}")
        
        try:
            # Read metadata and find where data starts
            metadata = {}
            data_lines = []
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Parse metadata and find data start
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    # Parse metadata
                    if ':' in line:
                        key, value = line[1:].strip().split(':', 1)
                        metadata[key.strip()] = value.strip()
                else:
                    # Found the start of data
                    data_start = i
                    break
            
            # Extract the data portion
            data_content = ''.join(lines[data_start:])
            
            # Read the data portion as CSV
            from io import StringIO
            df = pd.read_csv(StringIO(data_content))
            
            # Add metadata as columns
            for key, value in metadata.items():
                df[key] = value
            
            # Add filename for tracking
            df['source_file'] = csv_file
            
            all_dfs.append(df)
            print(f"  Successfully loaded {len(df)} rows from {csv_file}")
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {str(e)}")
    
    if not all_dfs:
        return None
    
    return pd.concat(all_dfs, ignore_index=True)

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
    else:
        return 0
    
    return flops / bytes_accessed if bytes_accessed > 0 else 0

def get_peak_performance_i5_12600KF():
    """Return accurate peak performance for i5-12600KF with DDR4-3600"""
    
     # Core configuration
    p_core_count = 6
    e_core_count = 4
    
    # Clock speeds
    p_core_freq = 3.7  # GHz
    e_core_freq = 2.8  # GHz (approximate for E-cores)
    
    # FMA throughput
    p_core_flops_per_cycle = 2 * 8  # 2 FMAs × 8 SP FLOPs (AVX2 256-bit)
    e_core_flops_per_cycle = 1 * 4  # 1 FMA × 4 SP FLOPs (AVX2 128-bit)
    
    # Compute peak FLOPs
    peak_flops_p = p_core_count * p_core_flops_per_cycle * p_core_freq
    peak_flops_e = e_core_count * e_core_flops_per_cycle * e_core_freq
    total_peak_flops = peak_flops_p + peak_flops_e
    
    # Memory bandwidth (DDR4-3200 dual channel)
    theoretical_bandwidth = 2 * 3200 * 8 / 8  # 51.2 GB/s
    realistic_bandwidth = theoretical_bandwidth * 0.80  # Assume 80% efficiency
    
    print(f"i5-12600KF Peak Performance:")
    print(f"  P-core peak: {peak_flops_p:.1f} GFLOP/s")
    print(f"  E-core peak: {peak_flops_e:.1f} GFLOP/s")
    print(f"  Total peak: {total_peak_flops:.1f} GFLOP/s")
    print(f"  Theoretical Bandwidth: {theoretical_bandwidth:.1f} GB/s")
    print(f"  Realistic Bandwidth: {realistic_bandwidth:.1f} GB/s")
    
    return total_peak_flops, realistic_bandwidth

def plot_roofline_model(df, peak_flops, peak_bandwidth, output_dir):
    """Plot roofline model for performance analysis"""
    plt.figure(figsize=(12, 8))
    
    # Filter for relevant data
    filtered_df = df[(df['aligned'] == 1) & (df['stride'] == 1) & 
                    (df['implementation'] == 'vectorized')]
    
    if filtered_df.empty:
        print("Warning: No data available for roofline analysis")
        return
    
    # Calculate arithmetic intensity for each measurement
    ai_values = []
    gflops_values = []
    kernel_types = []
    data_types = []
    array_sizes = []
    
    for _, row in filtered_df.iterrows():
        ai = calculate_arithmetic_intensity(row['kernel'], row['data_type'])
        ai_values.append(ai)
        gflops_values.append(row['gflops'])
        kernel_types.append(row['kernel'])
        data_types.append(row['data_type'])
        array_sizes.append(row['array_size'])
    
    # Create roofline model
    ai_range = np.logspace(-3, 1, 100)  # Reduced range to better show our data
    compute_bound = np.full_like(ai_range, peak_flops)
    memory_bound = peak_bandwidth * ai_range
    
    roofline = np.minimum(compute_bound, memory_bound)
    
    # Plot roofline
    plt.loglog(ai_range, roofline, 'k-', label='Theoretical Roofline', linewidth=2)
    plt.loglog(ai_range, compute_bound, 'r--', label='Compute Bound', alpha=0.7)
    plt.loglog(ai_range, memory_bound, 'b--', label='Memory Bound', alpha=0.7)
    
    # Plot actual measurements with KERNEL-based coloring
    unique_kernels = sorted(list(set(kernel_types)))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_kernels)))
    
    # Create a mapping from kernel to color
    kernel_color_map = {}
    for i, kernel in enumerate(unique_kernels):
        kernel_color_map[kernel] = colors[i]
    
    # Plot each kernel with its assigned color
    for kernel in unique_kernels:
        kernel_ai = [ai for ai, k in zip(ai_values, kernel_types) if k == kernel]
        kernel_gflops = [gf for gf, k in zip(gflops_values, kernel_types) if k == kernel]
        kernel_sizes = [size for size, k in zip(array_sizes, kernel_types) if k == kernel]
        
        if kernel_ai and kernel_gflops:
            plt.scatter(kernel_ai, kernel_gflops, color=kernel_color_map[kernel], 
                       label=kernel, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (GFLOP/s)')
    plt.title('Roofline Model Analysis - i5-12600KF (DDR4-3600)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add compute/memory boundary line
    boundary_ai = peak_flops / peak_bandwidth
    plt.axvline(x=boundary_ai, color='green', linestyle=':', alpha=0.7, linewidth=2)
    plt.text(boundary_ai * 1.1, peak_flops * 0.1, 
             f'Compute/Memory Boundary\n(AI = {boundary_ai:.3f} FLOPs/Byte)',
             rotation=90, verticalalignment='bottom', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'roofline_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed analysis
    print("\nRoofline Analysis:")
    print("==================")
    print(f"Peak Compute: {peak_flops:.1f} GFLOP/s")
    print(f"Peak Bandwidth: {peak_bandwidth:.1f} GB/s")
    print(f"Compute/Memory Boundary: AI = {boundary_ai:.3f} FLOPs/Byte")
    print()
    
    for kernel in unique_kernels:
        kernel_indices = [j for j, k in enumerate(kernel_types) if k == kernel]
        if kernel_indices:
            kernel_ai = [ai_values[j] for j in kernel_indices]
            kernel_gflops = [gflops_values[j] for j in kernel_indices]
            kernel_sizes = [array_sizes[j] for j in kernel_indices]
            
            avg_ai = np.mean(kernel_ai)
            avg_gflops = np.mean(kernel_gflops)
            
            # Determine if compute-bound or memory-bound
            expected_perf = min(peak_flops, peak_bandwidth * avg_ai)
            efficiency = avg_gflops / expected_perf * 100
            
            bound_type = "Compute-bound" if peak_flops < peak_bandwidth * avg_ai else "Memory-bound"
            
            print(f"{kernel}:")
            print(f"  AI={avg_ai:.4f} FLOPs/byte")
            print(f"  Performance={avg_gflops:.2f} GFLOP/s")
            print(f"  Expected={expected_perf:.2f} GFLOP/s")
            print(f"  Efficiency={efficiency:.1f}%")
            print(f"  Characterization: {bound_type}")
            print(f"  Typical sizes: {min(kernel_sizes):,} to {max(kernel_sizes):,} elements")
            print()

def main():
    # Set up directories
    results_dir = get_results_dir()
    plots_dir = get_plots_dir()
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist!")
        sys.exit(1)
    
    # Load results with proper metadata handling
    print("Loading results for roofline analysis...")
    df = load_results_with_metadata(results_dir)
    
    if df is None or df.empty:
        print("No data available for roofline analysis!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows for analysis")
    
    # Use accurate peak performance for i5-12600KF
    peak_flops, peak_bandwidth = get_peak_performance_i5_12600KF()
    
    # Run roofline analysis
    plot_roofline_model(df, peak_flops, peak_bandwidth, plots_dir)
    
    print(f"Roofline analysis completed! Plot saved to {plots_dir}/roofline_analysis.png")

if __name__ == "__main__":
    main()