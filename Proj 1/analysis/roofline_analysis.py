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
    
    if filtered_df.empty:
        print("Warning: No data available for roofline analysis")
        return
    
    # Calculate arithmetic intensity for each measurement
    ai_values = []
    gflops_values = []
    kernel_types = []
    data_types = []
    
    for _, row in filtered_df.iterrows():
        ai = calculate_arithmetic_intensity(row['kernel'], row['data_type'])
        ai_values.append(ai)
        gflops_values.append(row['gflops'])
        kernel_types.append(row['kernel'])
        data_types.append(row['data_type'])
    
    # Create roofline model
    ai_range = np.logspace(-3, 2, 100)
    compute_bound = np.full_like(ai_range, peak_flops)
    memory_bound = peak_bandwidth * ai_range
    
    roofline = np.minimum(compute_bound, memory_bound)
    
    # Plot roofline
    plt.loglog(ai_range, roofline, 'k-', label='Roofline', linewidth=2)
    plt.loglog(ai_range, compute_bound, 'r--', label='Compute Bound')
    plt.loglog(ai_range, memory_bound, 'b--', label='Memory Bound')
    
    # Plot actual measurements
    unique_kernels = list(set(kernel_types))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_kernels)))
    
    for i, kernel in enumerate(unique_kernels):
        kernel_ai = [ai for ai, k in zip(ai_values, kernel_types) if k == kernel]
        kernel_gflops = [gf for gf, k in zip(gflops_values, kernel_types) if k == kernel]
        
        if kernel_ai and kernel_gflops:
            plt.scatter(kernel_ai, kernel_gflops, color=colors[i], label=kernel, 
                       alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (GFLOP/s)')
    plt.title('Roofline Model Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Add annotations for compute-bound and memory-bound regions
    plt.axvline(x=peak_flops/peak_bandwidth, color='green', linestyle=':', alpha=0.7)
    plt.text(peak_flops/peak_bandwidth * 1.1, peak_flops * 0.1, 
             'Compute/Memory Boundary', rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    output_path = output_dir / 'roofline_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print("\nRoofline Analysis:")
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
            
            print(f"{kernel}:")
            print(f"  AI={avg_ai:.4f} FLOPs/byte")
            print(f"  Performance={avg_gflops:.2f} GFLOP/s")
            print(f"  Expected={expected_perf:.2f} GFLOP/s")
            print(f"  Efficiency={efficiency:.1f}%")
            print(f"  Characterization: {bound_type}")
            print()

def estimate_peak_performance(df):
    """Estimate peak performance based on the data"""
    print("Estimating peak performance from data...")
    
    # Get memory bandwidth from memory_bandwidth tests
    bw_tests = df[df['kernel'] == 'memory_bandwidth']
    if not bw_tests.empty:
        peak_bandwidth = bw_tests['bandwidth_gbs'].max() * 1.1  # Add 10% margin
        print(f"Estimated peak bandwidth: {peak_bandwidth:.2f} GB/s (from memory tests)")
    else:
        # Fallback estimation
        peak_bandwidth = 25.0  # GB/s (typical for DDR4)
        print(f"Using default peak bandwidth: {peak_bandwidth:.2f} GB/s")
    
    # Estimate peak FLOPs based on CPU frequency and vector width
    try:
        # Try to get CPU frequency from metadata
        if 'CPU Frequency' in df.columns:
            freq_str = df['CPU Frequency'].iloc[0]
            if 'GHz' in freq_str:
                cpu_freq = float(freq_str.replace('GHz', '').strip())
            else:
                cpu_freq = float(freq_str)
        else:
            cpu_freq = 2.5  # Default fallback
        
        # Conservative estimate: 4 FLOPs/cycle/core for AVX2 + FMA
        peak_flops = cpu_freq * 4 * 1.1  # Add 10% margin
        print(f"Estimated peak FLOPs: {peak_flops:.2f} GFLOP/s ({cpu_freq} GHz × 4 FLOPs/cycle)")
        
    except:
        peak_flops = 10.0  # Default fallback
        print(f"Using default peak FLOPs: {peak_flops:.2f} GFLOP/s")
    
    return peak_flops, peak_bandwidth

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
    
    # Estimate peak performance
    peak_flops, peak_bandwidth = estimate_peak_performance(df)
    
    # Run roofline analysis
    plot_roofline_model(df, peak_flops, peak_bandwidth, plots_dir)
    
    print(f"Roofline analysis completed! Plot saved to {plots_dir}/roofline_analysis.png")

if __name__ == "__main__":
    main()