import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
import os
import sys
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

def get_project_root():
    """Get the absolute path to the project root directory"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # The project root should be the parent of the 'analysis' directory
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
    plots_dir.mkdir(exist_ok=True)  # Create if it doesn't exist
    return plots_dir

def setup_plot_style():
    """Set up consistent plot styling"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14

def load_and_combine_results():
    """Load all CSV results files and combine them into a single DataFrame"""
    results_dir = get_results_dir()
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist!")
        print("Please run the experiments first using run_experiments.bat")
        sys.exit(1)
    
    # Look for CSV files
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in '{results_dir}'!")
        print("Please run the experiments first using run_experiments.bat")
        print("\nTo run experiments:")
        print("1. Open a command prompt")
        print("2. Navigate to the 'src' directory")
        print("3. Run 'build.bat' to compile")
        print("4. Run 'run_experiments.bat' to run experiments")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV file(s) in results directory")
    
    all_data = []
    
    for file in csv_files:
        print(f"Processing: {file.name}")
        
        try:
            # Read metadata from CSV
            metadata = {}
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#'):
                        key_value = line[1:].strip().split(':', 1)
                        if len(key_value) == 2:
                            metadata[key_value[0].strip()] = key_value[1].strip()
                    else:
                        break
            
            # Read actual data
            df = pd.read_csv(file, comment='#')
            
            # Add metadata as columns if they don't exist
            for key, value in metadata.items():
                if key not in df.columns:
                    df[key] = value
            
            # Add filename as a column to track source
            df['source_file'] = file.name
            
            all_data.append(df)
            print(f"  Successfully loaded {len(df)} rows")
            
        except Exception as e:
            print(f"  Error processing {file.name}: {str(e)}")
            print(f"  File path: {file}")
    
    if not all_data:
        print("Error: No data could be loaded from any CSV files!")
        sys.exit(1)
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset contains {len(combined_df)} rows")
    
    return combined_df

def check_and_clean_data(df):
    """Check data quality and clean if necessary"""
    print("\nData Quality Check:")
    print("==================")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # Check for missing values
    missing_values = df_clean.isnull().sum()
    if missing_values.any():
        print("Missing values found:")
        for col, count in missing_values.items():
            if count > 0:
                print(f"  {col}: {count} missing values")
        
        # Remove rows with missing critical values
        critical_cols = ['kernel', 'implementation', 'data_type', 'array_size', 'time_seconds']
        df_clean = df_clean.dropna(subset=critical_cols)
        print(f"Removed {len(df) - len(df_clean)} rows with missing critical values")
    else:
        print("No missing values in critical columns")
    
    # Check for duplicate rows
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows, removing them")
        df_clean = df_clean.drop_duplicates()
    
    # Check data types
    '''print("\nData types:")
    for col in df_clean.columns:
        print(f"  {col}: {df_clean[col].dtype}")'''
    
    # Convert appropriate columns to categorical using .loc to avoid warnings
    categorical_cols = ['kernel', 'implementation', 'data_type', 'aligned', 'stride', 'compiler_flags']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean.loc[:, col] = df_clean[col].astype('category')
    
    return df_clean

def plot_performance_vs_size(df, output_dir):
    """Plot performance metrics vs array size"""
    print("Generating performance vs size plots...")
    
    # Filter for unit stride and aligned data
    filtered_df = df[(df['stride'] == 1) & (df['aligned'] == 1)]
    
    if filtered_df.empty:
        print("Warning: No data available for performance vs size plots")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot 1: GFLOP/s vs Array Size
    ax = axes[0]
    for (kernel, impl, dtype), group in filtered_df.groupby(['kernel', 'implementation', 'data_type']):
        if len(group) > 3:  # Only plot if we have enough data points
            mean_data = group.groupby('array_size')['gflops'].mean()
            std_data = group.groupby('array_size')['gflops'].std()
            
            ax.errorbar(mean_data.index, mean_data, yerr=std_data, 
                        marker='o', linestyle='-', 
                        label=f'{kernel}_{impl}_{dtype}', alpha=0.7)
    
    ax.set_xscale('log')
    ax.set_xlabel('Array Size')
    ax.set_ylabel('GFLOP/s')
    ax.set_title('Performance vs Array Size')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which="both", ls="--")
    
    # Plot 2: Speedup vs Array Size
    ax = axes[1]
    vectorized = filtered_df[filtered_df['implementation'] == 'vectorized']
    if not vectorized.empty and 'speedup' in vectorized.columns:
        for (kernel, dtype), group in vectorized.groupby(['kernel', 'data_type']):
            if len(group) > 3:
                mean_data = group.groupby('array_size')['speedup'].mean()
                std_data = group.groupby('array_size')['speedup'].std()
                
                ax.errorbar(mean_data.index, mean_data, yerr=std_data, 
                            marker='s', linestyle='-', 
                            label=f'{kernel}_{dtype}', alpha=0.7)
        
        ax.set_xscale('log')
        ax.set_xlabel('Array Size')
        ax.set_ylabel('Speedup (Scalar/Vectorized)')
        ax.set_title('SIMD Speedup vs Array Size')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, which="both", ls="--")
    else:
        ax.text(0.5, 0.5, 'No speedup data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('SIMD Speedup vs Array Size (No Data)')
    
    '''# Plot 3: Bandwidth vs Array Size
    ax = axes[2]
    for (kernel, impl, dtype), group in filtered_df.groupby(['kernel', 'implementation', 'data_type']):
        if len(group) > 3 and kernel == 'memory_bandwidth':
            mean_data = group.groupby('array_size')['bandwidth_gbs'].mean()
            std_data = group.groupby('array_size')['bandwidth_gbs'].std()
            
            ax.errorbar(mean_data.index, mean_data, yerr=std_data, 
                        marker='o', linestyle='-', 
                        label=f'{kernel}_{impl}_{dtype}', alpha=0.7)
    
    ax.set_xscale('log')
    ax.set_xlabel('Array Size')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('Memory Bandwidth vs Array Size')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', handles=[])
    ax.grid(True, which="both", ls="--")'''
    
    # Plot 4: CPE vs Array Size
    ax = axes[2]
    for (kernel, impl, dtype), group in filtered_df.groupby(['kernel', 'implementation', 'data_type']):
        if len(group) > 3:
            mean_data = group.groupby('array_size')['cpe'].mean()
            std_data = group.groupby('array_size')['cpe'].std()
            
            ax.errorbar(mean_data.index, mean_data, yerr=std_data, 
                        marker='o', linestyle='-', 
                        label=f'{kernel}_{impl}_{dtype}', alpha=0.7)
    
    ax.set_xscale('log')
    ax.set_xlabel('Array Size')
    ax.set_ylabel('Cycles Per Element (CPE)')
    ax.set_title('CPE vs Array Size')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    output_path = output_dir / 'performance_vs_size.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance vs size plot to {output_path}")

def plot_alignment_impact(df, output_dir):
    """Plot the impact of alignment on performance"""
    print("Generating alignment impact plots...")
    
    # Filter for unit stride and medium-sized arrays
    filtered_df = df[(df['stride'] == 1) & (df['array_size'] == 65536)]
    
    if filtered_df.empty:
        print("Warning: No data available for alignment impact plots")
        return
    
    # Group by kernel, implementation, data type, and alignment
    try:
        grouped = filtered_df.groupby(['kernel', 'implementation', 'data_type', 'aligned'])
        
        # Calculate mean and std for each group
        means = grouped['gflops'].mean().unstack()
        stds = grouped['gflops'].std().unstack()
        
        # Create bar plot
        x = np.arange(len(means))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, (aligned, values) in enumerate(means.items()):
            offset = width * i
            rects = ax.bar(x + offset, values, width, label=f'Aligned={aligned}',
                          yerr=stds[aligned] if aligned in stds else None)
        
        ax.set_xlabel('Configuration (Kernel_Implementation_DataType)')
        ax.set_ylabel('GFLOP/s')
        ax.set_title('Impact of Alignment on Performance (Array Size=65536)')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f'{k}_{i}_{d}' for k, i, d in means.index], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        output_path = output_dir / 'alignment_impact.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved alignment impact plot to {output_path}")
        
    except Exception as e:
        print(f"Error creating alignment plot: {str(e)}")

def plot_stride_impact(df, output_dir):
    """Plot the impact of stride on performance"""
    print("Generating stride impact plots...")
    
    # Filter for aligned data and fixed size
    filtered_df = df[(df['aligned'] == 1) & (df['array_size'] == 1048576)]
    
    if filtered_df.empty:
        print("Warning: No data available for stride impact plots")
        return
    
    # Group by kernel, implementation, data type, and stride
    try:
        grouped = filtered_df.groupby(['kernel', 'implementation', 'data_type', 'stride'])
        
        # Calculate mean and std for each group
        means = grouped['gflops'].mean().unstack()
        
        # Create line plot - FIXED: Changed subforms to subplots
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for (kernel, impl, dtype), group_data in means.iterrows():
            label = f'{kernel}_{impl}_{dtype}'
            ax.plot(group_data.index, group_data.values, marker='o', label=label)
        
        ax.set_xlabel('Stride')
        ax.set_ylabel('GFLOP/s')
        ax.set_title('Impact of Stride on Performance (Array Size=1M)')
        ax.set_xscale('log', base=2)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, which="both", ls="--")
        
        plt.tight_layout()
        output_path = output_dir / 'stride_impact.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved stride impact plot to {output_path}")
        
    except Exception as e:
        print(f"Error creating stride plot: {str(e)}")

def plot_data_type_comparison(df, output_dir):
    """Compare performance across different data types"""
    print("Generating data type comparison plots...")
    
    # Filter for aligned data, unit stride, and vectorized implementation
    filtered_df = df[(df['aligned'] == 1) & (df['stride'] == 1) & 
                    (df['implementation'] == 'vectorized')]
    
    if filtered_df.empty:
        print("Warning: No data available for data type comparison plots")
        return
    
    # Group by kernel, data type, and array size
    try:
        grouped = filtered_df.groupby(['kernel', 'data_type', 'array_size'])
        
        # Calculate mean GFLOP/s for each group
        means = grouped['gflops'].mean().unstack(level=1)
        
        # Create subplots for each kernel
        kernels = means.index.get_level_values(0).unique()
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, kernel in enumerate(kernels):
            if i >= len(axes):
                break
                
            kernel_data = means.loc[kernel]
            ax = axes[i]
            
            for dtype in kernel_data.columns:
                ax.loglog(kernel_data.index, kernel_data[dtype], marker='o', label=dtype)
            
            ax.set_xlabel('Array Size')
            ax.set_ylabel('GFLOP/s')
            ax.set_title(f'{kernel} - Data Type Comparison')
            ax.legend()
            ax.grid(True, which="both", ls="--")
        
        plt.tight_layout()
        output_path = output_dir / 'data_type_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved data type comparison plot to {output_path}")
        
    except Exception as e:
        print(f"Error creating data type comparison plot: {str(e)}")

def plot_compiler_comparison(df, output_dir):
    """Compare performance across different compiler flags"""
    print("Generating compiler comparison plots...")
    
    # Filter for aligned data, unit stride, and medium-sized arrays
    filtered_df = df[(df['aligned'] == 1) & (df['stride'] == 1) & 
                    (df['array_size'] == 65536)]
    
    if filtered_df.empty:
        print("Warning: No data available for compiler comparison plots")
        return
    
    # Group by kernel, implementation, data type, and compiler flags
    try:
        grouped = filtered_df.groupby(['kernel', 'implementation', 'data_type', 'compiler_flags'])
        
        # Calculate mean GFLOP/s for each group
        means = grouped['gflops'].mean().unstack()
        
        # Create bar plot
        x = np.arange(len(means))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        compilers = means.columns
        for i, compiler in enumerate(compilers):
            offset = width * i
            rects = ax.bar(x + offset, means[compiler], width, label=compiler)
        
        ax.set_xlabel('Configuration (Kernel_Implementation_DataType)')
        ax.set_ylabel('GFLOP/s')
        ax.set_title('Compiler Optimization Comparison (Array Size=65536)')
        ax.set_xticks(x + width * (len(compilers) - 1) / 2)
        ax.set_xticklabels([f'{k}_{i}_{d}' for k, i, d in means.index], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        output_path = output_dir / 'compiler_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved compiler comparison plot to {output_path}")
        
    except Exception as e:
        print(f"Error creating compiler comparison plot: {str(e)}")

def generate_summary_report(df, output_dir):
    """Generate a comprehensive summary report"""
    
    report_path = output_dir / 'summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("SIMD Performance Analysis Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write("Basic Statistics:\n")
        f.write(f"Total measurements: {len(df)}\n")
        f.write(f"Number of kernels: {df['kernel'].nunique()}\n")
        if 'compiler_flags' in df.columns:
            f.write(f"Number of compiler configurations: {df['compiler_flags'].nunique()}\n")
        f.write(f"Data types: {', '.join(map(str, df['data_type'].unique()))}\n\n")
        
        # Performance summary by kernel
        f.write("Performance Summary by Kernel:\n")
        f.write("-" * 40 + "\n")
        
        for kernel in df['kernel'].unique():
            kernel_data = df[df['kernel'] == kernel]
            f.write(f"\n{kernel}:\n")
            
            for impl in kernel_data['implementation'].unique():
                impl_data = kernel_data[kernel_data['implementation'] == impl]
                avg_gflops = impl_data['gflops'].mean()
                
                f.write(f"  {impl}: {avg_gflops:.2f} GFLOP/s")
                
                if 'speedup' in impl_data.columns and impl == 'vectorized':
                    avg_speedup = impl_data['speedup'].mean()
                    f.write(f", Speedup: {avg_speedup:.2f}x")
                f.write("\n")
        
        # Best and worst cases
        f.write("\nBest and Worst Cases:\n")
        f.write("-" * 40 + "\n")
        
        if 'gflops' in df.columns:
            best_case_idx = df['gflops'].idxmax()
            worst_case_idx = df['gflops'].idxmin()
            
            best_case = df.loc[best_case_idx]
            worst_case = df.loc[worst_case_idx]
            
            f.write(f"Best performance: {best_case['gflops']:.2f} GFLOP/s "
                   f"({best_case['kernel']}, {best_case['implementation']}, "
                   f"{best_case['data_type']}, size={best_case['array_size']})\n")
            
            f.write(f"Worst performance: {worst_case['gflops']:.2f} GFLOP/s "
                   f"({worst_case['kernel']}, {worst_case['implementation']}, "
                   f"{worst_case['data_type']}, size={worst_case['array_size']})\n")
        
        # Speedup analysis
        if 'speedup' in df.columns:
            f.write("\nSpeedup Analysis (Vectorized vs Scalar):\n")
            f.write("-" * 40 + "\n")
            
            vectorized_data = df[df['implementation'] == 'vectorized']
            avg_speedup = vectorized_data['speedup'].mean()
            max_speedup = vectorized_data['speedup'].max()
            min_speedup = vectorized_data['speedup'].min()
            
            f.write(f"Average speedup: {avg_speedup:.2f}x\n")
            f.write(f"Maximum speedup: {max_speedup:.2f}x\n")
            f.write(f"Minimum speedup: {min_speedup:.2f}x\n")
            
            # Speedup by kernel
            f.write("\nSpeedup by kernel:\n")
            for kernel in vectorized_data['kernel'].unique():
                kernel_speedup = vectorized_data[vectorized_data['kernel'] == kernel]['speedup'].mean()
                f.write(f"  {kernel}: {kernel_speedup:.2f}x\n")
        
        f.write("\nReport generated successfully!\n")
    
    print(f"Summary report saved to {report_path}")

def plot_baseline_vs_vectorized(df, output_dir):
    """Plot baseline (scalar) vs auto-vectorized comparison"""
    print("Generating baseline vs vectorized comparison plots...")
    
    # Filter for relevant data: unit stride, aligned, and both implementations
    filtered_df = df[(df['stride'] == 1) & (df['aligned'] == 1)]
    
    if filtered_df.empty:
        print("Warning: No data available for baseline vs vectorized comparison")
        return
    
    # Create subplots for each kernel
    kernels = filtered_df['kernel'].unique()
    n_kernels = len(kernels)
    
    fig, axes = plt.subplots(2, n_kernels, figsize=(6 * n_kernels, 10))
    if n_kernels == 1:
        axes = axes.reshape(2, 1)
    
    # Define cache boundaries for annotation (adjust based on your CPU)
    cache_boundaries = {
        'L1': 32 * 1024,      # 32 KB
        'L2': 256 * 1024,     # 256 KB  
        'L3': 8 * 1024 * 1024, # 8 MB
        'DRAM': 16 * 1024 * 1024 # 16 MB
    }
    
    for i, kernel in enumerate(kernels):
        kernel_data = filtered_df[filtered_df['kernel'] == kernel]
        
        # Plot 1: GFLOP/s comparison
        ax1 = axes[0, i]
        for impl in ['scalar', 'vectorized']:
            impl_data = kernel_data[kernel_data['implementation'] == impl]
            if not impl_data.empty:
                # Group by array size and calculate mean and std
                grouped = impl_data.groupby('array_size')
                mean_gflops = grouped['gflops'].mean()
                std_gflops = grouped['gflops'].std()
                
                ax1.errorbar(mean_gflops.index, mean_gflops, yerr=std_gflops,
                            marker='o', linestyle='-', label=impl.capitalize(),
                            alpha=0.8, capsize=3)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Array Size')
        ax1.set_ylabel('GFLOP/s')
        ax1.set_title(f'{kernel.upper()} - Performance')
        ax1.legend()
        ax1.grid(True, which="both", ls="--", alpha=0.3)
        
        # Add cache boundary annotations
        for cache_name, cache_size in cache_boundaries.items():
            ax1.axvline(x=cache_size, color='gray', linestyle=':', alpha=0.7)
            ax1.text(cache_size * 1.1, ax1.get_ylim()[0] * 10, cache_name, 
                    rotation=90, verticalalignment='bottom', fontsize=10)
        
        # Plot 2: Speedup
        ax2 = axes[1, i]
        # Calculate speedup for each configuration
        speedup_data = []
        for config in kernel_data.groupby(['array_size', 'data_type']):
            size, dtype = config[0]
            config_data = config[1]
            
            if len(config_data) >= 2:  # Need both scalar and vectorized
                scalar_perf = config_data[config_data['implementation'] == 'scalar']['gflops'].mean()
                vectorized_perf = config_data[config_data['implementation'] == 'vectorized']['gflops'].mean()
                
                if scalar_perf > 0 and vectorized_perf > 0:
                    speedup = vectorized_perf / scalar_perf
                    speedup_data.append((size, speedup, dtype))
        
        if speedup_data:
            # Group by array size
            sizes, speedups, dtypes = zip(*speedup_data)
            unique_sizes = sorted(set(sizes))
            
            # Calculate mean speedup for each size
            mean_speedups = []
            std_speedups = []
            for size in unique_sizes:
                size_speedups = [s for s, sz, dt in zip(speedups, sizes, dtypes) if sz == size]
                mean_speedups.append(np.mean(size_speedups))
                std_speedups.append(np.std(size_speedups))
            
            ax2.errorbar(unique_sizes, mean_speedups, yerr=std_speedups,
                        marker='s', linestyle='-', color='green',
                        alpha=0.8, capsize=3, label='Speedup')
            
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No speedup')
            
            # Add ideal speedup lines based on vector width
            vector_widths = {'f32': 8, 'f64': 4}  # AVX2: 8 floats, 4 doubles
            for dtype, width in vector_widths.items():
                ax2.axhline(y=width, color='blue', linestyle=':', alpha=0.7, 
                           label=f'Ideal {dtype} ({width}x)')
        
        ax2.set_xscale('log')
        ax2.set_xlabel('Array Size')
        ax2.set_ylabel('Speedup (Vectorized/Scalar)')
        ax2.set_title(f'{kernel.upper()} - Speedup')
        ax2.legend()
        ax2.grid(True, which="both", ls="--", alpha=0.3)
        
        # Add cache boundary annotations
        for cache_name, cache_size in cache_boundaries.items():
            ax2.axvline(x=cache_size, color='gray', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    output_path = output_dir / 'baseline_vs_vectorized.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved baseline vs vectorized comparison to {output_path}")

def plot_detailed_speedup_analysis(df, output_dir):
    """Detailed speedup analysis with data type breakdown"""
    print("Generating detailed speedup analysis...")
    
    # Filter for relevant data
    filtered_df = df[(df['stride'] == 1) & (df['aligned'] == 1)]
    
    if filtered_df.empty:
        print("Warning: No data available for detailed speedup analysis")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot 1: Speedup by kernel and data type
    ax1 = axes[0]
    speedup_data = []
    
    for (kernel, dtype), group in filtered_df.groupby(['kernel', 'data_type']):
        # Calculate speedup for each array size
        for size, size_group in group.groupby('array_size'):
            scalar_data = size_group[size_group['implementation'] == 'scalar']
            vectorized_data = size_group[size_group['implementation'] == 'vectorized']
            
            if len(scalar_data) > 0 and len(vectorized_data) > 0:
                scalar_perf = scalar_data['gflops'].mean()
                vectorized_perf = vectorized_data['gflops'].mean()
                
                if scalar_perf > 0:
                    speedup = vectorized_perf / scalar_perf
                    speedup_data.append({
                        'kernel': kernel,
                        'dtype': dtype,
                        'size': size,
                        'speedup': speedup
                    })
    
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        
        # Plot speedup by kernel and data type
        for (kernel, dtype), group in speedup_df.groupby(['kernel', 'dtype']):
            mean_speedup = group.groupby('size')['speedup'].mean()
            ax1.loglog(mean_speedup.index, mean_speedup, marker='o', 
                      label=f'{kernel}_{dtype}', alpha=0.8)
        
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax1.set_xlabel('Array Size')
        ax1.set_ylabel('Speedup (Vectorized/Scalar)')
        ax1.set_title('Speedup by Kernel and Data Type')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 2: Average speedup by kernel
    ax2 = axes[1]
    if speedup_data:
        avg_speedup = speedup_df.groupby('kernel')['speedup'].mean()
        colors = plt.cm.Set3(np.arange(len(avg_speedup)))
        
        bars = ax2.bar(range(len(avg_speedup)), avg_speedup.values, color=colors)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Kernel')
        ax2.set_ylabel('Average Speedup')
        ax2.set_title('Average Speedup by Kernel')
        ax2.set_xticks(range(len(avg_speedup)))
        ax2.set_xticklabels(avg_speedup.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_speedup.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}x', ha='center', va='bottom')
    
    # Plot 3: Speedup vs Array Size (all kernels)
    ax3 = axes[2]
    if speedup_data:
        for kernel in speedup_df['kernel'].unique():
            kernel_data = speedup_df[speedup_df['kernel'] == kernel]
            mean_speedup = kernel_data.groupby('size')['speedup'].mean()
            ax3.loglog(mean_speedup.index, mean_speedup, marker='s', 
                      label=kernel, alpha=0.8)
        
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax3.set_xlabel('Array Size')
        ax3.set_ylabel('Speedup')
        ax3.set_title('Speedup vs Array Size (All Kernels)')
        ax3.legend()
        ax3.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 4: Performance comparison for largest array size
    ax4 = axes[3]
    largest_size = filtered_df['array_size'].max()
    large_data = filtered_df[filtered_df['array_size'] == largest_size]
    
    if not large_data.empty:
        # Group by kernel and implementation
        performance_data = []
        for (kernel, impl), group in large_data.groupby(['kernel', 'implementation']):
            performance_data.append({
                'kernel': kernel,
                'implementation': impl,
                'gflops': group['gflops'].mean()
            })
        
        perf_df = pd.DataFrame(performance_data)
        
        # Create grouped bar plot
        x = np.arange(len(perf_df['kernel'].unique()))
        width = 0.35
        
        for i, impl in enumerate(['scalar', 'vectorized']):
            impl_data = perf_df[perf_df['implementation'] == impl]
            values = impl_data['gflops'].values
            ax4.bar(x + i * width, values, width, label=impl.capitalize())
        
        ax4.set_xlabel('Kernel')
        ax4.set_ylabel('GFLOP/s')
        ax4.set_title(f'Performance at Largest Size ({largest_size} elements)')
        ax4.set_xticks(x + width / 2)
        ax4.set_xticklabels(perf_df['kernel'].unique(), rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'detailed_speedup_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed speedup analysis to {output_path}")

def generate_speedup_report(df, output_dir):
    """Generate a detailed speedup analysis report"""
    print("Generating speedup analysis report...")
    
    report_path = output_dir / 'speedup_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("SIMD Speedup Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        
        # Filter for relevant data
        filtered_df = df[(df['stride'] == 1) & (df['aligned'] == 1)]
        
        if filtered_df.empty:
            f.write("No data available for speedup analysis.\n")
            return
        
        # Calculate speedup for each configuration
        speedup_results = []
        
        for (kernel, dtype, size), group in filtered_df.groupby(['kernel', 'data_type', 'array_size']):
            scalar_data = group[group['implementation'] == 'scalar']
            vectorized_data = group[group['implementation'] == 'vectorized']
            
            if len(scalar_data) > 0 and len(vectorized_data) > 0:
                scalar_perf = scalar_data['gflops'].mean()
                vectorized_perf = vectorized_data['gflops'].mean()
                
                if scalar_perf > 0:
                    speedup = vectorized_perf / scalar_perf
                    speedup_results.append({
                        'kernel': kernel,
                        'dtype': dtype,
                        'size': size,
                        'speedup': speedup,
                        'scalar_gflops': scalar_perf,
                        'vectorized_gflops': vectorized_perf
                    })
        
        if not speedup_results:
            f.write("No speedup data available.\n")
            return
        
        speedup_df = pd.DataFrame(speedup_results)
        
        # Overall statistics
        f.write("Overall Speedup Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average speedup: {speedup_df['speedup'].mean():.3f}x\n")
        f.write(f"Maximum speedup: {speedup_df['speedup'].max():.3f}x\n")
        f.write(f"Minimum speedup: {speedup_df['speedup'].min():.3f}x\n")
        f.write(f"Number of measurements: {len(speedup_df)}\n\n")
        
        # By kernel
        f.write("Speedup by Kernel:\n")
        f.write("-" * 30 + "\n")
        for kernel in speedup_df['kernel'].unique():
            kernel_data = speedup_df[speedup_df['kernel'] == kernel]
            f.write(f"{kernel}: {kernel_data['speedup'].mean():.3f}x "
                   f"(min: {kernel_data['speedup'].min():.3f}x, "
                   f"max: {kernel_data['speedup'].max():.3f}x)\n")
        f.write("\n")
        
        # By data type
        f.write("Speedup by Data Type:\n")
        f.write("-" * 30 + "\n")
        for dtype in speedup_df['dtype'].unique():
            dtype_data = speedup_df[speedup_df['dtype'] == dtype]
            f.write(f"{dtype}: {dtype_data['speedup'].mean():.3f}x\n")
        f.write("\n")
        
        # By array size (cache regions)
        f.write("Speedup by Cache Region:\n")
        f.write("-" * 30 + "\n")
        
        # Define cache regions
        cache_regions = {
            'L1': (0, 32 * 1024),
            'L2': (32 * 1024, 256 * 1024),
            'L3': (256 * 1024, 8 * 1024 * 1024),
            'DRAM': (8 * 1024 * 1024, float('inf'))
        }
        
        for region, (min_size, max_size) in cache_regions.items():
            region_data = speedup_df[(speedup_df['size'] >= min_size) & 
                                   (speedup_df['size'] < max_size)]
            if not region_data.empty:
                f.write(f"{region}: {region_data['speedup'].mean():.3f}x "
                       f"(n={len(region_data)} measurements)\n")
        f.write("\n")
        
        # Top 5 best and worst speedups
        f.write("Top 5 Best Speedups:\n")
        f.write("-" * 30 + "\n")
        best_speedups = speedup_df.nlargest(5, 'speedup')
        for _, row in best_speedups.iterrows():
            f.write(f"{row['kernel']}_{row['dtype']} (size={row['size']}): "
                   f"{row['speedup']:.3f}x\n")
        f.write("\n")
        
        f.write("Top 5 Worst Speedups:\n")
        f.write("-" * 30 + "\n")
        worst_speedups = speedup_df.nsmallest(5, 'speedup')
        for _, row in worst_speedups.iterrows():
            f.write(f"{row['kernel']}_{row['dtype']} (size={row['size']}): "
                   f"{row['speedup']:.3f}x\n")
    
    print(f"Speedup analysis report saved to {report_path}")

def plot_locality_sweep(df, output_dir):
    """Plot locality sweep showing cache transitions and SIMD gains compression"""
    print("Generating locality sweep analysis...")
    
    # Filter for relevant data: unit stride, aligned, and focus on AXPY kernel
    filtered_df = df[(df['stride'] == 1) & (df['aligned'] == 1) & 
                    (df['kernel'] == 'axpy')]  # Focus on AXPY as requested
    
    if filtered_df.empty:
        # Fallback to any available kernel
        filtered_df = df[(df['stride'] == 1) & (df['aligned'] == 1)]
        if filtered_df.empty:
            print("Warning: No data available for locality sweep analysis")
            return
    
    # Get the first available kernel if AXPY isn't available
    kernel = filtered_df['kernel'].iloc[0] if filtered_df['kernel'].iloc[0] else filtered_df['kernel'].unique()[0]
    filtered_df = filtered_df[filtered_df['kernel'] == kernel]
    
    # Define cache boundaries (adjust based on your CPU architecture)
    # Typical cache sizes for modern CPUs:
    cache_boundaries = {
        'L1': 32 * 1024,        # 32 KB
        'L2': 256 * 1024,       # 256 KB
        'L3': 8 * 1024 * 1024,  # 8 MB
        'DRAM': 16 * 1024 * 1024 # 16 MB (transition to full DRAM)
    }
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for different implementations
    colors = {'scalar': 'blue', 'vectorized': 'red'}
    
    # Plot 1: GFLOP/s vs Array Size
    for implementation in ['scalar', 'vectorized']:
        impl_data = filtered_df[filtered_df['implementation'] == implementation]
        if not impl_data.empty:
            # Group by array size and calculate statistics
            grouped = impl_data.groupby('array_size')
            mean_gflops = grouped['gflops'].mean()
            std_gflops = grouped['gflops'].std()
            
            ax1.errorbar(mean_gflops.index, mean_gflops, yerr=std_gflops,
                        marker='o', linestyle='-', color=colors[implementation],
                        label=implementation.capitalize(), alpha=0.8, capsize=3)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Array Size (elements)')
    ax1.set_ylabel('GFLOP/s')
    ax1.set_title(f'{kernel.upper()} - Performance vs Working Set Size')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 2: Bandwidth vs Array Size
    for implementation in ['scalar', 'vectorized']:
        impl_data = filtered_df[filtered_df['implementation'] == implementation]
        if not impl_data.empty:
            grouped = impl_data.groupby('array_size')
            mean_bandwidth = grouped['bandwidth_gbs'].mean()
            std_bandwidth = grouped['bandwidth_gbs'].std()
            
            ax2.errorbar(mean_bandwidth.index, mean_bandwidth, yerr=std_bandwidth,
                        marker='s', linestyle='-', color=colors[implementation],
                        label=implementation.capitalize(), alpha=0.8, capsize=3)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Array Size (elements)')
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.set_title(f'{kernel.upper()} - Memory Bandwidth vs Working Set Size')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 3: CPE (Cycles Per Element) vs Array Size
    for implementation in ['scalar', 'vectorized']:
        impl_data = filtered_df[filtered_df['implementation'] == implementation]
        if not impl_data.empty:
            grouped = impl_data.groupby('array_size')
            mean_cpe = grouped['cpe'].mean()
            std_cpe = grouped['cpe'].std()
            
            ax3.errorbar(mean_cpe.index, mean_cpe, yerr=std_cpe,
                        marker='^', linestyle='-', color=colors[implementation],
                        label=implementation.capitalize(), alpha=0.8, capsize=3)
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Array Size (elements)')
    ax3.set_ylabel('Cycles Per Element (CPE)')
    ax3.set_title(f'{kernel.upper()} - CPE vs Working Set Size')
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 4: Speedup vs Array Size (showing where SIMD gains compress)
    speedup_data = []
    for array_size in filtered_df['array_size'].unique():
        size_data = filtered_df[filtered_df['array_size'] == array_size]
        scalar_data = size_data[size_data['implementation'] == 'scalar']
        vectorized_data = size_data[size_data['implementation'] == 'vectorized']
        
        if len(scalar_data) > 0 and len(vectorized_data) > 0:
            scalar_perf = scalar_data['gflops'].mean()
            vectorized_perf = vectorized_data['gflops'].mean()
            
            if scalar_perf > 0:
                speedup = vectorized_perf / scalar_perf
                speedup_data.append((array_size, speedup))
    
    if speedup_data:
        sizes, speedups = zip(*speedup_data)
        ax4.loglog(sizes, speedups, marker='D', linestyle='-', color='green',
                  label='Speedup (Vectorized/Scalar)', alpha=0.8)
        
        # Add ideal speedup lines
        ideal_speedups = {'f32': 8, 'f64': 4}  # AVX2 ideal speedups
        for dtype, ideal in ideal_speedups.items():
            ax4.axhline(y=ideal, color='gray', linestyle=':', 
                       label=f'Ideal {dtype} ({ideal}x)', alpha=0.7)
        
        ax4.axhline(y=1.0, color='red', linestyle='--', label='No speedup', alpha=0.7)
    
    ax4.set_xscale('log')
    ax4.set_xlabel('Array Size (elements)')
    ax4.set_ylabel('Speedup (Vectorized/Scalar)')
    ax4.set_title(f'{kernel.upper()} - SIMD Speedup vs Working Set Size')
    ax4.legend()
    ax4.grid(True, which="both", ls="--", alpha=0.3)
    
    # Add cache boundary annotations to all plots
    for ax in [ax1, ax2, ax3, ax4]:
        for cache_name, cache_size in cache_boundaries.items():
            ax.axvline(x=cache_size, color='orange', linestyle='--', alpha=0.7)
            # Add text annotation only to the first plot to avoid clutter
            if ax == ax1:
                ax.text(cache_size * 1.1, ax.get_ylim()[0] * 10, cache_name, 
                       rotation=90, verticalalignment='bottom', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / 'locality_sweep_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved locality sweep analysis to {output_path}")
    
    # Generate detailed analysis report
    generate_locality_report(filtered_df, cache_boundaries, output_dir)

def generate_locality_report(df, cache_boundaries, output_dir):
    """Generate detailed locality analysis report"""
    print("Generating locality analysis report...")
    
    report_path = output_dir / 'locality_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("Locality Sweep Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        
        kernel = df['kernel'].iloc[0]
        f.write(f"Kernel: {kernel}\n")
        f.write(f"Data points: {len(df)}\n\n")
        
        # Analyze performance in different cache regions
        f.write("Performance Analysis by Cache Region:\n")
        f.write("-" * 40 + "\n")
        
        cache_regions = {
            'L1': (0, cache_boundaries['L1']),
            'L2': (cache_boundaries['L1'], cache_boundaries['L2']),
            'L3': (cache_boundaries['L2'], cache_boundaries['L3']),
            'DRAM': (cache_boundaries['L3'], float('inf'))
        }
        
        for region, (min_size, max_size) in cache_regions.items():
            f.write(f"\n{region} Cache Region:\n")
            region_data = df[(df['array_size'] >= min_size) & 
                           (df['array_size'] < max_size)]
            
            if not region_data.empty:
                for implementation in ['scalar', 'vectorized']:
                    impl_data = region_data[region_data['implementation'] == implementation]
                    if not impl_data.empty:
                        avg_gflops = impl_data['gflops'].mean()
                        avg_bandwidth = impl_data['bandwidth_gbs'].mean()
                        avg_cpe = impl_data['cpe'].mean()
                        
                        f.write(f"  {implementation.capitalize()}:\n")
                        f.write(f"    Avg GFLOP/s: {avg_gflops:.2f}\n")
                        f.write(f"    Avg Bandwidth: {avg_bandwidth:.2f} GB/s\n")
                        f.write(f"    Avg CPE: {avg_cpe:.2f}\n")
            else:
                f.write(f"  No data in this region\n")
        
        # Analyze SIMD gains compression
        f.write("\nSIMD Gains Compression Analysis:\n")
        f.write("-" * 40 + "\n")
        
        speedup_data = []
        for array_size in df['array_size'].unique():
            size_data = df[df['array_size'] == array_size]
            scalar_data = size_data[size_data['implementation'] == 'scalar']
            vectorized_data = size_data[size_data['implementation'] == 'vectorized']
            
            if len(scalar_data) > 0 and len(vectorized_data) > 0:
                scalar_gflops = scalar_data['gflops'].mean()
                vectorized_gflops = vectorized_data['gflops'].mean()
                
                if scalar_gflops > 0:
                    speedup = vectorized_gflops / scalar_gflops
                    speedup_data.append((array_size, speedup, scalar_gflops, vectorized_gflops))
        
        if speedup_data:
            # Find where SIMD gains start to compress (speedup decreases significantly)
            sizes, speedups, scalar_gflops, vectorized_gflops = zip(*speedup_data)
            
            # Calculate derivative of speedup to find compression points
            speedup_changes = []
            for i in range(1, len(speedups)):
                if sizes[i] > sizes[i-1]:  # Ensure increasing sizes
                    change = (speedups[i] - speedups[i-1]) / (sizes[i] - sizes[i-1])
                    speedup_changes.append((sizes[i], change))
            
            # Find significant compression points
            compression_points = []
            for size, change in speedup_changes:
                if change < -0.000001:  # Significant negative change
                    # Find which cache boundary this is near
                    for cache_name, cache_size in cache_boundaries.items():
                        if abs(size - cache_size) < cache_size * 0.5:  # Within 50% of cache size
                            compression_points.append((size, cache_name))
                            break
            
            f.write("Significant SIMD gains compression detected at:\n")
            for size, cache_name in compression_points:
                f.write(f"  {cache_name} boundary (~{size} elements)\n")
            
            if not compression_points:
                f.write("  No significant compression detected in measured range\n")
            
            # Analyze memory-bound vs compute-bound behavior
            f.write("\nMemory-bound vs Compute-bound Analysis:\n")
            f.write("(Based on bandwidth utilization and speedup patterns)\n")
            
            for region, (min_size, max_size) in cache_regions.items():
                region_speedups = [s for s, sz, sg, vg in speedup_data 
                                 if min_size <= sz < max_size]
                if region_speedups:
                    avg_speedup = np.mean(region_speedups)
                    f.write(f"  {region}: Average speedup = {avg_speedup:.2f}x\n")
                    
                    if avg_speedup < 2.0:
                        f.write(f"    --> Likely memory-bound (limited SIMD gains)\n")
                    else:
                        f.write(f"    --> Likely compute-bound (good SIMD gains)\n")
        
        f.write("\nKey Observations:\n")
        f.write("-" * 40 + "\n")
        f.write("1. L1 Cache: Typically shows best SIMD speedup (compute-bound)\n")
        f.write("2. L2 Cache: Speedup may start to decrease as memory bandwidth becomes limiting\n")
        f.write("3. L3 Cache: Often shows significant compression of SIMD gains\n")
        f.write("4. DRAM: Usually memory-bound with minimal SIMD benefits\n")
        f.write("5. The transition points indicate where memory bandwidth becomes the bottleneck\n")
    
    print(f"Locality analysis report saved to {report_path}")

def plot_memory_bound_analysis(df, output_dir):
    """Additional analysis showing memory-bound behavior"""
    print("Generating memory-bound behavior analysis...")
    
    # Filter for relevant data
    filtered_df = df[(df['stride'] == 1) & (df['aligned'] == 1)]
    
    if filtered_df.empty:
        print("Warning: No data available for memory-bound analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Bandwidth utilization vs Array Size
    for implementation in ['scalar', 'vectorized']:
        impl_data = filtered_df[filtered_df['implementation'] == implementation]
        if not impl_data.empty:
            grouped = impl_data.groupby('array_size')
            mean_bandwidth = grouped['bandwidth_gbs'].mean()
            
            axes[0].loglog(mean_bandwidth.index, mean_bandwidth,
                          marker='o', linestyle='-', 
                          label=implementation.capitalize(), alpha=0.8)
    
    # Add estimated memory bandwidth limits
    # These are typical values - adjust based on your system
    bandwidth_limits = {
        'L1 Bandwidth': 100,  # GB/s
        'L2 Bandwidth': 80,   # GB/s
        'L3 Bandwidth': 40,   # GB/s
        'DRAM Bandwidth': 25  # GB/s
    }
    
    for limit_name, limit_value in bandwidth_limits.items():
        axes[0].axhline(y=limit_value, color='gray', linestyle=':', 
                       label=limit_name, alpha=0.7)
    
    axes[0].set_xlabel('Array Size (elements)')
    axes[0].set_ylabel('Bandwidth (GB/s)')
    axes[0].set_title('Memory Bandwidth Utilization')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 2: SIMD efficiency vs Array Size
    efficiency_data = []
    for array_size in filtered_df['array_size'].unique():
        size_data = filtered_df[filtered_df['array_size'] == array_size]
        scalar_data = size_data[size_data['implementation'] == 'scalar']
        vectorized_data = size_data[size_data['implementation'] == 'vectorized']
        
        if len(scalar_data) > 0 and len(vectorized_data) > 0:
            scalar_bw = scalar_data['bandwidth_gbs'].mean()
            vectorized_bw = vectorized_data['bandwidth_gbs'].mean()
            
            if scalar_bw > 0:
                # SIMD efficiency: how much more bandwidth vectorized code uses
                efficiency = vectorized_bw / scalar_bw
                efficiency_data.append((array_size, efficiency))
    
    if efficiency_data:
        sizes, efficiencies = zip(*efficiency_data)
        axes[1].loglog(sizes, efficiencies, marker='s', linestyle='-', 
                      color='purple', label='SIMD Bandwidth Efficiency', alpha=0.8)
        
        axes[1].axhline(y=1.0, color='red', linestyle='--', 
                       label='No efficiency gain', alpha=0.7)
    
    axes[1].set_xlabel('Array Size (elements)')
    axes[1].set_ylabel('Bandwidth Efficiency (Vectorized/Scalar)')
    axes[1].set_title('SIMD Memory Access Efficiency')
    axes[1].legend()
    axes[1].grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'memory_bound_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved memory-bound analysis to {output_path}")

def generate_vectorization_verification(df, output_dir):
    """Generate vectorization verification summary from results"""
    print("Generating vectorization verification summary...")
    
    # Create a simple verification based on performance data
    verification_path = output_dir / 'vectorization_verification.txt'
    
    with open(verification_path, 'w') as f:
        f.write("Vectorization Verification Based on Performance Data\n")
        f.write("=" * 55 + "\n\n")
        
        # Analyze speedup patterns
        f.write("SPEEDUP ANALYSIS (Evidence of Vectorization)\n")
        f.write("-" * 40 + "\n\n")
        
        # Filter for comparable data
        filtered_df = df[(df['stride'] == 1) & (df['aligned'] == 1)]
        
        speedup_results = []
        for (kernel, dtype, size), group in filtered_df.groupby(['kernel', 'data_type', 'array_size']):
            scalar_data = group[group['implementation'] == 'scalar']
            vectorized_data = group[group['implementation'] == 'vectorized']
            
            if len(scalar_data) > 0 and len(vectorized_data) > 0:
                scalar_perf = scalar_data['gflops'].mean()
                vectorized_perf = vectorized_data['gflops'].mean()
                
                if scalar_perf > 0:
                    speedup = vectorized_perf / scalar_perf
                    speedup_results.append({
                        'kernel': kernel,
                        'dtype': dtype,
                        'size': size,
                        'speedup': speedup,
                        'scalar_gflops': scalar_perf,
                        'vectorized_gflops': vectorized_perf
                    })
        
        if speedup_results:
            speedup_df = pd.DataFrame(speedup_results)
            
            f.write("Average Speedup by Kernel:\n")
            for kernel in speedup_df['kernel'].unique():
                kernel_data = speedup_df[speedup_df['kernel'] == kernel]
                avg_speedup = kernel_data['speedup'].mean()
                
                # Expected speedup ranges based on vector width
                expected_speedup = {'f32': 8.0, 'f64': 4.0}  # AVX2 theoretical
                
                f.write(f"  {kernel}: {avg_speedup:.2f}x")
                
                # Check if speedup is reasonable
                for dtype in ['f32', 'f64']:
                    dtype_data = kernel_data[kernel_data['dtype'] == dtype]
                    if len(dtype_data) > 0:
                        dtype_speedup = dtype_data['speedup'].mean()
                        expected = expected_speedup.get(dtype, 1.0)
                        if dtype_speedup >= expected * 0.5:  # At least 50% of theoretical
                            f.write(f" ({dtype}: {dtype_speedup:.2f}) reasonable")
                        else:
                            f.write(f" ({dtype}: {dtype_speedup:.2f}) unreasonable")
                f.write("\n")
            
            f.write("\nEVIDENCE OF VECTORIZATION:\n")
            f.write("-" * 30 + "\n")
            
            # Look for patterns that indicate vectorization
            significant_speedups = speedup_df[speedup_df['speedup'] > 2.0]
            if len(significant_speedups) > 0:
                f.write(f"Found {len(significant_speedups)} measurements with >2x speedup\n")
                f.write(f"Maximum speedup: {significant_speedups['speedup'].max():.2f}x\n")
                
                # Check if speedup correlates with data type (should be higher for f32)
                f32_speedup = speedup_df[speedup_df['dtype'] == 'f32']['speedup'].mean()
                f64_speedup = speedup_df[speedup_df['dtype'] == 'f64']['speedup'].mean()
                
                if f32_speedup > f64_speedup * 1.5:  # f32 should be ~2x f64 for AVX2
                    f.write("f32 speedup > f64 speedup (consistent with vectorization)\n")
                else:
                    f.write("f32 and f64 speedups are similar (may indicate limited vectorization)\n")
            else:
                f.write("No significant speedup detected (>2x)\n")
        
        f.write("\nRECOMMENDATIONS FOR VERIFICATION:\n")
        f.write("-" * 35 + "\n")
        f.write("1. Run analysis/vectorization_verify.py for detailed compiler reports\n")
        f.write("2. Check for vector instructions in disassembly\n")
        f.write("3. Verify compiler flags are enabling vectorization\n")
        f.write("4. Look for 'vectorized loop' messages in compiler output\n")
    
    print(f"Vectorization verification summary saved to {verification_path}")

def main():
    # Print current working directory for debugging
    #print(f"Current working directory: {os.getcwd()}")
    #print(f"Script location: {Path(__file__).parent.absolute()}")
    
    # Set up plot style
    setup_plot_style()
    
    # Get directories using absolute paths
    results_dir = get_results_dir()
    plots_dir = get_plots_dir()
    
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")
    
    # Load and combine results
    print("Loading results...")
    df = load_and_combine_results()
    
    # Check and clean data
    df_clean = check_and_clean_data(df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_performance_vs_size(df_clean, plots_dir)
    plot_baseline_vs_vectorized(df_clean, plots_dir)
    plot_locality_sweep(df_clean, plots_dir)
    plot_alignment_impact(df_clean, plots_dir)
    plot_stride_impact(df_clean, plots_dir)
    plot_data_type_comparison(df_clean, plots_dir)
    plot_compiler_comparison(df_clean, plots_dir)

    # Vectorization verification
    generate_vectorization_verification(df_clean, plots_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df_clean, plots_dir)
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to: {plots_dir}")

if __name__ == "__main__":
    main()