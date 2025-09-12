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
    print("\nData types:")
    for col in df_clean.columns:
        print(f"  {col}: {df_clean[col].dtype}")
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
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
    
    # Plot 3: Bandwidth vs Array Size
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
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which="both", ls="--")
    
    # Plot 4: CPE vs Array Size
    ax = axes[3]
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
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
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
    print("Generating summary report...")
    
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

def main():
    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent.absolute()}")
    
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
    plot_alignment_impact(df_clean, plots_dir)
    plot_stride_impact(df_clean, plots_dir)
    plot_data_type_comparison(df_clean, plots_dir)
    plot_compiler_comparison(df_clean, plots_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df_clean, plots_dir)
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to: {plots_dir}")

if __name__ == "__main__":
    main()