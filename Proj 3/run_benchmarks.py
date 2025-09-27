#!/usr/bin/env python3
"""
Main benchmark runner for SSD characterization
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
from pathlib import Path
from src.fio_runner import FioRunner
from src.result_parser import ResultParser
from src.plot_generator import PlotGenerator

def check_fio_available():
    """Check if FIO is available in the system PATH"""
    if not shutil.which('fio'):
        print("ERROR: FIO not found in PATH.")
        print("Please ensure FIO is installed and available in your system PATH.")
        return False
    
    try:
        result = subprocess.run(['fio', '--version'], 
                              capture_output=True, text=True, timeout=30, check=True)
        print(f"✓ FIO detected: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"ERROR: FIO check failed: {e}")
        return False

def validate_parsed_data(df, config_name):
    """Validate that the parsed data has required columns"""
    required_columns = ['job_name', 'run_id']
    
    print(f"Validating {config_name} data...")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        return False
    
    # Check for key analysis columns
    analysis_columns = ['block_size', 'queue_depth', 'operation']
    available_analysis_cols = [col for col in analysis_columns if col in df.columns]
    print(f"Available analysis columns: {available_analysis_cols}")
    
    if 'block_size' in df.columns:
        print(f"Block sizes found: {df['block_size'].unique()}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='SSD Benchmark Suite')
    parser.add_argument('--drive', default='D:', help='Target drive letter')
    parser.add_argument('--size', default='10G', help='Test file size')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per test')
    parser.add_argument('--skip-precondition', action='store_true', 
                       help='Skip drive preconditioning')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Check if FIO is available
    print("=== FIO Environment Check ===")
    if not check_fio_available():
        return 1
    
    # Create directories
    Path('results').mkdir(exist_ok=True)
    Path('plots').mkdir(exist_ok=True)
    
    runner = FioRunner(args.drive, args.size)
    result_parser = ResultParser()
    plotter = PlotGenerator()
    
    if not args.skip_precondition:
        print("\n=== Preconditioning Drive ===")
        runner.precondition_drive()
    
    benchmarks = [
        ('zero_queue', 'Zero Queue Depth Baseline'),
        ('block_size_sweep', 'Block Size Sweep'),
        ('rw_mix_sweep', 'Read/Write Mix Sweep'), 
        ('queue_depth_sweep', 'Queue Depth Sweep'),
        ('tail_latency', 'Tail Latency Analysis')
    ]
    
    all_successful = True
    
    for config_name, description in benchmarks:
        print(f"\n=== Running {description} ===")
        try:
            results = runner.run_benchmark(config_name, num_runs=args.runs)
            
            if results:
                # Parse and save results
                df = result_parser.parse_results(results)
                timestamp = int(time.time())
                
                # Validate the parsed data
                if not validate_parsed_data(df, config_name):
                    print(f"Warning: Data validation issues for {config_name}")
                
                output_file = f'results/{config_name}_{timestamp}.csv'
                df.to_csv(output_file, index=False)
                print(f"✓ Saved results to {output_file}")
                
                # Generate plots if requested
                if not args.skip_plots:
                    plotter.generate_plots(df, config_name)
                    print(f"✓ Generated plots for {description}")
                else:
                    print(f"⏭️  Skipped plots for {description}")
                    
            else:
                print(f"✗ No results for {description}")
                all_successful = False
                
        except Exception as e:
            print(f"✗ Error running {description}: {e}")
            import traceback
            traceback.print_exc()
            all_successful = False
    
    if all_successful:
        print("\n=== Benchmark Complete Successfully ===")
    else:
        print("\n=== Benchmark Completed with Errors ===")
        
    print("Results saved to 'results/' directory")
    if not args.skip_plots:
        print("Plots saved to 'plots/' directory")
    
    # Run debug data analysis
    if args.debug:
        print("\n=== Debug Data Analysis ===")
        try:
            from scripts.debug_data import debug_parsed_data
            debug_parsed_data()
        except ImportError:
            print("Debug script not available")
    
    return 0 if all_successful else 1

if __name__ == "__main__":
    sys.exit(main())