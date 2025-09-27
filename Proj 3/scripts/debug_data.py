#!/usr/bin/env python3
"""
Debug script to check the structure of parsed FIO data
"""

import pandas as pd
import json
from pathlib import Path
from src.result_parser import ResultParser

def debug_parsed_data():
    """Check what columns are being created by the parser"""
    # Find the most recent results file
    results_dir = Path('results')
    if not results_dir.exists():
        print("No results directory found")
        return
    
    result_files = list(results_dir.glob('zero_queue_*.csv'))
    if not result_files:
        print("No zero_queue results found")
        return
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Analyzing: {latest_file}")
    
    df = pd.read_csv(latest_file)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Job names: {df['job_name'].unique()}")
    
    if 'block_size' in df.columns:
        print(f"Block sizes: {df['block_size'].unique()}")
    else:
        print("No 'block_size' column found")
    
    if 'queue_depth' in df.columns:
        print(f"Queue depths: {df['queue_depth'].unique()}")
    else:
        print("No 'queue_depth' column found")
    
    # Show first few rows
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    
    # Show sample of parsed parameters
    print("\nSample job name parsing:")
    parser = ResultParser()
    for job_name in df['job_name'].unique()[:3]:
        params = parser._parse_job_name_improved(job_name, {})
        print(f"  '{job_name}' -> {params}")

if __name__ == "__main__":
    debug_parsed_data()