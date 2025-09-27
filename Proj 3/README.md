# SSD Performance Characterization Project

A comprehensive toolset for analyzing SSD performance characteristics using FIO.


## Features

- Zero-queue depth latency measurements
- Block size sweeps (4KB - 1MB)
- Read/write mix analysis
- Queue depth scalability testing
- Tail latency characterization
- Automated plotting and analysis

## Prerequisites
- Windows 10/11
- Python 3.8+
- Empty partition (D: drive recommended)
- Administrator privileges for setup
- FIO installed and added to PATH

## Quick Start

1. **Run the setup script as Administrator::**
   ```bash     scripts\setup_environment.bat    ```

2. **Install Python dependencies:**
    ```bash    pip install -r requirements.txt    ```
    
3. **Run all benchmarks:**
```bash py run_benchmarks.py --drive D: --size 10G --runs 3```

4. **View results in results/ directory**
```bash 
py scripts/parse_results.py
py scripts/plot_results.py
```

below is all wrong

## Project Structure
- scripts/: Python scripts for automation and analysis

- configs/: FIO configuration templates

- results/: Raw and processed experiment data

- plots/: Generated visualizations

- analysis/: Advanced analysis notebooks

## Experiments
- Zero-queue baselines: QD=1 latency measurements

- Block size sweep: 4KB to 1MB performance analysis

- RW mix sweep: 100%R to 100%W performance impact

- Queue depth sweep: Throughput-latency trade-off curves

## Output Metrics
- IOPS and bandwidth measurements

- Average and percentile latencies (p50, p95, p99, p99.9)

- Throughput-latency trade-off analysis

- Block size optimization insights

