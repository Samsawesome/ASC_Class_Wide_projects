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
# Run as Administrator
1. **Run the setup script as Administrator::**
   ```bash     scripts\setup_environment.bat    ```
    
2. **Run all benchmarks:**
```bash py run_benchmarks.py --drive D: --size 10G --runs 3```

3. **View results in results/ and plots/ directories**

4. **View analysis and methodology in docs/ directory**

## Additional Notes
Two files are two big to upload to GitHub, so there will be missing files upon downloading this project from GitHub. A rerun of the experiments by running the quick start commands above is recommended to generate all files.


