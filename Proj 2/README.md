# Cache & Memory Performance Profiling

A comprehensive program for analyzing cache and memory performance characteristics using Intel's MLC tool.


## Features

- Zero latency hierarchy measurements
- TLB miss and cache miss impact on simple kernels
- Bandwidth versus pattern stride, and r/w ratios
- Automated plotting and analysis

## Prerequisites
- Windows 10/11
- Python 3.8+
- Administrator privileges for setup
- Python extensions: matplotlib, pandas, numpy, tabulate, scipy, and other OS specific imports

## Quick Start
# Run All as Administrator
1. **Run MLC**
```bash scripts\run_intel_MLC.bat```

2. **Run experiments**

```bash scripts\run_experiments.bat```

3. **View results in results/ directory**

4. **View analysis and methodology in docs/ directory**
