# SIMD Advantage Profiling

A comprehensive program for analyzing vectorized SIMD cache speedups versus  scalar instructions.

## Features
- Baseline scalar vesus SIMD speedup across multiple common kernels
- Analysis across multiple common kernels for vector alignment and locality sweeps
- Comparisons between different data types and stride impacts
- Automated vectorization reports and disassemblies
- Automated plotting and analysis


## Prerequisites
- Windows 10/11
- Python 3.8+
- Administrator privileges for setup
- Python extensions: pandas, matplotlib, glob, numpy, seaborn, and other OS specific imports

## Quick Start
# Run All as Administrator
1. **Compile Step**
```bash src\build.bat```

2. **Run experiments**

```bash scripts/run_experiments.bat```
3. **Analyze Data and Generate Plots**

```bash py analysis\analyze_results.py```

4. **Generate Roofline Analysis**

```bash py analysis\roofline_analysis.py```

5. **Generate Vectorization Verification Reports and Disassemblies**

```bash analysis\run_vectorization_verify.bat```

6. **View results in results/ directory**

7. **View vectorization results in analysis/ directory**

8. **View analysis and methodology in docs/ directory**

