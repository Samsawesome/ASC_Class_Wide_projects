WSL version: 2.5.10.0
Kernel version: 6.6.87.2-1
WSLg version: 1.0.66
MSRDC version: 1.2.6074
Direct3D version: 1.611.1-81528511
DXCore version: 10.0.26100.1-240331-1435.ge-release
Windows version: 10.0.19045.6216
gcc (Ubuntu 11.4.0-1ubuntu1~22.04.2) version:  11.4.0

ran with gcc -O3 -march=native performance_measuring.c -lm -o benchmark

results:
AXPY Performance Benchmark: Y = a*X + Y
============================================
Performing warm-up run...
Warm-up complete.

Performance Measurement Settings:
Array size: 1000000 elements
Iterations: 100
Total operations: 200000000 FLOPs
CPU frequency: 2.50 GHz
Number of runs: 10
============================================

Run  1: Time=0.037785s, GFLOP/s=5.293, CPE=94.462
Run  2: Time=0.034050s, GFLOP/s=5.874, CPE=85.126
Run  3: Time=0.033621s, GFLOP/s=5.949, CPE=84.054
Run  4: Time=0.033062s, GFLOP/s=6.049, CPE=82.656
Run  5: Time=0.033666s, GFLOP/s=5.941, CPE=84.164
Run  6: Time=0.032802s, GFLOP/s=6.097, CPE=82.005
Run  7: Time=0.034127s, GFLOP/s=5.860, CPE=85.318
Run  8: Time=0.033992s, GFLOP/s=5.884, CPE=84.980
Run  9: Time=0.033550s, GFLOP/s=5.961, CPE=83.876
Run 10: Time=0.033286s, GFLOP/s=6.009, CPE=83.215

============================================
STATISTICAL ANALYSIS RESULTS (10 runs)
============================================
TIME:
  Average:   0.033994 ± 0.001328 seconds
  Best:      0.032802 seconds
  Worst:     0.037785 seconds
  Range:     0.004983 seconds (14.7% variation)

GFLOP/s:
  Average:   5.892 ± 0.212
  Best:      6.097
  Worst:     5.293
  Range:     0.804 GFLOP/s

CYCLES PER ELEMENT (CPE):
  Average:   84.986 cycles/element
  Theoretical best: ~0.5 cycles/element (2 FLOPS/cycle on modern CPUs)

PERFORMANCE SUMMARY:
  Peak efficiency: 0.6% of theoretical best
  Consistency:     3.9% (lower is better)
