@echo off
mkdir results\raw_data 2>nul

echo Compiling benchmarks...
clang-cl /O2 /Iinclude benchmarks\latency.cpp /link pdh.lib /out:latency.exe
clang-cl /O2 /Iinclude benchmarks\bandwidth.cpp /link pdh.lib /out:bandwidth.exe
clang-cl /O2 /Iinclude benchmarks\loaded_latency.cpp /link pdh.lib /out:loaded_latency.exe
clang-cl /O2 /Iinclude benchmarks\kernel_bench.cpp /link pdh.lib /out:kernel_bench.exe

echo Running latency tests...
latency.exe > results\raw_data\latency.txt

echo Running bandwidth tests...
bandwidth.exe > results\raw_data\bandwidth.txt

echo Running loaded latency tests...
loaded_latency.exe > results\raw_data\loaded_latency.txt

echo Running kernel benchmark tests...
kernel_bench.exe > results\raw_data\kernel_bench.txt

echo Processing results...
py scripts\plot_results.py