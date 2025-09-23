@echo off
mkdir ..\results\raw_data 2>nul

echo Compiling benchmarks...
clang -O2 -Iinclude ..\benchmarks\latency.cpp -lpdh -olatency.exe -lAdvapi32
clang -O2 -Iinclude ..\benchmarks\bandwidth.cpp -lpdh -obandwidth.exe -lAdvapi32
clang -O2 -Iinclude ..\benchmarks\loaded_latency.cpp -lpdh -oloaded_latency.exe -lAdvapi32
clang -O2 -Iinclude ..\benchmarks\kernel_bench.cpp -lpdh -okernel_bench.exe -lAdvapi32
clang -O2 -Iinclude ..\benchmarks\pattern_latency.cpp -lpdh -opattern_latency.exe -lAdvapi32

echo Running latency tests...
latency.exe > ..\results\raw_data\latency.txt

echo Running bandwidth tests...
bandwidth.exe > ..\results\raw_data\bandwidth.txt

echo Running loaded latency tests...
loaded_latency.exe > ..\results\raw_data\loaded_latency.txt

echo Running kernel benchmark tests...
kernel_bench.exe > ..\results\raw_data\kernel_bench.txt

echo Running pattern latency tests...
pattern_latency.exe > ..\results\raw_data\pattern_latency.txt

echo Processing results...
py plot_results.py