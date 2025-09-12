@echo off
echo Building SIMD Performance Analysis Project...
echo.

echo Building scalar version...
clang -O2 -fno-tree-vectorize -mno-sse -mno-avx -o scalar_performance.exe simd_performance.c -lpdh -ladvapi32

if %errorlevel% neq 0 (
    echo Error building scalar version!
    pause
    exit /b %errorlevel%
)

echo Building auto-vectorized version...
clang -O3 -march=native -ffast-math -o vectorized_performance.exe simd_performance.c -lpdh -ladvapi32

if %errorlevel% neq 0 (
    echo Error building vectorized version!
    pause
    exit /b %errorlevel%
)

echo Building AVX2 version...
clang -O3 -mavx2 -mfma -ffast-math -o avx2_performance.exe simd_performance.c -lpdh -ladvapi32

if %errorlevel% neq 0 (
    echo Error building AVX2 version!
    pause
    exit /b %errorlevel%
)

echo.
echo All builds completed successfully!
pause