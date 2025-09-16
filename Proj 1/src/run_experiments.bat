@echo off
echo Running SIMD Performance Analysis Experiments...
echo.

if not exist "..\results" mkdir "..\results"
if not exist "..\results\plots" mkdir "..\results\plots"

echo Running scalar version...
scalar_performance.exe scalar 0 nosmt fixfreq > ..\results\scalar_log.txt
if %errorlevel% neq 0 (
    echo Error running scalar version!
    pause
    exit /b %errorlevel%
)
move simd_performance_scalar.csv ..\results\

echo Running vectorized version...
vectorized_performance.exe vectorized 0 nosmt fixfreq > ..\results\vectorized_log.txt
if %errorlevel% neq 0 (
    echo Error running vectorized version!
    pause
    exit /b %errorlevel%
)
move simd_performance_vectorized.csv ..\results\

echo Running AVX2 version...
avx2_performance.exe avx2 0 nosmt fixfreq > ..\results\avx2_log.txt
if %errorlevel% neq 0 (
    echo Error running AVX2 version!
    pause
    exit /b %errorlevel%
)
move simd_performance_avx2.csv ..\results\

echo.
echo All experiments completed!
echo Results saved to ..\results\
pause