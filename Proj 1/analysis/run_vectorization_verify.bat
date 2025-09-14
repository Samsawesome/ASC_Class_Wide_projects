@echo off
echo Running Vectorization Verification...
echo.

py vectorization_verify.py

if %errorlevel% neq 0 (
    echo Error running vectorization verification!
    pause
    exit /b %errorlevel%
)

echo.
echo Vectorization verification completed!
echo Check the reports in: analysis/vectorization_reports/
pause