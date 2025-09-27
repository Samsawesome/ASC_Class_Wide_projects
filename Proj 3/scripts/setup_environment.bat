@echo off
REM Simple SSD Benchmark Setup - Customize FIO path for your system

echo Setting up SSD Benchmark Environment...

REM Set this to your actual FIO installation path
set FIO_PATH=C:\Program Files\fio

if not exist "%FIO_PATH%\fio.exe" (
    echo Error: FIO not found at %FIO_PATH%
    echo Please edit this script to point to your FIO installation directory.
    pause
    exit /b 1
)

echo Adding FIO to PATH...
setx PATH "%PATH%;%FIO_PATH%" /m
set PATH=%PATH%;%FIO_PATH%

REM Verify FIO
fio --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('fio --version') do echo FIO: %%i
) else (
    echo Error: FIO verification failed.
    pause
    exit /b 1
)

REM Install Python dependencies
echo Installing Python packages...
pip install -r requirements.txt

echo Setup complete!
echo Run: py run_benchmarks.py --drive D: --size 10G --runs 3
pause