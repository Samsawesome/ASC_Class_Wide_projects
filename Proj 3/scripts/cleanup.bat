@echo off
REM SSD Benchmark Cleanup Script

set DRIVE_LETTER=D:
set REMOVE_FIO=0

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :endparse
if "%~1"=="/removefio" set REMOVE_FIO=1
if "%~1"=="/drive" (
    set DRIVE_LETTER=%~2
    shift
)
shift
goto :parse_args

:endparse

echo Cleaning up benchmark files...

REM Remove test files
if exist "%DRIVE_LETTER%\testfile" (
    del "%DRIVE_LETTER%\testfile" /f /q
    echo Removed: %DRIVE_LETTER%\testfile
)

if exist "%DRIVE_LETTER%\benchmark" (
    rmdir "%DRIVE_LETTER%\benchmark" /s /q
    echo Removed: %DRIVE_LETTER%\benchmark
)

if exist "tail_latency.log" (
    del "tail_latency.log" /f /q
    echo Removed: tail_latency.log
)

if exist "*.fio.tmp" del "*.fio.tmp" /f /q
if exist "results\*.tmp" del "results\*.tmp" /f /q
if exist "plots\*.tmp" del "plots\*.tmp" /f /q

REM Remove FIO installation if requested
if %REMOVE_FIO% equ 1 (
    echo Uninstalling FIO...
    wmic product where "name like '%%fio%%'" call uninstall /nointeractive >nul 2>&1
    if %errorLevel% equ 0 (
        echo Uninstalled FIO
    ) else (
        echo Could not uninstall FIO automatically
        echo Please uninstall manually from 'Add or Remove Programs'
    )
)

REM Clean Python cache files
if exist "__pycache__" rmdir "__pycache__" /s /q
if exist "src\__pycache__" rmdir "src\__pycache__" /s /q
if exist "*.pyc" del "*.pyc" /f /q
if exist "results\.cache" rmdir "results\.cache" /s /q
if exist "plots\.cache" rmdir "plots\.cache" /s /q

echo Cleanup complete!

REM Display remaining data usage
for /f "skip=1" %%i in ('wmic logicaldisk where "deviceid='%DRIVE_LETTER%'" get size,freespace') do (
    set FREE=%%i
    set TOTAL=%%j
)

set /a FREE_GB=%FREE%/1073741824
set /a TOTAL_GB=%TOTAL%/1073741824

echo Drive %DRIVE_LETTER% free space: %FREE_GB% GB / %TOTAL_GB% GB

pause