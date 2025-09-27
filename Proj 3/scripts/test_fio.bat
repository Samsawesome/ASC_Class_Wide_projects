@echo off
echo Testing FIO installation...

fio --version
if errorlevel 1 (
    echo FIO not found in PATH
    exit /b 1
)

echo.
echo Testing basic FIO command...
fio --name=test --ioengine=windowsaio --filename=D:\testfile --size=1G --rw=read --bs=4k --iodepth=1 --runtime=5s --output-format=json
if errorlevel 1 (
    echo Basic FIO test failed
    exit /b 1
)

echo.
echo FIO test completed successfully!
pause