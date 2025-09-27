@echo off
echo FIO Debug Script
echo.

echo 1. Testing FIO with thread option:
fio --thread --name=debug_test --ioengine=windowsaio --filename=D:\debug.file --size=100M --rw=read --bs=4k --iodepth=1 --runtime=5s --output-format=json
echo.

echo 2. Testing with a config file:
echo [global] > debug_config.fio
echo ioengine=windowsaio >> debug_config.fio
echo size=100M >> debug_config.fio
echo filename=D:\debug_config.file >> debug_config.fio
echo thread >> debug_config.fio
echo. >> debug_config.fio
echo [test] >> debug_config.fio
echo rw=read >> debug_config.fio
echo bs=4k >> debug_config.fio
echo iodepth=1 >> debug_config.fio
echo runtime=5s >> debug_config.fio

fio --output-format=json debug_config.fio
del debug_config.fio
echo.

pause