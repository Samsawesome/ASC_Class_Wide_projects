@echo off
REM Drive preconditioning script

set DRIVE_LETTER=D:
if "%~1" neq "" set DRIVE_LETTER=%~1

set TEST_SIZE=10G
if "%~2" neq "" set TEST_SIZE=%~2

echo Preconditioning drive %DRIVE_LETTER%...

REM Create precondition FIO config
echo [global] > precondition.fio
echo ioengine=windowsaio >> precondition.fio
echo direct=1 >> precondition.fio
echo size=%TEST_SIZE% >> precondition.fio
echo filename=%DRIVE_LETTER%\testfile >> precondition.fio
echo runtime=120s >> precondition.fio
echo. >> precondition.fio
echo [precondition_sequential] >> precondition.fio
echo rw=write >> precondition.fio
echo bs=1M >> precondition.fio
echo iodepth=32 >> precondition.fio
echo numjobs=4 >> precondition.fio
echo. >> precondition.fio
echo [precondition_random] >> precondition.fio
echo rw=randwrite >> precondition.fio
echo bs=4k >> precondition.fio
echo iodepth=32 >> precondition.fio
echo numjobs=8 >> precondition.fio

REM Run preconditioning
echo Running drive preconditioning...
fio precondition.fio

REM Cleanup
del precondition.fio

echo Preconditioning complete!
pause