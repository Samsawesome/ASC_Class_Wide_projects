@echo off
echo Running Intel Memory Latency Checker...
echo.

mlc.exe --latency_matrix
mlc.exe --bandwidth_matrix
mlc.exe --idle_latency

echo.
echo Results saved to mlc_output.txt
pause