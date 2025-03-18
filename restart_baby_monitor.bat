@echo off
echo ===================================
echo Baby Monitor System Restart Utility
echo ===================================
echo.

REM Kill any existing Python processes running the baby monitor
echo Stopping any running Baby Monitor processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Baby Monitor*" 2>nul
timeout /t 2 /nobreak >nul

REM Clear screen
cls
echo ===================================
echo Baby Monitor System Restart Utility
echo ===================================
echo.
echo Starting Baby Monitor in development mode...
echo.

REM Start the baby monitor in a new window
start "Baby Monitor - YOLOv8 Edition" cmd /k python main.py --mode dev

echo.
echo Baby Monitor started successfully!
echo.
echo The web interface should be available at:
echo http://localhost:5000
echo.
echo Press any key to exit...
pause >nul 