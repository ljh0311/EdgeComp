@echo off
echo.
echo Baby Monitor System - Metrics Dashboard Demo
echo =================================================
echo.
echo This script will run a standalone demo server for the metrics dashboard
echo with simulated data to help fix and test the metrics display.
echo.
echo Press Ctrl+C to stop the server when finished testing.
echo.

REM Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found, continuing with system Python...
)

REM Run the metrics demo script
echo Starting metrics demo server...
python scripts\metrics_demo.py

REM If the script exits with an error, wait before closing
if %errorlevel% neq 0 (
    echo.
    echo The metrics demo server exited with an error. Check the output above for details.
    echo.
    pause
) 