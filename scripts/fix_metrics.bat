@echo off
echo.
echo Baby Monitor System - Metrics Dashboard Fix
echo =================================================
echo.
echo This script will run a test server for the metrics dashboard
echo with simulated data to help fix any issues with the metrics display.
echo.
echo Press Ctrl+C to stop the server when finished testing.
echo.

REM Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Make sure you're in the project root directory.
    echo You can still continue, but you may need to have the required packages installed globally.
    pause
)

REM Change to the src directory
echo Starting metrics demo server...
cd src\babymonitor\scripts
python fix_metrics.py

REM If the script exits with an error, wait before closing
if %errorlevel% neq 0 (
    echo.
    echo The metrics demo server exited with an error. Check the output above for details.
    echo.
    pause
)

REM Return to the original directory
cd ..\..\.. 