@echo off
echo Checking Python dependencies...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed. Please install Python first.
    pause
    exit /b 1
)

:: Check if tkinter is available
python -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo Error: tkinter is not available.
    echo Please install Python with tkinter support.
    echo You can reinstall Python and make sure to check "tcl/tk and IDLE" during installation.
    pause
    exit /b 1
)

:: Check if pip is installed
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip is not installed. Please install pip first.
    pause
    exit /b 1
)

:: Install required packages
echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r src/babymonitor/tools/requirements.txt

:: Check if installation was successful
if errorlevel 1 (
    echo Error: Failed to install required packages.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo Starting Emotion Model Manager...
python src/babymonitor/tools/emotion_model_manager.py
pause 