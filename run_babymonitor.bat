@echo off
setlocal enabledelayedexpansion

echo ======================================
echo       Baby Monitor System Launcher
echo ======================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher and try again.
    pause
    exit /b 1
)

REM Check if the virtual environment exists
if not exist venv\ (
    echo Virtual environment not found. Setting up...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
    echo Setup complete!
) else (
    call venv\Scripts\activate
)

:menu
cls
echo ======================================
echo       Baby Monitor System Launcher
echo ======================================
echo.
echo Please select an option:
echo.
echo [1] Start Baby Monitor (Normal Mode)
echo [2] Start Baby Monitor (Developer Mode)
echo [3] Run API Test
echo [4] Backup and Restore
echo [5] Check Requirements
echo [6] Exit
echo.
set /p choice=Enter your choice (1-6): 

if "%choice%"=="1" (
    echo.
    echo Starting Baby Monitor in Normal Mode...
    python -m babymonitor.launcher -i gui -m normal
    goto menu
)
if "%choice%"=="2" (
    echo.
    echo Starting Baby Monitor in Developer Mode...
    python -m babymonitor.launcher -i gui -m dev
    goto menu
)
if "%choice%"=="3" (
    echo.
    echo Running API Tests...
    python tests\updated\api_test.py
    echo.
    pause
    goto menu
)
if "%choice%"=="4" (
    call tools\backup_restore.bat
    goto menu
)
if "%choice%"=="5" (
    echo.
    echo Checking requirements...
    pip list | findstr "opencv-python PyQt5 torch transformers sounddevice librosa"
    echo.
    pause
    goto menu
)
if "%choice%"=="6" (
    echo.
    echo Thank you for using Baby Monitor System.
    echo Exiting...
    goto :eof
)

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu 