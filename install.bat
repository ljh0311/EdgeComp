@echo off
echo ===============================================
echo    Baby Monitor System - Windows Installer
echo ===============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or newer from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYVER=%%I
echo Detected Python version: %PYVER%

:: Run the installer
echo.
echo Starting the Baby Monitor System installer...
echo.
python install.py

if %errorlevel% neq 0 (
    echo.
    echo Installation failed. Please check the error messages above.
    echo For more information, see INSTALL.md
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo.
echo Installation completed successfully!
echo.
echo You can now start the Baby Monitor System using:
echo 1. The desktop shortcut (if created during installation)
echo 2. Command line: python main.py
echo.
echo Press any key to exit...
pause >nul 