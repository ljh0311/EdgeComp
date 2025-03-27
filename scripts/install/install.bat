@echo off
echo ===============================================
echo    Baby Monitor System - Windows Installer
echo ===============================================
echo.

:: Check for Python launcher
where py >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python launcher (py^) not found.
    echo Please install Python from https://www.python.org/downloads/release/python-3115/
    echo Make sure to check "Add python.exe to PATH" during installation.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: Check for Python 3.11
echo Checking for Python 3.11...
py -3.11 --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 is not installed.
    echo Please install Python 3.11 from https://www.python.org/downloads/release/python-3115/
    echo.
    echo Current Python versions installed:
    py --list
    echo.
    echo After installing Python 3.11:
    echo 1. Make sure to check "Add python.exe to PATH" during installation
    echo 2. Run 'py -3.11 -m pip install --upgrade pip'
    echo 3. Run this script again
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: Check for Visual C++ Redistributable
if not exist "C:\Windows\System32\vcruntime140.dll" (
    echo WARNING: Visual C++ Redistributable might be missing.
    echo Please download and install from: https://aka.ms/vs/16/release/vc_redist.x64.exe
    echo.
    echo Press any key to continue anyway...
    pause >nul
)

:: Show selected Python version
echo Using Python 3.11:
py -3.11 --version
echo.

:: Create virtual environment
echo Setting up Python virtual environment...
py -3.11 -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    echo Please make sure you have sufficient permissions.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: Activate virtual environment and install packages
echo Installing required packages...
call venv\Scripts\activate.bat

:: Upgrade pip first
python -m pip install --upgrade pip

:: Install numpy first (required for opencv)
python -m pip install numpy==1.24.3

:: Install opencv-python with a compatible version
python -m pip install opencv-python-headless==4.8.1.78

:: Install other packages
python -m pip install eventlet==0.33.3 flask-socketio==5.3.6 werkzeug==2.3.7 python-engineio==4.5.1 python-socketio==5.8.0

:: Run the installer
echo.
echo Starting the Baby Monitor System installer...
echo.
python install.py %*

if %errorlevel% neq 0 (
    echo.
    echo Installation failed. Please check the error messages above.
    echo For more information, see INSTALL.md
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: Create start script
echo.
echo Creating start script...
(
echo @echo off
echo call venv\Scripts\activate.bat
echo set PYTHONPATH=.
echo set EVENTLET_NO_GREENDNS=yes
echo python main.py --mode normal
echo pause
) > start.bat

echo.
echo Installation completed successfully!
echo.
echo You can now start the Baby Monitor System using:
echo 1. Run: python main.py --mode normal
echo 2. Or use the generated start.bat file
echo 3. Open http://localhost:5000 in your web browser
echo.
echo For more information, see README.md
echo.
echo Press any key to exit...
pause >nul 