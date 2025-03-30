@echo off
echo ===============================================
echo    Baby Monitor System - Windows Installer
echo ===============================================
echo.

:: Check Python and create virtual environment
py -3.11 --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 is required. Please install from:
    echo https://www.python.org/downloads/release/python-3115/
    echo.
    pause
    exit /b 1
)

echo Setting up Python environment...
py -3.11 -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

:: Install packages
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install numpy==1.24.3 opencv-python-headless==4.8.1.78
python -m pip install eventlet==0.33.3 flask-socketio==5.3.6 werkzeug==2.3.7 python-engineio==4.5.1 python-socketio==5.8.0

:: Run installer
echo Running installer...
python install.py %*
if %errorlevel% neq 0 (
    echo Installation failed. See INSTALL.md for help.
    pause
    exit /b 1
)

:: Create start script
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
echo Installation complete!
echo Run start.bat to launch the Baby Monitor System
echo Then open http://localhost:5000 in your browser
echo.
pause