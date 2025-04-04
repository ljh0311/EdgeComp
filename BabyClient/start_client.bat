@echo off
setlocal enabledelayedexpansion

:: Baby Monitor Client Starter Script
echo Baby Monitor Client - Start Script

:: Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Check for required packages
python -c "import PyQt5" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    python -m pip install PyQt5 PyQtWebEngine python-socketio paho-mqtt requests
)

:: Get the script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Default values
set HOST=localhost
set PORT=5000
set MQTT_HOST=%HOST%
set MQTT_PORT=1883
set RESOLUTION=640x480

:: Parse command line arguments
if not "%~1"=="" set HOST=%~1
if not "%~2"=="" set PORT=%~2
if not "%~3"=="" set MQTT_HOST=%~3
if not "%~4"=="" set MQTT_PORT=%~4
if not "%~5"=="" set RESOLUTION=%~5

echo.
echo Starting Baby Monitor Client with the following settings:
echo Server Host: %HOST%
echo Server Port: %PORT%
echo MQTT Host: %MQTT_HOST%
echo MQTT Port: %MQTT_PORT%
echo Resolution: %RESOLUTION%
echo.
echo Press Ctrl+C to stop the client
echo.

:: Start the client
python baby_client.py --host %HOST% --port %PORT% --mqtt-host %MQTT_HOST% --mqtt-port %MQTT_PORT% --resolution %RESOLUTION%

if %errorlevel% neq 0 (
    echo.
    echo Client exited with error code: %errorlevel%
    echo.
    echo Usage: start_client.bat [host] [port] [mqtt_host] [mqtt_port] [resolution]
    echo Example: start_client.bat localhost 5000 localhost 1883 640x480
)

pause 