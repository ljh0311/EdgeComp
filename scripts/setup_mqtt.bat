@echo off
setlocal enabledelayedexpansion

echo === Baby Monitor MQTT Setup for Windows ===
echo This script will help you set up Mosquitto MQTT broker for the Baby Monitor.
echo.

set MQTT_DIR=%PROGRAMFILES%\mosquitto
set MOSQUITTO_EXE=%MQTT_DIR%\mosquitto.exe
set DOWNLOAD_URL=https://mosquitto.org/files/binary/win64/mosquitto-2.0.15-install-windows-x64.exe
set INSTALLER=%TEMP%\mosquitto-installer.exe

:: Check if Mosquitto is already installed
if exist "%MOSQUITTO_EXE%" (
    echo Mosquitto is already installed at %MQTT_DIR%
    goto :CONFIGURE
)

echo Mosquitto MQTT broker is not installed. Would you like to download and install it?
echo.
echo 1. Yes, download and install Mosquitto
echo 2. No, I'll install it manually
echo.
set /p CHOICE="Enter your choice (1 or 2): "

if "%CHOICE%"=="1" (
    echo.
    echo Downloading Mosquitto installer...
    powershell -Command "Invoke-WebRequest -Uri %DOWNLOAD_URL% -OutFile '%INSTALLER%'"
    
    if %errorlevel% neq 0 (
        echo Failed to download Mosquitto installer.
        echo Please download it manually from https://mosquitto.org/download/
        goto :END
    )
    
    echo.
    echo Running Mosquitto installer...
    echo Please complete the installation wizard.
    echo.
    start /wait "" "%INSTALLER%"
    
    if not exist "%MOSQUITTO_EXE%" (
        echo.
        echo Could not detect Mosquitto installation.
        echo If you installed to a non-standard location, please configure it manually.
        goto :END
    )
    
    echo.
    echo Mosquitto installation completed.
) else (
    echo.
    echo Please install Mosquitto manually from https://mosquitto.org/download/
    echo After installation, restart this script.
    goto :END
)

:CONFIGURE
echo.
echo Configuring Mosquitto...

:: Create config file if it doesn't exist
set CONFIG_FILE=%MQTT_DIR%\mosquitto.conf
if not exist "%CONFIG_FILE%" (
    echo # Baby Monitor MQTT Broker Configuration > "%CONFIG_FILE%"
    echo listener 1883 0.0.0.0 >> "%CONFIG_FILE%"
    echo allow_anonymous true >> "%CONFIG_FILE%"
)

:: Check if Mosquitto service is installed
sc query mosquitto > nul 2>&1
if %errorlevel% equ 0 (
    echo Mosquitto service is already installed.
) else (
    echo Installing Mosquitto as a service...
    "%MQTT_DIR%\mosquitto.exe" install
    sc config mosquitto start= auto
)

:: Start Mosquitto service
echo Starting Mosquitto service...
net start mosquitto
if %errorlevel% neq 0 (
    echo.
    echo Failed to start Mosquitto service. 
    echo Starting Mosquitto in console mode instead.
    echo Press Ctrl+C to stop the broker when done.
    echo.
    "%MQTT_DIR%\mosquitto.exe" -c "%CONFIG_FILE%"
) else (
    echo.
    echo Mosquitto service started successfully.
)

echo.
echo MQTT Broker Status:
echo -------------------
netstat -an | findstr 1883

:END
echo.
echo To run the Baby Monitor with MQTT support, use:
echo python main.py --mqtt-host localhost --mqtt-port 1883
echo.
echo For the Baby Monitor Client, use:
echo python baby_client.py --host localhost --port 5000 --mqtt-host localhost --mqtt-port 1883
echo.
pause
