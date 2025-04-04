@echo off
setlocal enabledelayedexpansion

echo === Baby Monitor Setup for Raspberry Pi ===
echo.
echo This script will prepare files to be transferred to your Raspberry Pi.
echo.

set MODEL_DIR=src\babymonitor\models
set MODEL_FILE=%MODEL_DIR%\yolov8n.pt
set MODEL_URL=https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

:: Create models directory if it doesn't exist
if not exist %MODEL_DIR% (
    echo Creating models directory...
    mkdir %MODEL_DIR%
)

:: Check if model file exists
if not exist %MODEL_FILE% (
    echo YOLOv8n model not found. Downloading...
    powershell -Command "Invoke-WebRequest -Uri %MODEL_URL% -OutFile '%MODEL_FILE%'"
    
    if %errorlevel% neq 0 (
        echo Failed to download model file.
        goto :ERROR
    )
    echo Model downloaded successfully.
) else (
    echo YOLOv8n model already exists.
)

echo.
echo Creating Raspberry Pi setup script...

:: Create a Raspberry Pi setup script
set PI_SCRIPT=setup_raspberry_pi.sh
echo #!/bin/bash > %PI_SCRIPT%
echo. >> %PI_SCRIPT%
echo echo "=== Baby Monitor Setup for Raspberry Pi ===" >> %PI_SCRIPT%
echo echo "This script will install dependencies and set up the Baby Monitor system." >> %PI_SCRIPT%
echo echo. >> %PI_SCRIPT%

echo # Install required packages >> %PI_SCRIPT%
echo sudo apt-get update >> %PI_SCRIPT%
echo sudo apt-get install -y python3-pip python3-opencv libopencv-dev libatlas-base-dev >> %PI_SCRIPT%
echo. >> %PI_SCRIPT%

echo # Install Python dependencies >> %PI_SCRIPT%
echo pip3 install -r requirements.txt >> %PI_SCRIPT%
echo. >> %PI_SCRIPT%

echo # Create models directory >> %PI_SCRIPT%
echo mkdir -p src/babymonitor/models >> %PI_SCRIPT%
echo. >> %PI_SCRIPT%

echo # Check if model file exists >> %PI_SCRIPT%
echo if [ ! -f "src/babymonitor/models/yolov8n.pt" ]; then >> %PI_SCRIPT%
echo     echo "YOLOv8n model not found. Downloading..." >> %PI_SCRIPT%
echo     wget -O src/babymonitor/models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt >> %PI_SCRIPT%
echo     if [ $? -ne 0 ]; then >> %PI_SCRIPT%
echo         echo "Failed to download model file." >> %PI_SCRIPT%
echo         exit 1 >> %PI_SCRIPT%
echo     fi >> %PI_SCRIPT%
echo     echo "Model downloaded successfully." >> %PI_SCRIPT%
echo else >> %PI_SCRIPT%
echo     echo "YOLOv8n model already exists." >> %PI_SCRIPT%
echo fi >> %PI_SCRIPT%
echo. >> %PI_SCRIPT%

echo echo. >> %PI_SCRIPT%
echo echo "Setup completed successfully!" >> %PI_SCRIPT%
echo echo "You can now run the Baby Monitor with:" >> %PI_SCRIPT%
echo echo "python3 main.py --device pi" >> %PI_SCRIPT%
echo echo. >> %PI_SCRIPT%

:: Make the script executable
echo chmod +x setup_raspberry_pi.sh >> %PI_SCRIPT%

echo.
echo Setup script created: %PI_SCRIPT%
echo.
echo Instructions:
echo 1. Transfer the entire project directory to your Raspberry Pi
echo 2. On your Raspberry Pi, navigate to the project directory
echo 3. Run the setup script: ./setup_raspberry_pi.sh
echo 4. After setup completes, run: python3 main.py --device pi
echo.
echo If you still get the "person detection model not available" error,
echo try running with explicit model path:
echo python3 main.py --device pi --person-model src/babymonitor/models/yolov8n.pt
echo.
goto :END

:ERROR
echo An error occurred during setup.
exit /b 1

:END
echo Setup completed successfully!
pause 