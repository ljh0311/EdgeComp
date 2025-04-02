@echo off
setlocal enabledelayedexpansion

echo =========================================
echo Baby Monitor System - Model Manager v1.0
echo =========================================
echo.

:: Check if we're in the correct directory
if not exist "setup.py" (
    echo ERROR: This script must be run from the project root directory.
    echo Current directory: %CD%
    echo.
    echo Please navigate to the project root directory and try again.
    pause
    exit /b 1
)

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found.
    echo Please install Python 3.8 or higher and ensure it's in your PATH.
    pause
    exit /b 1
)

:: Display model status
echo Checking model status...
python -c "import os; print('\nRequired models directory: ' + os.path.abspath('src/babymonitor/models'))"
python setup.py --models --no-gui

echo.
echo What would you like to do?
echo.
echo [1] Download missing models (recommended)
echo [2] Download all models (overwrite existing)
echo [3] Train emotion model
echo [4] Train speech recognition model
echo [5] Train all models
echo [6] Exit
echo.

set /p choice=Enter your choice (1-6): 

if "%choice%"=="1" (
    echo.
    echo Downloading missing models...
    python setup.py --download --no-gui
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Downloading all models (this will overwrite existing models)...
    python setup.py --download --train-all --no-gui
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Training emotion model...
    echo NOTE: This may take several hours depending on your computer's performance.
    echo.
    set /p confirm=Are you sure you want to proceed? (y/n): 
    if /i "!confirm!"=="y" (
        python setup.py --train --specific-model emotion_model --no-gui
    ) else (
        echo Training cancelled.
    )
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Training speech recognition model...
    echo NOTE: This requires significant computational resources and may take several hours.
    echo.
    set /p confirm=Are you sure you want to proceed? (y/n): 
    if /i "!confirm!"=="y" (
        python setup.py --train --specific-model wav2vec2_model --no-gui
    ) else (
        echo Training cancelled.
    )
    goto end
)

if "%choice%"=="5" (
    echo.
    echo Training all models...
    echo NOTE: This will take a very long time and requires significant resources.
    echo Consider downloading pretrained models instead if possible.
    echo.
    set /p confirm=Are you sure you want to proceed? (y/n): 
    if /i "!confirm!"=="y" (
        python setup.py --train --train-all --no-gui
    ) else (
        echo Training cancelled.
    )
    goto end
)

if "%choice%"=="6" (
    echo Exiting...
    goto end
)

echo Invalid choice. Please try again.

:end
echo.
echo Model management complete. Displaying final model status...
python setup.py --models --no-gui

echo.
echo Thank you for using the Baby Monitor System Model Manager.
pause 