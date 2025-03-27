@echo off
echo ===================================
echo Baby Monitor Emotion Models Fix Utility
echo ===================================
echo.

echo Step 1: Checking Python installation...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found.
    echo Please install Python and make sure it's in your PATH.
    pause
    exit /b 1
)

echo.
echo Step 2: Updating emotion models...
echo.

echo Running the setup_models.py script to ensure all emotion models are properly copied...
python -m src.babymonitor.utils.setup_models

echo.
echo Step 3: Verifying emotion model installation...
echo.

python -m src.babymonitor.utils.verify_emotion_models

echo.
echo === Completed Emotion Models Fix ===
echo.
echo If there are still missing models, try running:
echo   python setup.py --download-models
echo.

pause 