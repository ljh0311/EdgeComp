@echo off
setlocal enabledelayedexpansion

echo ======================================
echo   Baby Monitor System Installation
echo ======================================
echo   Version 2.1.0 - State Detection
echo ======================================
echo.

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher and try again.
    pause
    exit /b 1
)

REM Determine Python version
python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" > temp_version.txt
set /p py_version=<temp_version.txt
del temp_version.txt

REM Verify Python version is at least 3.8
for /f "tokens=1,2 delims=." %%a in ("%py_version%") do (
    set major=%%a
    set minor=%%b
)

if %major% LSS 3 (
    echo Error: Python version %py_version% is not supported.
    echo Please install Python 3.8 or higher.
    pause
    exit /b 1
) else (
    if %major% EQU 3 (
        if %minor% LSS 8 (
            echo Error: Python version %py_version% is not supported.
            echo Please install Python 3.8 or higher.
            pause
            exit /b 1
        )
    )
)

echo Detected Python %py_version%

REM Parse command line arguments
set INSTALL=false
set MODELS=false
set FIX=false
set CONFIG=false
set TRAIN=false
set DOWNLOAD=false
set NO_GUI=false
set STATE_DETECTION=true

:parse_args
if "%~1"=="" goto done_parsing
if /i "%~1"=="--install" set INSTALL=true
if /i "%~1"=="--models" set MODELS=true
if /i "%~1"=="--fix" set FIX=true
if /i "%~1"=="--config" set CONFIG=true
if /i "%~1"=="--train" set TRAIN=true
if /i "%~1"=="--download" set DOWNLOAD=true
if /i "%~1"=="--no-gui" set NO_GUI=true
if /i "%~1"=="--no-state-detection" set STATE_DETECTION=false
shift
goto parse_args

:done_parsing

REM Set up virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    
    echo Activating virtual environment...
    call venv\Scripts\activate
    
    echo Installing required packages...
    pip install -r requirements.txt
    
    if %ERRORLEVEL% NEQ 0 (
        echo Error installing requirements. 
        echo Please check your internet connection and try again.
        pause
        exit /b 1
    )
) else (
    echo Activating existing virtual environment...
    call venv\Scripts\activate
)

REM Create models directory if it doesn't exist
if not exist src\babymonitor\models (
    echo Creating models directory...
    mkdir src\babymonitor\models
)

REM Check for models
echo.
echo Checking model status...

REM Count total models and missing models
set TOTAL_MODELS=0
set MISSING_MODELS=0

if not exist src\babymonitor\models\emotion_model.pt (
    echo - Missing: Emotion recognition model
    set /a MISSING_MODELS+=1
)
set /a TOTAL_MODELS+=1

if not exist src\babymonitor\models\yolov8n.pt (
    echo - Missing: Person detection model
    set /a MISSING_MODELS+=1
)
set /a TOTAL_MODELS+=1

if not exist src\babymonitor\models\wav2vec2_emotion.pt (
    echo - Missing: Wav2Vec2 emotion model
    set /a MISSING_MODELS+=1
)
set /a TOTAL_MODELS+=1

REM Check for state detection model if enabled
if %STATE_DETECTION%==true (
    if not exist src\babymonitor\models\person_state_classifier.pkl (
        echo - Missing: Person state detection model
        set /a MISSING_MODELS+=1
    )
    set /a TOTAL_MODELS+=1
)

echo.
echo Found %MISSING_MODELS% missing models out of %TOTAL_MODELS% required models.

REM Handle models based on command line args or user input
if %DOWNLOAD%==true (
    echo Downloading pretrained models...
    python setup.py --download --no-gui
) else if %TRAIN%==true (
    echo Training missing models...
    python setup.py --train --no-gui
) else if %MISSING_MODELS% GTR 0 (
    echo.
    echo Some required models are missing. What would you like to do?
    echo 1. Download pretrained models (recommended)
    echo 2. Train models from scratch (takes time and requires data)
    echo 3. Continue without models (not recommended)
    echo.
    set /p MODEL_CHOICE=Enter your choice (1-3): 
    
    if "%MODEL_CHOICE%"=="1" (
        echo Downloading pretrained models...
        python setup.py --download --no-gui
    ) else if "%MODEL_CHOICE%"=="2" (
        echo Training models from scratch...
        python setup.py --train --no-gui
    ) else (
        echo Continuing without all models - some functionality may be limited.
    )
)

REM Handle specific installation modes
if %INSTALL%==true (
    echo Installing Baby Monitor System...
    if %STATE_DETECTION%==false (
        python setup.py --install --no-gui --no-state-detection
    ) else (
        python setup.py --install --no-gui
    )
) else if %MODELS%==true (
    echo Managing models...
    python setup.py --models --no-gui
) else if %FIX%==true (
    echo Fixing common issues...
    python setup.py --fix --no-gui
) else if %CONFIG%==true (
    echo Configuring system...
    python setup.py --config --no-gui
) else if %NO_GUI%==true (
    echo.
    echo Choose what you want to do:
    echo 1. Install/reinstall the system
    echo 2. Manage models
    echo 3. Fix common issues
    echo 4. Configure system
    echo 5. Toggle state detection (currently: %STATE_DETECTION%)
    echo.
    set /p MENU_CHOICE=Enter your choice (1-5): 
    
    if "%MENU_CHOICE%"=="1" (
        if %STATE_DETECTION%==false (
            python setup.py --install --no-gui --no-state-detection
        ) else (
            python setup.py --install --no-gui
        )
    ) else if "%MENU_CHOICE%"=="2" (
        python setup.py --models --no-gui
    ) else if "%MENU_CHOICE%"=="3" (
        python setup.py --fix --no-gui
    ) else if "%MENU_CHOICE%"=="4" (
        python setup.py --config --no-gui
    ) else if "%MENU_CHOICE%"=="5" (
        REM Toggle state detection
        if %STATE_DETECTION%==true (
            set STATE_DETECTION=false
            echo State detection disabled.
        ) else (
            set STATE_DETECTION=true
            echo State detection enabled.
        )
        
        REM Update .env file
        if exist .env (
            findstr /C:"STATE_DETECTION_ENABLED=" .env > nul
            if %ERRORLEVEL% EQU 0 (
                REM Create temporary file with updated setting
                set TEMPFILE=temp.env
                del %TEMPFILE% 2> nul
                
                for /f "delims=" %%a in (.env) do (
                    set "line=%%a"
                    if "!line:~0,22!"=="STATE_DETECTION_ENABLED" (
                        if %STATE_DETECTION%==true (
                            echo STATE_DETECTION_ENABLED=true >> %TEMPFILE%
                        ) else (
                            echo STATE_DETECTION_ENABLED=false >> %TEMPFILE%
                        )
                    ) else (
                        echo !line! >> %TEMPFILE%
                    )
                )
                
                REM Replace original file with temporary file
                del .env
                rename %TEMPFILE% .env
            ) else (
                if %STATE_DETECTION%==true (
                    echo STATE_DETECTION_ENABLED=true >> .env
                ) else (
                    echo STATE_DETECTION_ENABLED=false >> .env
                )
            )
            echo Updated .env configuration.
        )
    ) else (
        echo Invalid choice.
    )
) else (
    REM No specific mode - launch GUI if available
    python setup.py
)

REM Check for successful installation
if exist src\babymonitor (
    if exist run_babymonitor.bat (
        echo.
        echo Baby Monitor has been successfully installed!
        echo.
        echo To launch the application, run:
        echo   run_babymonitor.bat
        echo.
    )
)

REM Display team information
echo.
echo ======================================
echo   Baby Monitor System Developed By:
echo ======================================
echo • JunHong: Backend Processing & Client Logic
echo • Darrel: Dashboard Frontend
echo • Ashraf: Datasets & Model Architecture
echo • Xuan Yu: Specialized Datasets & Training
echo • Javin: Camera Detection System
echo ======================================

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo.
echo Installation process completed.
echo.
pause 