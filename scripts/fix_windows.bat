@echo off
echo ===================================
echo Baby Monitor Windows Fix Utility
echo ===================================

echo.
echo Step 1: Checking Python installation...
where py >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python launcher (py^) not found.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

echo Detecting installed Python versions...
echo.

rem Create a temporary file to store Python versions
del pythonversions.txt 2>nul
del tmpver.txt 2>nul

rem Try Python versions from 3.8 to 3.12
set "VERSIONS_FOUND=0"
for %%v in (3.8 3.9 3.10 3.11 3.12) do (
    py -%%v --version >nul 2>&1
    if not errorlevel 1 (
        echo Found Python %%v
        py -%%v --version >> pythonversions.txt
        set /a VERSIONS_FOUND+=1
    )
)

if %VERSIONS_FOUND% EQU 0 (
    echo ERROR: No compatible Python version found.
    echo Please install Python 3.8 - 3.12 from https://www.python.org/downloads/
    pause
    exit /b 1
)

if %VERSIONS_FOUND% EQU 1 (
    for /f "tokens=2 delims= " %%a in (pythonversions.txt) do (
        set PYVER=%%a
    )
    echo Using the only available version: %PYVER%
) else (
    echo.
    echo Multiple Python versions found. Please choose which version to use:
    set "COUNT=0"
    for /f "tokens=2 delims= " %%a in (pythonversions.txt) do (
        set /a COUNT+=1
        echo !COUNT!^) Python %%a
    )
    echo.
    set /p CHOICE="Enter the number of your choice (1-%VERSIONS_FOUND%): "
    
    set "LINE_NUM=0"
    for /f "tokens=2 delims= " %%a in (pythonversions.txt) do (
        set /a LINE_NUM+=1
        if !LINE_NUM! EQU !CHOICE! (
            set PYVER=%%a
        )
    )
)

del pythonversions.txt

rem Extract major and minor version numbers
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set major=%%a
    set minor=%%b
)

echo.
echo Using Python %PYVER%
echo.

echo Step 2: Checking Visual C++ Redistributable...
if not exist "C:\Windows\System32\vcruntime140.dll" (
    echo WARNING: Visual C++ Redistributable might be missing
    echo Please download and install from: https://aka.ms/vs/16/release/vc_redist.x64.exe
    echo Press any key to continue anyway...
    pause >nul
)

echo.
echo Step 3: Stopping any running Baby Monitor processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Baby Monitor*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Step 4: Cleaning Python cache...
del /s /q *.pyc 2>nul
rd /s /q __pycache__ 2>nul
rd /s /q src\babymonitor\__pycache__ 2>nul
rd /s /q src\babymonitor\web\__pycache__ 2>nul
rd /s /q src\babymonitor\detectors\__pycache__ 2>nul

echo.
echo Step 5: Please ensure no other applications are using your camera or audio devices
echo Close any applications that might be using the camera or microphone (Zoom, Teams, etc.^)
echo Press any key when ready...
pause >nul

echo.
echo Step 6: Resetting virtual environment...
rd /s /q venv 2>nul
py -%major%.%minor% -m venv venv
call venv\Scripts\activate.bat

echo.
echo Step 7: Installing required packages...
echo Installing pip and wheel...
python -m pip install --upgrade pip wheel setuptools

echo Installing build tools and basic dependencies...
python -m pip install numpy==1.24.3
python -m pip install --no-cache-dir Cython

echo Installing CFFI and audio dependencies...
python -m pip install --no-cache-dir cffi==1.15.1
python -m pip install --no-cache-dir pycparser==2.21
python -m pip install --no-cache-dir sounddevice==0.4.6
python -m pip install --no-cache-dir soundfile==0.12.1
python -m pip install --no-cache-dir librosa==0.10.0

echo Installing WebRTC dependencies...
python -m pip install --no-cache-dir aiortc==1.5.0
python -m pip install --no-cache-dir aiohttp==3.8.5

echo Installing computer vision dependencies...
python -m pip install opencv-python-headless==4.8.1.78

echo Installing web server dependencies...
python -m pip install eventlet==0.33.3 flask-socketio==5.3.6 werkzeug==2.3.7
python -m pip install python-engineio==4.5.1 python-socketio==5.8.0
python -m pip install flask==2.3.3 python-dotenv==1.0.0

echo Installing machine learning dependencies...
if %minor% GEQ 11 (
    python -m pip install torch==2.0.0 torchaudio==2.0.0
) else (
    python -m pip install torch==1.13.1 torchaudio==0.13.1
)

echo Installing system utilities...
python -m pip install psutil==5.9.5

echo.
echo Step 8: Verifying installations...
python -c "import _cffi_backend; print('CFFI backend: OK')" || echo CFFI check failed
python -c "import aiortc; print('aiortc: OK')" || echo aiortc check failed
python -c "import sounddevice as sd; print('sounddevice: OK')" || echo sounddevice check failed
python -c "import cv2; print('OpenCV: OK')" || echo OpenCV check failed

echo.
echo Step 9: Setting up environment variables...
set PYTHONPATH=%CD%\..
set EVENTLET_NO_GREENDNS=yes

echo.
echo Step 10: Testing camera access...
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera test:', cap.isOpened()); cap.release()" || (
    echo WARNING: Could not access default camera.
    echo You may need to:
    echo 1. Check Windows camera privacy settings
    echo 2. Try a different camera ID when starting the application
)

echo.
echo Step 11: Creating start script...
(
echo @echo off
echo call venv\Scripts\activate.bat
echo set PYTHONPATH=%%CD%%
echo set EVENTLET_NO_GREENDNS=yes
echo python main.py %%*
echo pause
) > ..\start.bat

echo Setup completed! You can now:
echo 1. Run the application using: start.bat --mode normal
echo 2. Access the web interface at: http://localhost:5000
echo.
echo If you encounter issues:
echo - Try a different camera (use --camera_id 1^)
echo - Check Windows camera privacy settings
echo - Check Windows sound settings
echo - Ensure no other application is using the camera or microphone
echo - Check the logs in the console for detailed error messages
echo.
pause 