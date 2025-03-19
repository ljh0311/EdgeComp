@echo off
REM Baby Monitor System Startup Script for Windows
setlocal EnableDelayedExpansion

REM Default parameters
set MODE=normal
set CAMERA_ID=0
set DEBUG=
set PORT=5000
set INPUT_DEVICE=
set HOST=0.0.0.0

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--mode" (
    set MODE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--camera_id" (
    set CAMERA_ID=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--debug" (
    set DEBUG=--debug
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--input_device" (
    set INPUT_DEVICE=--input_device %~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args
:end_parse_args

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Set environment variables
set PYTHONPATH=.
set EVENTLET_NO_GREENDNS=yes

REM Display startup information
echo ╔════════════════════════════════════════════╗
echo ║      Baby Monitor System - Starting        ║
echo ╚════════════════════════════════════════════╝
echo.
echo Mode:          %MODE%
echo Camera ID:     %CAMERA_ID%
echo Host:          %HOST%
echo Port:          %PORT%
if defined DEBUG echo Debug:         Enabled
echo.
echo Starting application...
echo.

REM Run the application with the specified parameters
python main.py --mode %MODE% --camera_id %CAMERA_ID% --host %HOST% --port %PORT% %DEBUG% %INPUT_DEVICE%

REM If the application crashes, wait before closing
echo.
echo Application has exited. Press any key to close this window.
pause > nul 