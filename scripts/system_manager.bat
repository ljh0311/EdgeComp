@echo off
setlocal enabledelayedexpansion

:: Check for Python installation
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    exit /b 1
)

:: Get the directory of this script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Check if running with admin privileges
net session >nul 2>&1
set "ADMIN=0"
if %errorlevel% == 0 set "ADMIN=1"

:: Parse arguments
set "ACTION=%1"
if "%ACTION%"=="" set "ACTION=gui"

:: Run the system manager
if "%ADMIN%"=="1" (
    python system_manager.py %ACTION% %2 %3 %4 %5
) else (
    powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/c cd /d %SCRIPT_DIR% && python system_manager.py %ACTION% %2 %3 %4 %5'"
)

exit /b %errorlevel% 