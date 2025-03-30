@echo off
echo Baby Monitor Repair and Cleanup Utility
echo.

REM Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found! Please install Python and add it to PATH.
    goto :eof
)

REM Install required packages
python -m pip install flask pyaudio psutil >nul 2>nul

echo Choose an option:
echo 1. Run API Repair Server
echo 2. Clean up backups
echo 3. List backups 
echo 4. Restore backup
echo 5. Exit
echo.

set /p choice=Choice (1-5): 

if "%choice%"=="1" (
    echo Starting API Repair Server on port 5001...
    python repair_api_endpoints.py --debug
    goto :eof
)

if "%choice%"=="2" (
    set /p keep=Backups to keep (default 3): 
    if "%keep%"=="" set keep=3
    python tools\cleanup_backups.py --keep %keep% --dir .
    pause
    goto :eof
)

if "%choice%"=="3" (
    python tools\restore_backup.py --list --app-root .
    pause
    goto :eof
)

if "%choice%"=="4" (
    python tools\restore_backup.py --list --app-root .
    set /p backup=Backup folder to restore: 
    set /p confirm=Restore from %backup%? (y/n): 
    if /i not "%confirm%"=="y" goto :eof
    python tools\restore_backup.py --backup %backup% --app-root .
    pause
    goto :eof
)

if "%choice%"=="5" (
    echo Goodbye!
    goto :eof
)

echo Invalid choice!