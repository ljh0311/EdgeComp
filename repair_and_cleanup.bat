@echo off
echo ======================================================
echo Baby Monitor Repair and Cleanup Utility
echo ======================================================
echo.

REM Check if python is installed and in PATH
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found! Please install Python and add it to your PATH.
    goto :eof
)

echo Checking for required Python packages...
python -m pip install flask pyaudio psutil >nul 2>nul

echo.
echo Select an operation:
echo 1 - Run API Repair Server
echo 2 - Clean up backup folders
echo 3 - List backup folders
echo 4 - Restore from backup
echo 5 - Exit
echo.

set /p choice=Enter your choice (1-5): 

if "%choice%"=="1" goto run_server
if "%choice%"=="2" goto cleanup_backups
if "%choice%"=="3" goto list_backups
if "%choice%"=="4" goto restore_backup
if "%choice%"=="5" goto end

echo Invalid choice. Please try again.
goto :eof

:run_server
echo.
echo Starting API Repair Server...
echo.
echo The server will run on port 5001.
echo Press Ctrl+C to stop the server.
echo.
python repair_api_endpoints.py --debug
goto :eof

:cleanup_backups
echo.
set /p keep=Number of backups to keep (default: 3): 

if "%keep%"=="" set keep=3

echo.
echo Cleaning up backup folders, keeping the %keep% most recent backups...
echo.
python tools\cleanup_backups.py --keep %keep% --dir .
echo.
pause
goto :eof

:list_backups
echo.
echo Listing available backup folders...
echo.
python tools\restore_backup.py --list --app-root .
echo.
pause
goto :eof

:restore_backup
echo.
python tools\restore_backup.py --list --app-root .
echo.
set /p backup=Enter the name of the backup folder to restore from: 
echo.
set /p confirm=Are you sure you want to restore from %backup%? (y/n): 

if /i not "%confirm%"=="y" goto :eof

echo.
echo Restoring from backup %backup%...
python tools\restore_backup.py --backup %backup% --app-root .
echo.
pause
goto :eof

:end
echo Goodbye! 