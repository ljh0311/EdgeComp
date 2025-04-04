@echo off
setlocal enabledelayedexpansion

:menu
cls
echo ======================================
echo    Baby Monitor Backup and Restore
echo ======================================
echo.
echo Please select an option:
echo.
echo [1] List backup folders
echo [2] Create a new backup
echo [3] Restore from backup
echo [4] Clean up old backups
echo [5] Return to main menu
echo.
set /p choice=Enter your choice (1-5): 

if "%choice%"=="1" (
    echo.
    echo Available backups:
    echo -----------------
    python tools\backup\list_backups.py
    echo.
    pause
    goto menu
)
if "%choice%"=="2" (
    echo.
    echo Creating new backup...
    python tools\backup\create_backup.py
    echo.
    pause
    goto menu
)
if "%choice%"=="3" (
    echo.
    echo Available backups:
    echo -----------------
    python tools\backup\list_backups.py
    echo.
    set /p backup_id=Enter backup ID to restore (or press Enter to cancel): 
    if "!backup_id!"=="" goto menu
    
    echo.
    echo Warning: Restoring will overwrite current files.
    set /p confirm=Are you sure you want to restore from backup !backup_id!? (Y/N): 
    if /i "!confirm!"=="Y" (
        python tools\backup\restore_backup.py --backup !backup_id!
    ) else (
        echo Restore cancelled.
    )
    echo.
    pause
    goto menu
)
if "%choice%"=="4" (
    echo.
    set /p keep=How many recent backups would you like to keep? (default: 3): 
    if "!keep!"=="" set keep=3
    
    echo.
    echo Warning: This will permanently delete old backups.
    set /p confirm=Are you sure you want to clean up old backups, keeping the !keep! most recent? (Y/N): 
    if /i "!confirm!"=="Y" (
        python tools\backup\cleanup_backups.py --keep !keep!
    ) else (
        echo Cleanup cancelled.
    )
    echo.
    pause
    goto menu
)
if "%choice%"=="5" (
    exit /b 0
)

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu 