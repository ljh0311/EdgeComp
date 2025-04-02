@echo off
setlocal enabledelayedexpansion

echo ======================================
echo   Baby Monitor System Cleanup Script
echo ======================================
echo.
echo This script will organize the project files according to the
echo new directory structure and remove obsolete files.
echo.
echo The following actions will be performed:
echo.
echo  1. Create necessary directories if they don't exist
echo  2. Move files to their appropriate locations
echo  3. Delete obsolete files
echo  4. Update documentation
echo.
echo WARNING: This process will modify your file structure.
echo It's recommended to create a backup before proceeding.
echo.
set /p confirm=Do you want to proceed? (Y/N): 

if /i not "%confirm%"=="Y" (
    echo.
    echo Cleanup cancelled.
    goto :eof
)

echo.
echo Creating backup before proceeding...
echo.

REM Create a backup
python tools\backup\create_backup.py -n "pre_cleanup"

echo.
echo Starting cleanup and organization process...
echo.

REM Create necessary directories
echo Creating directories...
if not exist src\babymonitor\models mkdir src\babymonitor\models
if not exist tools\backup mkdir tools\backup
if not exist tools\system mkdir tools\system
if not exist tests\updated mkdir tests\updated

REM First, move files to their appropriate locations
echo Moving files to appropriate locations...

REM Move models to models directory
if exist pretrained_models\*.* (
    echo - Moving pretrained models...
    move pretrained_models\*.* src\babymonitor\models\ > nul
)

REM Move or update API test
if exist api_test.py (
    echo - Moving and updating API test...
    move api_test.py tests\updated\ > nul
)

REM Remove obsolete files
echo Removing obsolete files...

if exist cleanup_repair_scripts.sh (
    echo - Removing cleanup_repair_scripts.sh
    del cleanup_repair_scripts.sh
)

if exist cleanup_repair_scripts.bat (
    echo - Removing cleanup_repair_scripts.bat
    del cleanup_repair_scripts.bat
)

if exist README_repair.md (
    echo - Removing README_repair.md
    del README_repair.md
)

if exist repair_and_cleanup.bat (
    echo - Removing repair_and_cleanup.bat
    del repair_and_cleanup.bat
)

if exist repair_and_cleanup.ps1 (
    echo - Removing repair_and_cleanup.ps1
    del repair_and_cleanup.ps1
)

REM Consolidate startup scripts
echo Consolidating startup scripts...

if exist start_monitor.bat (
    echo - Removing start_monitor.bat (functionality in run_babymonitor.bat)
    del start_monitor.bat
)

if exist start.bat (
    echo - Removing start.bat (functionality in run_babymonitor.bat)
    del start.bat
)

if exist start_pi.sh (
    echo - Removing start_pi.sh (functionality in run_babymonitor.sh)
    del start_pi.sh
)

if exist start.sh (
    echo - Removing start.sh (functionality in run_babymonitor.sh)
    del start.sh
)

REM Move README_scripts.md contents to README.md (this was manual step already completed)
echo Documentation has been updated (README_scripts.md content merged into README.md)
echo You can now delete README_scripts.md if desired.

REM Remove empty directories
echo Cleaning up empty directories...
if exist pretrained_models (
    rmdir pretrained_models 2>nul
)

echo.
echo ======================================
echo         Cleanup Complete!
echo ======================================
echo.
echo The baby monitor system files have been organized.
echo New directory structure:
echo.
echo  - src/babymonitor/models/    : Pretrained models
echo  - tools/backup/              : Backup utilities
echo  - tools/system/              : System utilities
echo  - tests/updated/             : Updated test scripts
echo.
echo You can now use:
echo  - run_babymonitor.bat        : Windows launcher
echo  - run_babymonitor.sh         : Linux/macOS launcher
echo  - tools\backup_restore.bat   : Windows backup utility
echo  - tools/backup/restore.sh    : Linux/macOS backup utility
echo.
echo Please check that the system still works as expected.
echo.
pause 