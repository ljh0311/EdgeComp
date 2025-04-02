@echo off
REM Create a backup of files before cleanup
echo Creating backup of files before cleanup...

REM Create backup directory with timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set BACKUP_DIR=backup_%datetime:~0,8%_%datetime:~8,6%

mkdir %BACKUP_DIR%
echo Backup directory created: %BACKUP_DIR%

REM Copy all files to be cleaned up to the backup directory
echo Copying files to backup directory...

REM Scripts
copy cleanup_repair_scripts.sh %BACKUP_DIR%\ 2>nul
copy cleanup_repair_scripts.bat %BACKUP_DIR%\ 2>nul
copy repair_and_cleanup.bat %BACKUP_DIR%\ 2>nul
copy repair_and_cleanup.ps1 %BACKUP_DIR%\ 2>nul
copy start_monitor.bat %BACKUP_DIR%\ 2>nul
copy start.bat %BACKUP_DIR%\ 2>nul
copy api_test.py %BACKUP_DIR%\ 2>nul

REM Documentation
copy README_repair.md %BACKUP_DIR%\ 2>nul
copy README_scripts.md %BACKUP_DIR%\ 2>nul

REM Create directories for larger content
mkdir %BACKUP_DIR%\tests 2>nul
mkdir %BACKUP_DIR%\scripts 2>nul
mkdir %BACKUP_DIR%\static 2>nul
mkdir %BACKUP_DIR%\tools 2>nul
mkdir %BACKUP_DIR%\pretrained_models 2>nul

REM Copy directories (will only work if not too large)
xcopy /E /I /Y tests %BACKUP_DIR%\tests 2>nul
xcopy /E /I /Y scripts %BACKUP_DIR%\scripts 2>nul
xcopy /E /I /Y static %BACKUP_DIR%\static 2>nul
xcopy /E /I /Y tools %BACKUP_DIR%\tools 2>nul
xcopy /E /I /Y pretrained_models %BACKUP_DIR%\pretrained_models 2>nul

REM Copy shell scripts (if on WSL or Git Bash)
copy start_pi.sh %BACKUP_DIR%\ 2>nul
copy start.sh %BACKUP_DIR%\ 2>nul

echo.
echo Backup completed successfully to directory: %BACKUP_DIR%
echo Please verify the backup before proceeding with cleanup.
echo.

pause 