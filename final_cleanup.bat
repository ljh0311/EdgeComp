@echo off
echo ======================================
echo    Baby Monitor Final Cleanup
echo ======================================
echo.
echo This script will remove unnecessary files from your
echo root directory to make it clean and organized.
echo.
echo Files to be removed:
echo - cleanup_and_organize.bat (no longer needed)
echo - backup_before_cleanup.bat (functionality moved to tools/backup)
echo - README_update.md (already merged into README.md)
echo - Any remaining old scripts that have been replaced
echo.
echo WARNING: Make sure you've already run cleanup_and_organize.bat
echo and that you have a backup before proceeding.
echo.
set /p confirm=Are you sure you want to proceed? (Y/N): 

if /i not "%confirm%"=="Y" (
    echo Cleanup cancelled.
    goto :eof
)

echo.
echo Removing unnecessary files...

REM Create list of files to keep in root directory
set "KEEP_FILES=run_babymonitor.bat run_babymonitor.sh README.md requirements.txt LICENSE"

REM Create list of directories to keep
set "KEEP_DIRS=src tools tests venv"

REM Remove specific files that are no longer needed
if exist cleanup_and_organize.bat (
    echo - Removing cleanup_and_organize.bat
    del cleanup_and_organize.bat
)

if exist backup_before_cleanup.bat (
    echo - Removing backup_before_cleanup.bat
    del backup_before_cleanup.bat
)

if exist README_update.md (
    echo - Removing README_update.md
    del README_update.md
)

if exist README_scripts.md (
    echo - Removing README_scripts.md
    del README_scripts.md
)

REM Look for any remaining .bat files that might be obsolete
for %%F in (*.bat) do (
    set "KEEP=false"
    for %%K in (%KEEP_FILES%) do (
        if "%%F"=="%%K" set "KEEP=true"
    )
    if not "!KEEP!"=="true" (
        if not "%%F"=="final_cleanup.bat" (
            echo - Removing %%F
            del %%F
        )
    )
)

REM Look for any remaining .sh files that might be obsolete
for %%F in (*.sh) do (
    set "KEEP=false"
    for %%K in (%KEEP_FILES%) do (
        if "%%F"=="%%K" set "KEEP=true"
    )
    if not "!KEEP!"=="true" (
        echo - Removing %%F
        del %%F
    )
)

REM Look for any remaining .py files in root (should be moved to proper directories)
for %%F in (*.py) do (
    echo - Removing %%F (Python scripts should be in appropriate directories)
    del %%F
)

REM Remove any empty directories in root that aren't in the keep list
for /d %%D in (*) do (
    set "KEEP=false"
    for %%K in (%KEEP_DIRS%) do (
        if "%%D"=="%%K" set "KEEP=true"
    )
    if not "!KEEP!"=="true" (
        echo - Checking if %%D is empty
        dir /b "%%D" | findstr "^" > nul
        if errorlevel 1 (
            echo - Removing empty directory: %%D
            rmdir "%%D"
        ) else (
            echo - Directory not empty, skipping: %%D
        )
    )
)

echo.
echo ======================================
echo        Final Cleanup Complete!
echo ======================================
echo.
echo Your directory structure is now clean and organized.
echo.
echo Essential files in root directory:
echo - run_babymonitor.bat (Windows launcher)
echo - run_babymonitor.sh (Linux/macOS launcher)
echo - README.md (Documentation)
echo - requirements.txt (Dependencies)
echo.
echo Essential directories:
echo - src/ (Source code)
echo - tools/ (Utility scripts)
echo - tests/ (Test scripts)
echo.
echo You can safely delete this cleanup script (final_cleanup.bat)
echo when you're satisfied with the results.
echo.
pause 