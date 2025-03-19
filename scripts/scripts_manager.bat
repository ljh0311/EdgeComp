@echo off
REM Baby Monitor System - Unified Installer and Repair Tool
echo Starting Baby Monitor Unified Installer and Repair Tool...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the unified manager GUI
python scripts\scripts_manager_gui.py

REM If the GUI crashes, wait before closing
if %errorlevel% neq 0 (
  echo.
  echo The Unified Manager has crashed. Press any key to close this window.
  pause > nul
) 