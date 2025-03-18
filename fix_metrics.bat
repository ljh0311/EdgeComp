@echo off
echo ===================================
echo Baby Monitor Metrics Fix Utility
echo ===================================
echo.

REM Kill any existing Python processes running the baby monitor
echo Stopping any running Baby Monitor processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Baby Monitor*" 2>nul
timeout /t 2 /nobreak >nul

REM Clear browser cache command instructions
echo.
echo ===================================
echo BROWSER CACHE CLEARING INSTRUCTIONS
echo ===================================
echo.
echo Before restarting the application, please:
echo 1. Press Ctrl+Shift+Delete in your browser
echo 2. Select "Cached images and files"
echo 3. Click "Clear data"
echo.
echo This will ensure the metrics JavaScript is reloaded properly.
echo.
pause

REM Clear screen
cls
echo ===================================
echo Baby Monitor Metrics Fix Utility
echo ===================================
echo.
echo Starting Baby Monitor in development mode...
echo.

REM Start the baby monitor in a new window
start "Baby Monitor - YOLOv8 Edition" cmd /k python main.py --mode dev

echo.
echo Baby Monitor started successfully!
echo.
echo The web interface should be available at:
echo http://localhost:5000/metrics
echo.
echo If you still see issues with the metrics display, try:
echo 1. Opening in a new browser window
echo 2. Using Chrome developer tools to check for JavaScript errors (F12)
echo 3. Restarting your browser completely
echo.
echo Press any key to exit...
pause >nul 