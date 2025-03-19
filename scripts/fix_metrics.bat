@echo off
REM Baby Monitor System - Metrics Fix Utility
REM This script helps troubleshoot and fix issues with the metrics dashboard

echo ============================================================
echo             Baby Monitor System - Metrics Fix Utility
echo ============================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if the metrics.js file exists
if not exist "src\babymonitor\web\static\js\metrics.js" (
    echo ERROR: Metrics JavaScript file not found.
    echo Attempting to create a default metrics.js file...
    
    if not exist "src\babymonitor\web\static\js" (
        mkdir "src\babymonitor\web\static\js"
    )
    
    REM Create a basic metrics.js file
    echo // Baby Monitor Metrics JavaScript > "src\babymonitor\web\static\js\metrics.js"
    echo console.log("Metrics script loaded"); >> "src\babymonitor\web\static\js\metrics.js"
    echo // Socket.IO connection for real-time updates >> "src\babymonitor\web\static\js\metrics.js"
    echo const socket = io(); >> "src\babymonitor\web\static\js\metrics.js"
    echo socket.on('connect', function() { >> "src\babymonitor\web\static\js\metrics.js"
    echo     console.log('Connected to server'); >> "src\babymonitor\web\static\js\metrics.js"
    echo }); >> "src\babymonitor\web\static\js\metrics.js"
    
    echo Metrics JavaScript file created.
)

REM Check if we want to enable demo mode
echo Would you like to enable demo mode with simulated data? (Y/N)
set /p ENABLE_DEMO=
if /i "%ENABLE_DEMO%"=="Y" (
    echo Creating metrics_demo.py script...
    
    REM Create a Python script to simulate metrics data
    echo # Baby Monitor Metrics Demo > scripts\metrics_demo.py
    echo import time >> scripts\metrics_demo.py
    echo import random >> scripts\metrics_demo.py
    echo import json >> scripts\metrics_demo.py
    echo import argparse >> scripts\metrics_demo.py
    echo import os >> scripts\metrics_demo.py
    echo import sys >> scripts\metrics_demo.py
    echo import threading >> scripts\metrics_demo.py
    echo. >> scripts\metrics_demo.py
    echo # Add the project root to the path >> scripts\metrics_demo.py
    echo sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) >> scripts\metrics_demo.py
    echo. >> scripts\metrics_demo.py
    echo try: >> scripts\metrics_demo.py
    echo     from src.babymonitor.web.server import socketio, app >> scripts\metrics_demo.py
    echo except ImportError: >> scripts\metrics_demo.py
    echo     print("Error importing socketio from server. Make sure the server module is properly installed.") >> scripts\metrics_demo.py
    echo     sys.exit(1) >> scripts\metrics_demo.py
    echo. >> scripts\metrics_demo.py
    echo def generate_random_data(): >> scripts\metrics_demo.py
    echo     """Generate random metrics data for demonstration""" >> scripts\metrics_demo.py
    echo     return { >> scripts\metrics_demo.py
    echo         'fps': [random.uniform(10.0, 30.0) for _ in range(20)], >> scripts\metrics_demo.py
    echo         'detectionCount': [random.randint(0, 3) for _ in range(20)], >> scripts\metrics_demo.py
    echo         'detectionConfidence': [random.uniform(70.0, 98.0) for _ in range(20)], >> scripts\metrics_demo.py
    echo         'cpuUsage': [random.uniform(10.0, 50.0) for _ in range(20)], >> scripts\metrics_demo.py
    echo         'memoryUsage': [random.uniform(20.0, 60.0) for _ in range(20)], >> scripts\metrics_demo.py
    echo         'emotions': { >> scripts\metrics_demo.py
    echo             'crying': random.uniform(0.0, 25.0), >> scripts\metrics_demo.py
    echo             'laughing': random.uniform(0.0, 25.0), >> scripts\metrics_demo.py
    echo             'babbling': random.uniform(0.0, 25.0), >> scripts\metrics_demo.py
    echo             'silence': 0  # Will be calculated to ensure sum is 100%% >> scripts\metrics_demo.py
    echo         } >> scripts\metrics_demo.py
    echo     } >> scripts\metrics_demo.py
    echo. >> scripts\metrics_demo.py
    echo def emit_demo_data(): >> scripts\metrics_demo.py
    echo     """Emit random metrics data to connected clients""" >> scripts\metrics_demo.py
    echo     while True: >> scripts\metrics_demo.py
    echo         data = generate_random_data() >> scripts\metrics_demo.py
    echo         # Calculate silence to ensure sum is 100%% >> scripts\metrics_demo.py
    echo         total = data['emotions']['crying'] + data['emotions']['laughing'] + data['emotions']['babbling'] >> scripts\metrics_demo.py
    echo         data['emotions']['silence'] = max(0, 100 - total) >> scripts\metrics_demo.py
    echo. >> scripts\metrics_demo.py
    echo         # Emit the metrics data >> scripts\metrics_demo.py
    echo         print(f"Emitting metrics data: FPS {data['fps'][-1]:.1f}, Detections {data['detectionCount'][-1]}") >> scripts\metrics_demo.py
    echo         socketio.emit('metrics_update', data) >> scripts\metrics_demo.py
    echo. >> scripts\metrics_demo.py
    echo         # Randomly emit crying events (10%% chance) >> scripts\metrics_demo.py
    echo         if random.random() < 0.1: >> scripts\metrics_demo.py
    echo             timestamp = time.strftime("%%Y-%%m-%%d %%H:%%M:%%S") >> scripts\metrics_demo.py
    echo             print(f"Emitting crying event at {timestamp}") >> scripts\metrics_demo.py
    echo             socketio.emit('crying_detected', {'timestamp': timestamp}) >> scripts\metrics_demo.py
    echo. >> scripts\metrics_demo.py
    echo         time.sleep(2)  # Update every 2 seconds >> scripts\metrics_demo.py
    echo. >> scripts\metrics_demo.py
    echo def run_demo_server(host='0.0.0.0', port=5000): >> scripts\metrics_demo.py
    echo     """Run the Flask server with demo data""" >> scripts\metrics_demo.py
    echo     # Start the data emission in a separate thread >> scripts\metrics_demo.py
    echo     threading.Thread(target=emit_demo_data, daemon=True).start() >> scripts\metrics_demo.py
    echo     print(f"Starting demo server on {host}:{port}") >> scripts\metrics_demo.py
    echo     socketio.run(app, host=host, port=port, debug=True, use_reloader=False) >> scripts\metrics_demo.py
    echo. >> scripts\metrics_demo.py
    echo if __name__ == '__main__': >> scripts\metrics_demo.py
    echo     parser = argparse.ArgumentParser(description='Run the Baby Monitor metrics demo server') >> scripts\metrics_demo.py
    echo     parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on') >> scripts\metrics_demo.py
    echo     parser.add_argument('--port', type=int, default=5000, help='Port to run the server on') >> scripts\metrics_demo.py
    echo     args = parser.parse_args() >> scripts\metrics_demo.py
    echo. >> scripts\metrics_demo.py
    echo     run_demo_server(host=args.host, port=args.port) >> scripts\metrics_demo.py

echo.
    echo Demo script created. Run it with:
    echo python scripts\metrics_demo.py
)

echo.
echo ============================================================
echo Checking for issues with metrics display...
echo.

echo Step 1: Verifying metrics.html template...
if not exist "src\babymonitor\web\templates\metrics.html" (
    echo ERROR: metrics.html template not found.
    echo The metrics dashboard template is missing. Please reinstall the application.
) else (
    echo metrics.html template found.
)

echo.
echo Step 2: Verifying JavaScript reference in metrics.html...
findstr /C:"metrics.js" "src\babymonitor\web\templates\metrics.html" > nul
if %errorlevel% neq 0 (
    echo WARNING: metrics.js reference not found in metrics.html.
    echo Make sure the metrics.html file contains a reference to metrics.js.
) else (
    echo JavaScript reference found in metrics.html.
)

echo.
echo Step 3: Checking for Socket.IO script...
findstr /C:"socket.io" "src\babymonitor\web\templates\base.html" > nul
if %errorlevel% neq 0 (
    echo WARNING: Socket.IO reference not found in base.html.
    echo Make sure Socket.IO is properly included in your templates.
) else (
    echo Socket.IO reference found.
)

echo.
echo ============================================================
echo Metrics fix utility completed.
echo.
echo Next steps:
echo 1. Start the Baby Monitor system with: .\start.bat
echo 2. For demo data, run: python scripts\metrics_demo.py
echo 3. Access the metrics dashboard at: http://localhost:5000/metrics
echo ============================================================

pause 