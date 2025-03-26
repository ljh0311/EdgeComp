@echo off
echo ============================================================
echo Starting Metrics Test Server...
echo ============================================================
echo This server simulates metrics data for testing the metrics page.
echo It provides the following functionality:
echo  - Real-time metrics data transmission via Socket.IO
echo  - API endpoints for system information
echo  - Support for camera resolution changes from repair_tools.html
echo.
echo Access the metrics page at: http://localhost:5000/metrics
echo Access the repair tools page at: http://localhost:5000/repair
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

python src/babymonitor/scripts/test_metrics_socket.py --port 5000

pause 