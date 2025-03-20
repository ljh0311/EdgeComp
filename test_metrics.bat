@echo off
echo Starting Metrics Test Server...
python src/babymonitor/scripts/test_metrics_socket.py --port 5000
pause 