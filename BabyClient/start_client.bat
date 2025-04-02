@echo off
REM Start Baby Monitor Client with optional MQTT support
if "%~1"=="" (
    python baby_client.py --host 192.168.1.100 --port 5000 --mqtt-host 192.168.1.100 --mqtt-port 1883
) else (
    python baby_client.py %*
)
pause 