@echo off
call venv\Scripts\activate.bat
set PYTHONPATH=.
set EVENTLET_NO_GREENDNS=yes
python run_monitor.py --mode dev --camera_id 0 --debug
pause
