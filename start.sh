#!/bin/bash
# Baby Monitor System Startup Script for macOS/Linux

# Default parameters
MODE="normal"
CAMERA_ID=0
DEBUG=""
PORT=5000
INPUT_DEVICE=""
HOST="0.0.0.0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --camera_id)
      CAMERA_ID="$2"
      shift 2
      ;;
    --debug)
      DEBUG="--debug"
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --input_device)
      INPUT_DEVICE="--input_device $2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Activate the virtual environment
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Set environment variables
export PYTHONPATH=.
export EVENTLET_NO_GREENDNS=yes

# Display startup information
echo "╔════════════════════════════════════════════╗"
echo "║      Baby Monitor System - Starting        ║"
echo "╚════════════════════════════════════════════╝"
echo
echo "Mode:          $MODE"
echo "Camera ID:     $CAMERA_ID"
echo "Host:          $HOST"
echo "Port:          $PORT"
if [ -n "$DEBUG" ]; then echo "Debug:         Enabled"; fi
echo
echo "Starting application..."
echo

# Run the application with the specified parameters
python3 main.py --mode "$MODE" --camera_id "$CAMERA_ID" --host "$HOST" --port "$PORT" $DEBUG $INPUT_DEVICE

# If the application crashes, give feedback
echo
echo "Application has exited. Press Enter to close this window."
read 