#!/bin/bash
# Baby Monitor System Startup Script for Raspberry Pi

# Default parameters
MODE="normal"
CAMERA_ID=0
DEBUG=""
PORT=5000
INPUT_DEVICE=""
HOST="0.0.0.0"
OPTIMIZE=true

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
    --no-optimize)
      OPTIMIZE=false
      shift
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

# Raspberry Pi optimizations
if [ "$OPTIMIZE" = true ]; then
  # Disable CPU throttling
  echo "Applying Raspberry Pi optimizations..."
  
  # Check if running as root
  if [ "$(id -u)" -eq 0 ]; then
    # Set CPU governor to performance
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      echo "performance" > "$cpu" 2>/dev/null || true
    done
    
    # Increase GPU memory if not already high
    vcgencmd get_mem gpu | grep -q "gpu=128M" && vcgencmd set_mem gpu 256
    
    # Optimize swap
    swapoff -a
    swapon -a
  else
    echo "Note: Some optimizations skipped (not running as root)"
  fi
fi

# Display startup information
echo "╔════════════════════════════════════════════╗"
echo "║   Baby Monitor System - Raspberry Pi       ║"
echo "╚════════════════════════════════════════════╝"
echo
echo "Mode:          $MODE"
echo "Camera ID:     $CAMERA_ID"
echo "Host:          $HOST"
echo "Port:          $PORT"
if [ -n "$DEBUG" ]; then echo "Debug:         Enabled"; fi
if [ "$OPTIMIZE" = true ]; then echo "Optimizations: Enabled"; fi
echo
echo "Starting application..."
echo

# Run the application with the specified parameters and Raspberry Pi optimizations
if [ "$OPTIMIZE" = true ]; then
  # Use lower resolution for better performance
  python3 main.py --mode "$MODE" --camera_id "$CAMERA_ID" --host "$HOST" --port "$PORT" --resolution 640x480 --frame_skip 2 $DEBUG $INPUT_DEVICE
else
  python3 main.py --mode "$MODE" --camera_id "$CAMERA_ID" --host "$HOST" --port "$PORT" $DEBUG $INPUT_DEVICE
fi

# If the application crashes, give feedback
echo
echo "Application has exited. Press Enter to close this window."
read 