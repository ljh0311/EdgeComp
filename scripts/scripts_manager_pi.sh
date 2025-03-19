#!/bin/bash
# Baby Monitor System - Unified Installer and Repair Tool (Raspberry Pi Version)

echo "Starting Baby Monitor Unified Installer and Repair Tool (Raspberry Pi Version)..."

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT" || { echo "Failed to navigate to project directory"; exit 1; }

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "Warning: This script is optimized for Raspberry Pi."
    echo "Your device may not be a Raspberry Pi. Continuing anyway..."
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    # Activate virtual environment
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating..."
    
    # Check Python version
    PYTHON_VER=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VER | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VER | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        echo "Error: Python 3.8 or higher is required. Found $PYTHON_VER"
        echo "Press Enter to exit"
        read -r
        exit 1
    fi
    
    # Create virtual environment with optimizations for Pi
    echo "Creating optimized environment for Raspberry Pi..."
    python3 -m venv venv --system-site-packages
    source venv/bin/activate
    
    # Install basic requirements 
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Run the unified manager GUI with reduced resource usage
export PYTHONOPTIMIZE=1  # Enable optimizations
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=1073741824  # Reduce memory warnings

# Run with lower priority to avoid interfering with other processes
nice -n 10 python3 "$SCRIPT_DIR/scripts_manager_gui.py"

# Check if GUI crashed
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "The Unified Manager has crashed. Press Enter to close this window."
    read -r
fi

# Deactivate virtual environment
deactivate 