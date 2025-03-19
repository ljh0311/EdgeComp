#!/bin/bash
# Baby Monitor System - Unified Installer and Repair Tool

echo "Starting Baby Monitor Unified Installer and Repair Tool..."

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT" || { echo "Failed to navigate to project directory"; exit 1; }

# Check if virtual environment exists
if [ -d "venv" ]; then
    # Activate virtual environment
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Run the unified manager GUI
python3 "$SCRIPT_DIR/scripts_manager_gui.py"

# Check if GUI crashed
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "The Unified Manager has crashed. Press Enter to close this window."
    read -r
fi

# Deactivate virtual environment
deactivate 