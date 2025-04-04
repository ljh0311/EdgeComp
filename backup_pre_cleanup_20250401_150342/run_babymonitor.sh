#!/bin/bash

echo "======================================"
echo "      Baby Monitor System Launcher"
echo "======================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed."
    echo "Please install Python 3.8 or higher and try again."
    read -p "Press Enter to continue..."
    exit 1
fi

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Setting up..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo "Setup complete!"
else
    source venv/bin/activate
fi

function show_menu {
    clear
    echo "======================================"
    echo "      Baby Monitor System Launcher"
    echo "======================================"
    echo
    echo "Please select an option:"
    echo
    echo "[1] Start Baby Monitor (Normal Mode)"
    echo "[2] Start Baby Monitor (Developer Mode)"
    echo "[3] Run API Test"
    echo "[4] Backup and Restore"
    echo "[5] Check Requirements"
    echo "[6] Exit"
    echo
}

while true; do
    show_menu
    read -p "Enter your choice (1-6): " choice
    
    case $choice in
        1)
            echo
            echo "Starting Baby Monitor in Normal Mode..."
            python -m babymonitor.launcher -i gui -m normal
            ;;
        2)
            echo
            echo "Starting Baby Monitor in Developer Mode..."
            python -m babymonitor.launcher -i gui -m dev
            ;;
        3)
            echo
            echo "Running API Tests..."
            python tests/updated/api_test.py
            echo
            read -p "Press Enter to continue..."
            ;;
        4)
            bash tools/backup/restore.sh
            ;;
        5)
            echo
            echo "Checking requirements..."
            pip list | grep -E "opencv-python|PyQt5|torch|transformers|sounddevice|librosa"
            echo
            read -p "Press Enter to continue..."
            ;;
        6)
            echo
            echo "Thank you for using Baby Monitor System."
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            sleep 2
            ;;
    esac
done 