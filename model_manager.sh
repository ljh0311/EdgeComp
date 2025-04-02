#!/bin/bash

echo "========================================="
echo "Baby Monitor System - Model Manager v1.0"
echo "========================================="
echo

# Check if we're in the correct directory
if [ ! -f "setup.py" ]; then
    echo "ERROR: This script must be run from the project root directory."
    echo "Current directory: $(pwd)"
    echo
    echo "Please navigate to the project root directory and try again."
    read -p "Press Enter to exit..." 
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found."
    echo "Please install Python 3.8 or higher and ensure it's in your PATH."
    read -p "Press Enter to exit..." 
    exit 1
fi

# Use python3 command explicitly on Linux/macOS
PYTHON_CMD="python3"

# If Raspberry Pi detected, mention special considerations
if [ -f "/proc/device-tree/model" ] && grep -q "Raspberry Pi" "/proc/device-tree/model"; then
    echo "Raspberry Pi detected. Note that training models directly on a Raspberry Pi"
    echo "is not recommended due to limited resources. Consider downloading pretrained models."
    echo
fi

# Display model status
echo "Checking model status..."
$PYTHON_CMD -c "import os; print('\nRequired models directory: ' + os.path.abspath('src/babymonitor/models'))"
$PYTHON_CMD setup.py --models --no-gui

echo
echo "What would you like to do?"
echo
echo "[1] Download missing models (recommended)"
echo "[2] Download all models (overwrite existing)"
echo "[3] Train emotion model"
echo "[4] Train speech recognition model"
echo "[5] Train all models"
echo "[6] Exit"
echo

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo
        echo "Downloading missing models..."
        $PYTHON_CMD setup.py --download --no-gui
        ;;
    2)
        echo
        echo "Downloading all models (this will overwrite existing models)..."
        $PYTHON_CMD setup.py --download --train-all --no-gui
        ;;
    3)
        echo
        echo "Training emotion model..."
        echo "NOTE: This may take several hours depending on your computer's performance."
        echo
        read -p "Are you sure you want to proceed? (y/n): " confirm
        if [[ "$confirm" == [yY]* ]]; then
            $PYTHON_CMD setup.py --train --specific-model emotion_model --no-gui
        else
            echo "Training cancelled."
        fi
        ;;
    4)
        echo
        echo "Training speech recognition model..."
        echo "NOTE: This requires significant computational resources and may take several hours."
        echo
        read -p "Are you sure you want to proceed? (y/n): " confirm
        if [[ "$confirm" == [yY]* ]]; then
            $PYTHON_CMD setup.py --train --specific-model wav2vec2_model --no-gui
        else
            echo "Training cancelled."
        fi
        ;;
    5)
        echo
        echo "Training all models..."
        echo "NOTE: This will take a very long time and requires significant resources."
        echo "Consider downloading pretrained models instead if possible."
        echo
        read -p "Are you sure you want to proceed? (y/n): " confirm
        if [[ "$confirm" == [yY]* ]]; then
            $PYTHON_CMD setup.py --train --train-all --no-gui
        else
            echo "Training cancelled."
        fi
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please try again."
        ;;
esac

echo
echo "Model management complete. Displaying final model status..."
$PYTHON_CMD setup.py --models --no-gui

echo
echo "Thank you for using the Baby Monitor System Model Manager."
read -p "Press Enter to exit..." 