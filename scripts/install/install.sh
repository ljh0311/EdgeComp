#!/bin/bash

echo "======================================"
echo "  Baby Monitor System Installation"
echo "======================================"
echo "  Version 2.1.0 - State Detection"
echo "======================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Please install Python 3.8 or higher and try again."
    read -p "Press Enter to exit..." 
    exit 1
fi

# Determine Python version
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MAJOR=$(echo $PY_VERSION | cut -d. -f1)
MINOR=$(echo $PY_VERSION | cut -d. -f2)

# Verify Python version is at least 3.8
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
    echo "Error: Python version $PY_VERSION is not supported."
    echo "Please install Python 3.8 or higher."
    read -p "Press Enter to exit..." 
    exit 1
fi

echo "Detected Python $PY_VERSION"

# Parse command line arguments
INSTALL=false
MODELS=false
FIX=false
CONFIG=false
TRAIN=false
DOWNLOAD=false
NO_GUI=false
STATE_DETECTION=true  # Enable state detection by default

for arg in "$@"; do
    case $arg in
        --install)
            INSTALL=true
            ;;
        --models)
            MODELS=true
            ;;
        --fix)
            FIX=true
            ;;
        --config)
            CONFIG=true
            ;;
        --train)
            TRAIN=true
            ;;
        --download)
            DOWNLOAD=true
            ;;
        --no-gui)
            NO_GUI=true
            ;;
        --no-state-detection)
            STATE_DETECTION=false
            ;;
    esac
done

# Set up virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Install additional packages for state detection if enabled
    if [ "$STATE_DETECTION" = true ]; then
        echo "Installing additional packages for state detection..."
        pip install scikit-learn opencv-python
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error installing requirements."
        echo "Please check your internet connection and try again."
        read -p "Press Enter to exit..." 
    exit 1
    fi
else
    echo "Activating existing virtual environment..."
    source venv/bin/activate
fi

# Create models directory if it doesn't exist
if [ ! -d "src/babymonitor/models" ]; then
    echo "Creating models directory..."
    mkdir -p src/babymonitor/models
fi

# Check for models
echo
echo "Checking model status..."

# Count total models and missing models
TOTAL_MODELS=0
MISSING_MODELS=0

if [ ! -f "src/babymonitor/models/emotion_model.pt" ]; then
    echo "- Missing: Emotion recognition model"
    MISSING_MODELS=$((MISSING_MODELS + 1))
fi
TOTAL_MODELS=$((TOTAL_MODELS + 1))

if [ ! -f "src/babymonitor/models/yolov8n.pt" ]; then
    echo "- Missing: Person detection model"
    MISSING_MODELS=$((MISSING_MODELS + 1))
fi
TOTAL_MODELS=$((TOTAL_MODELS + 1))

if [ ! -f "src/babymonitor/models/wav2vec2_emotion.pt" ]; then
    echo "- Missing: Wav2Vec2 emotion model"
    MISSING_MODELS=$((MISSING_MODELS + 1))
fi
TOTAL_MODELS=$((TOTAL_MODELS + 1))

# Check for state detection model if enabled
if [ "$STATE_DETECTION" = true ]; then
    if [ ! -f "src/babymonitor/models/person_state_classifier.pkl" ]; then
        echo "- Missing: Person state detection model"
        MISSING_MODELS=$((MISSING_MODELS + 1))
    fi
    TOTAL_MODELS=$((TOTAL_MODELS + 1))
fi

echo
echo "Found $MISSING_MODELS missing models out of $TOTAL_MODELS required models."

# Handle models based on command line args or user input
if [ "$DOWNLOAD" = true ]; then
    echo "Downloading pretrained models..."
    python3 setup.py --download --no-gui
elif [ "$TRAIN" = true ]; then
    echo "Training missing models..."
    python3 setup.py --train --no-gui
elif [ $MISSING_MODELS -gt 0 ]; then
    echo
    echo "Some required models are missing. What would you like to do?"
    echo "1. Download pretrained models (recommended)"
    echo "2. Train models from scratch (takes time and requires data)"
    echo "3. Continue without models (not recommended)"
    echo
    read -p "Enter your choice (1-3): " MODEL_CHOICE
    
    if [ "$MODEL_CHOICE" = "1" ]; then
        echo "Downloading pretrained models..."
        python3 setup.py --download --no-gui
    elif [ "$MODEL_CHOICE" = "2" ]; then
        echo "Training models from scratch..."
        python3 setup.py --train --no-gui
    else
        echo "Continuing without all models - some functionality may be limited."
    fi
fi

# Update or create .env file
if [ -f ".env" ]; then
    # Backup existing .env file
    cp .env .env.backup
    
    # Update .env with state detection setting
    if ! grep -q "STATE_DETECTION_ENABLED=" .env; then
        echo "STATE_DETECTION_ENABLED=true" >> .env
    elif [ "$STATE_DETECTION" = true ]; then
        sed -i 's/STATE_DETECTION_ENABLED=.*/STATE_DETECTION_ENABLED=true/' .env
    else
        sed -i 's/STATE_DETECTION_ENABLED=.*/STATE_DETECTION_ENABLED=false/' .env
    fi
    
    echo "Updated configuration in .env file."
else
    # Create new .env file with default settings
    cat > .env << EOF
# Baby Monitor System Environment Configuration
# Generated by installer on $(date "+%Y-%m-%d %H:%M:%S")
MODE=normal
CAMERA_INDEX=0
CAMERA_RESOLUTION=640x480
CAMERA_FPS=15
AUDIO_DEVICE_INDEX=0
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
DETECTION_THRESHOLD=0.5
EMOTION_THRESHOLD=0.7
EMOTION_MODEL=basic_emotion
EMOTION_HISTORY_ENABLED=true
EMOTION_HISTORY_SAVE_INTERVAL=300
STATE_DETECTION_ENABLED=${STATE_DETECTION}
LOG_LEVEL=INFO
WEB_HOST=0.0.0.0
WEB_PORT=5000
EOF
    echo "Created default .env configuration file."
fi

# Handle specific installation modes
if [ "$INSTALL" = true ]; then
    echo "Installing Baby Monitor System..."
    if [ "$STATE_DETECTION" = false ]; then
        python3 setup.py --install --no-gui --no-state-detection
    else
        python3 setup.py --install --no-gui
    fi
elif [ "$MODELS" = true ]; then
    echo "Managing models..."
    python3 setup.py --models --no-gui
elif [ "$FIX" = true ]; then
    echo "Fixing common issues..."
    python3 setup.py --fix --no-gui
elif [ "$CONFIG" = true ]; then
    echo "Configuring system..."
    python3 setup.py --config --no-gui
elif [ "$NO_GUI" = true ]; then
    echo
    echo "Choose what you want to do:"
    echo "1. Install/reinstall the system"
    echo "2. Manage models"
    echo "3. Fix common issues"
    echo "4. Configure system"
    echo "5. Toggle state detection (currently: ${STATE_DETECTION})"
    echo
    read -p "Enter your choice (1-5): " MENU_CHOICE
    
    if [ "$MENU_CHOICE" = "1" ]; then
        if [ "$STATE_DETECTION" = false ]; then
            python3 setup.py --install --no-gui --no-state-detection
        else
            python3 setup.py --install --no-gui
        fi
    elif [ "$MENU_CHOICE" = "2" ]; then
        python3 setup.py --models --no-gui
    elif [ "$MENU_CHOICE" = "3" ]; then
        python3 setup.py --fix --no-gui
    elif [ "$MENU_CHOICE" = "4" ]; then
        python3 setup.py --config --no-gui
    elif [ "$MENU_CHOICE" = "5" ]; then
        # Toggle state detection
        if [ "$STATE_DETECTION" = true ]; then
            STATE_DETECTION=false
            echo "State detection disabled."
        else
            STATE_DETECTION=true
            echo "State detection enabled."
        fi
        
        # Update .env file
        if [ -f ".env" ]; then
            if grep -q "STATE_DETECTION_ENABLED=" .env; then
                sed -i "s/STATE_DETECTION_ENABLED=.*/STATE_DETECTION_ENABLED=${STATE_DETECTION}/" .env
            else
                echo "STATE_DETECTION_ENABLED=${STATE_DETECTION}" >> .env
            fi
            echo "Updated .env configuration."
        fi
    else
        echo "Invalid choice."
    fi
else
    # No specific mode - launch GUI if available
    python3 setup.py
fi

# Check for successful installation
if [ -d "src/babymonitor" ] && [ -f "run_babymonitor.sh" ]; then
    echo
    echo "Baby Monitor has been successfully installed!"
    echo
    echo "To launch the application, run:"
    echo "  ./run_babymonitor.sh"
    echo
    
    # Make the launch script executable
    chmod +x run_babymonitor.sh
    
    # Display information about state detection
    if [ "$STATE_DETECTION" = true ]; then
        echo
        echo "Person State Detection is ENABLED"
        echo "The system will identify whether people are seated, lying, moving, or standing"
        echo "View these states in the metrics dashboard"
    else
        echo
        echo "Note: Person State Detection is disabled"
        echo "To enable it, edit .env and set STATE_DETECTION_ENABLED=true"
    fi
fi

# Display team information
echo
echo "======================================"
echo "  Baby Monitor System Developed By:"
echo "======================================"
echo "• JunHong: Backend Processing & Client Logic"
echo "• Darrel: Dashboard Frontend"
echo "• Ashraf: Datasets & Model Architecture" 
echo "• Xuan Yu: Specialized Datasets & Training"
echo "• Javin: Camera Detection System"
echo "======================================"

# Deactivate virtual environment
deactivate

echo
echo "Installation process completed."
echo
read -p "Press Enter to exit..." 