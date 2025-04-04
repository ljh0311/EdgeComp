#!/bin/bash

echo "======================================"
echo "  Baby Monitor System Installation"
echo "    (Raspberry Pi Optimized)"
echo "======================================"
echo "  Version 2.1.0 - State Detection"
echo "======================================"
echo

# Check if we're on a Raspberry Pi
if [ ! -f "/proc/device-tree/model" ] || ! grep -q "Raspberry Pi" "/proc/device-tree/model"; then
    echo "This script is specifically for Raspberry Pi devices."
    echo "For other Linux systems, please use install.sh instead."
    read -p "Continue anyway? (y/n): " CONTINUE
    if [[ $CONTINUE != [Yy]* ]]; then
        exit 1
    fi
fi

# Check for Pi 400 specifically
PI_MODEL=""
if [ -f "/proc/device-tree/model" ]; then
    PI_MODEL=$(cat /proc/device-tree/model)
    if [[ "$PI_MODEL" == *"Raspberry Pi 400"* ]]; then
        echo "Raspberry Pi 400 detected! Optimizing installation for your device."
        PI_400=true
    else
        echo "Raspberry Pi detected: $PI_MODEL"
        PI_400=false
    fi
fi

# Check for root privileges for system optimizations
IS_ROOT=false
if [ "$(id -u)" -eq 0 ]; then
    IS_ROOT=true
    echo "Running with root privileges - system optimizations will be applied."
else
    echo "Warning: Not running with root privileges."
    echo "Some system optimizations will not be applied."
    echo "Consider running with sudo for full optimization."
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    if [ "$IS_ROOT" = true ]; then
        echo "Installing Python 3..."
        apt-get update
        apt-get install -y python3 python3-pip python3-venv
    else
        echo "Please install Python 3.8 or higher and try again."
        echo "You can do this with: sudo apt-get install python3 python3-pip python3-venv"
        read -p "Press Enter to exit..." 
        exit 1
    fi
fi

# Determine Python version
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MAJOR=$(echo $PY_VERSION | cut -d. -f1)
MINOR=$(echo $PY_VERSION | cut -d. -f2)

# Verify Python version is at least 3.7 (Raspberry Pi OS typically has 3.7+)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 7 ]); then
    echo "Error: Python version $PY_VERSION is not supported."
    echo "Please install Python 3.7 or higher."
    read -p "Press Enter to exit..." 
    exit 1
fi

echo "Detected Python $PY_VERSION on Raspberry Pi"

# Install required system dependencies
if [ "$IS_ROOT" = true ]; then
    echo "Installing system dependencies..."
    apt-get update
    apt-get install -y \
        python3-dev \
    python3-pip \
    python3-venv \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
        libjpeg-dev \
        zlib1g-dev \
        libopenjp2-7 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
        libatlas-base-dev \
        ffmpeg \
        libwebp-dev  # Added for better image processing support
        
    # Extra dependencies for Raspberry Pi 400 to optimize performance
    if [ "$PI_400" = true ]; then
        echo "Installing Pi 400 specific optimizations..."
        apt-get install -y \
            rpi-eeprom \
            libraspberrypi-bin \
            raspi-config \
            libopenblas-dev  # For improved mathematical operations performance
    fi
fi

# Parse command line arguments
INSTALL=false
MODELS=false
FIX=false
CONFIG=false
TRAIN=false
DOWNLOAD=false
NO_GUI=false
OPTIMIZE=false
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
        --optimize)
            OPTIMIZE=true
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

    echo "Upgrading pip..."
pip install --upgrade pip

    echo "Installing required packages..."
    
    # Install packages with Pi-specific versions
    pip install numpy==1.21.0
    pip install --extra-index-url https://www.piwheels.org/simple/ pillow
    
    # For Pi 400, we can use newer torch version since it has more RAM
    if [ "$PI_400" = true ]; then
        echo "Installing optimized PyTorch for Pi 400..."
        pip install --extra-index-url https://www.piwheels.org/simple/ torch==1.10.0
    else
        echo "Installing PyTorch for Raspberry Pi..."
        pip install --extra-index-url https://www.piwheels.org/simple/ torch==1.9.0
    fi
    
    # Install remaining requirements
    pip install -r requirements.txt
    
    # Install additional packages for state detection if enabled
    if [ "$STATE_DETECTION" = true ]; then
        echo "Installing additional packages for state detection..."
        pip install scikit-learn==1.0.2 opencv-python-headless==4.5.5.64
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

# Optimize system if requested
if [ "$OPTIMIZE" = true ] && [ "$IS_ROOT" = true ]; then
    echo "Applying Raspberry Pi optimizations..."
    
    # Run the Raspberry Pi optimizer script if it exists
    if [ -f "tools/system/optimize_raspberry_pi.sh" ]; then
        echo "Running optimization script..."
        bash tools/system/optimize_raspberry_pi.sh
    else
        echo "Applying manual optimizations..."
        
        # CPU governor optimization
        echo "Setting CPU governor to performance mode..."
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo "performance" > "$cpu" 2>/dev/null || true
        done
        
        # Reduce swappiness
        echo "Optimizing swap settings..."
        sysctl -w vm.swappiness=10 > /dev/null
        
        # Pi 400 specific optimizations
        if [ "$PI_400" = true ]; then
            echo "Applying Pi 400 specific optimizations..."
            
            # Increase GPU memory for Pi 400 (has 4GB RAM)
            if [ -f /boot/config.txt ]; then
                sed -i '/^gpu_mem=/d' /boot/config.txt
                echo "gpu_mem=256" >> /boot/config.txt
                echo "Set GPU memory to 256MB for better video processing"
            fi
            
            # Set CPU scaling governor to performance
            if command -v raspi-config > /dev/null; then
                echo "Setting highest performance mode..."
                # Use raspi-config nonint to set maximum performance
                raspi-config nonint do_overclock 0
            fi
        else
            # Regular Pi optimizations
            # Increase GPU memory allocation if memory is sufficient
            TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
            if [ $TOTAL_MEM -gt 2000000 ]; then
                echo "Setting GPU memory to 256MB..."
                if [ -f /boot/config.txt ]; then
                    sed -i '/^gpu_mem=/d' /boot/config.txt
                    echo "gpu_mem=256" >> /boot/config.txt
                fi
            else
                echo "Setting GPU memory to 128MB..."
                if [ -f /boot/config.txt ]; then
                    sed -i '/^gpu_mem=/d' /boot/config.txt
                    echo "gpu_mem=128" >> /boot/config.txt
                fi
            fi
        fi
    fi
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

# For Raspberry Pi, always recommend downloading models instead of training
if [ "$DOWNLOAD" = true ]; then
    echo "Downloading pretrained models..."
    python3 setup.py --download --no-gui
elif [ "$TRAIN" = true ]; then
    echo "Warning: Training models on Raspberry Pi can be extremely slow."
    echo "It's recommended to download pretrained models instead."
    read -p "Do you want to continue with training? (y/n): " CONTINUE_TRAIN
    if [[ $CONTINUE_TRAIN == [Yy]* ]]; then
        echo "Training missing models... This may take a very long time."
        python3 setup.py --train --no-gui
    else
        echo "Downloading pretrained models instead..."
        python3 setup.py --download --no-gui
    fi
elif [ $MISSING_MODELS -gt 0 ]; then
    echo
    echo "Some required models are missing. On Raspberry Pi, we recommend:"
    echo "1. Download pretrained models (recommended)"
    echo "2. Train models from scratch (very slow on Raspberry Pi)"
    echo "3. Continue without models (not recommended)"
    echo "4. Use model_manager.sh for more options"
    echo
    read -p "Enter your choice (1-4): " MODEL_CHOICE
    
    if [ "$MODEL_CHOICE" = "1" ] || [ "$MODEL_CHOICE" = "" ]; then
        echo "Downloading pretrained models..."
        python3 setup.py --download --no-gui
    elif [ "$MODEL_CHOICE" = "2" ]; then
        echo "Warning: Training models on Raspberry Pi can be extremely slow."
        echo "This may take many hours or even days depending on your Pi model."
        read -p "Are you sure you want to continue? (y/n): " CONTINUE
        if [[ $CONTINUE == [yY]* ]]; then
            echo "Training models from scratch..."
            python3 setup.py --train --no-gui
        else
            echo "Downloading pretrained models instead..."
            python3 setup.py --download --no-gui
        fi
    elif [ "$MODEL_CHOICE" = "4" ]; then
        echo "Launching model manager..."
        if [ -f "model_manager.sh" ]; then
            chmod +x model_manager.sh
            ./model_manager.sh
        else
            echo "Error: model_manager.sh not found."
            echo "Downloading pretrained models instead..."
            python3 setup.py --download --no-gui
        fi
    else
        echo "Continuing without all models - some functionality may be limited."
    fi
fi

# Check for high performance processing requirements
if [ "$PI_400" = true ]; then
    echo "Configuring Baby Monitor for optimal performance on Raspberry Pi 400..."
    
    # Create optimized configuration for Pi 400
    if [ -f ".env" ]; then
        # Backup existing .env file
        cp .env .env.backup
        
        # Update .env with Pi 400 optimized settings
        sed -i 's/CAMERA_RESOLUTION=.*/CAMERA_RESOLUTION=640x480/' .env
        sed -i 's/CAMERA_FPS=.*/CAMERA_FPS=15/' .env
        
        # Enable state detection in .env if not already set
        if ! grep -q "STATE_DETECTION_ENABLED=" .env; then
            echo "STATE_DETECTION_ENABLED=true" >> .env
        elif [ "$STATE_DETECTION" = true ]; then
            sed -i 's/STATE_DETECTION_ENABLED=.*/STATE_DETECTION_ENABLED=true/' .env
        else
            sed -i 's/STATE_DETECTION_ENABLED=.*/STATE_DETECTION_ENABLED=false/' .env
        fi
        
        echo "Updated camera resolution to 640x480 and FPS to 15 for better performance."
    else
        # Create new .env file with Pi 400 optimized settings
        cat > .env << EOF
# Baby Monitor System Environment Configuration for Raspberry Pi 400
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
GPU_ENABLED=true
LOG_LEVEL=INFO
WEB_HOST=0.0.0.0
WEB_PORT=5000
EOF
        echo "Created optimized .env file for Raspberry Pi 400."
    fi
fi

# Handle specific installation modes
if [ "$INSTALL" = true ]; then
    echo "Installing Baby Monitor System..."
    python3 setup.py --install --no-gui
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
    echo "5. Optimize Raspberry Pi (requires root)"
    echo "6. Use model manager"
    echo "7. Toggle state detection (currently: ${STATE_DETECTION})"
    echo
    read -p "Enter your choice (1-7): " MENU_CHOICE
    
    if [ "$MENU_CHOICE" = "1" ]; then
        python3 setup.py --install --no-gui
    elif [ "$MENU_CHOICE" = "2" ]; then
        python3 setup.py --models --no-gui
    elif [ "$MENU_CHOICE" = "3" ]; then
        python3 setup.py --fix --no-gui
    elif [ "$MENU_CHOICE" = "4" ]; then
        python3 setup.py --config --no-gui
    elif [ "$MENU_CHOICE" = "5" ]; then
        if [ "$IS_ROOT" = true ]; then
            bash tools/system/optimize_raspberry_pi.sh
        else
            echo "Optimization requires root privileges."
            echo "Please run with 'sudo' for system optimizations."
        fi
    elif [ "$MENU_CHOICE" = "6" ]; then
        if [ -f "model_manager.sh" ]; then
            chmod +x model_manager.sh
            ./model_manager.sh
        else
            echo "Error: model_manager.sh not found."
            echo "Using setup.py model management instead..."
            python3 setup.py --models --no-gui
        fi
    elif [ "$MENU_CHOICE" = "7" ]; then
        # Toggle state detection
        if [ "$STATE_DETECTION" = true ]; then
            STATE_DETECTION=false
            echo "State detection disabled."
        else
            STATE_DETECTION=true
            echo "State detection enabled."
        fi
        
        if [ -f ".env" ]; then
            # Update .env file
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
    
    if [ "$PI_400" = true ]; then
        echo "Raspberry Pi 400 Recommendations:"
        echo "  - Use the built-in keyboard and a monitor for best experience"
        echo "  - Ensure proper cooling during extended use"
        echo "  - For best audio detection, use an external microphone"
        echo "  - Adjust the camera position for optimal coverage"
    else
        echo "Note: For best performance on Raspberry Pi, you can enable system optimizations:"
        echo "  sudo bash tools/system/optimize_raspberry_pi.sh"
    fi
    
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

# Add team credit section
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