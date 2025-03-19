#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}Baby Monitor Raspberry Pi Fix Utility${NC}"
echo -e "${GREEN}====================================${NC}"
echo

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3.11; then
        echo -e "${GREEN}Found Python 3.11${NC}"
        return 0
    else
        echo -e "${RED}Python 3.11 not found${NC}"
        return 1
    fi
}

echo "Step 1: Checking system requirements..."

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo -e "${RED}Error: This script must be run on a Raspberry Pi${NC}"
    exit 1
fi

# Install system dependencies
echo "Step 2: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    libhdf5-serial-dev \
    libharfbuzz0b \
    libwebp6 \
    libtiff5 \
    libjasper1 \
    libilmbase23 \
    libopenexr23 \
    libgstreamer1.0-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libportaudio2 \
    libsndfile1

# Enable camera and audio if needed
echo "Step 3: Configuring Raspberry Pi..."
if ! grep -q "start_x=1" /boot/config.txt; then
    echo -e "${YELLOW}Enabling camera module...${NC}"
    sudo raspi-config nonint do_camera 0
    sudo sh -c 'echo "start_x=1" >> /boot/config.txt'
    sudo sh -c 'echo "gpu_mem=128" >> /boot/config.txt'
fi

# Stop any running processes
echo "Step 4: Stopping any running Baby Monitor processes..."
pkill -f "python.*run_monitor.py"
sleep 2

# Clean Python cache
echo "Step 5: Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Check camera module
echo "Step 6: Testing camera module..."
if ! vcgencmd get_camera | grep -q "supported=1 detected=1"; then
    echo -e "${YELLOW}Warning: Camera module not detected${NC}"
    echo "Please check if the camera module is properly connected"
    echo "You may need to reboot after connecting the camera"
fi

# Reset virtual environment
echo "Step 7: Setting up virtual environment..."
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate

# Install Python packages
echo "Step 8: Installing required packages..."
python -m pip install --upgrade pip
python -m pip install numpy==1.24.3
python -m pip install picamera2
python -m pip install opencv-python-headless==4.8.1.78
python -m pip install eventlet==0.33.3 flask-socketio==5.3.6 werkzeug==2.3.7
python -m pip install python-engineio==4.5.1 python-socketio==5.8.0
python -m pip install torch==2.0.0 torchaudio==2.0.0
python -m pip install librosa==0.10.0 sounddevice==0.4.6 soundfile==0.12.1
python -m pip install flask==2.3.3 python-dotenv==1.0.0 psutil==5.9.5

# Set up environment variables
echo "Step 9: Setting up environment variables..."
export PYTHONPATH=$(pwd)
export EVENTLET_NO_GREENDNS=yes

# Create systemd service for autostart (optional)
echo "Step 10: Creating systemd service (optional)..."
read -p "Would you like to create a systemd service to start Baby Monitor on boot? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo bash -c "cat > /etc/systemd/system/babymonitor.service << EOL
[Unit]
Description=Baby Monitor System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PYTHONPATH=$(pwd)
Environment=EVENTLET_NO_GREENDNS=yes
ExecStart=$(pwd)/venv/bin/python run_monitor.py --mode prod --camera_id 0
Restart=always

[Install]
WantedBy=multi-user.target
EOL"
    
    sudo systemctl daemon-reload
    sudo systemctl enable babymonitor.service
    echo -e "${GREEN}Systemd service created and enabled${NC}"
fi

echo -e "\n${GREEN}Setup completed!${NC}"
echo "You can now:"
echo "1. Run the application: ./venv/bin/python run_monitor.py --mode dev --camera_id 0 --debug"
echo "2. Access the web interface at: http://localhost:5000"
echo
echo "If you encounter issues:"
echo "- Check if the camera module is properly connected"
echo "- Ensure the camera is enabled in raspi-config"
echo "- Try rebooting the system if camera is not detected"
echo "- Check the system logs: journalctl -u babymonitor.service"
echo
echo -e "${YELLOW}Note: You may need to reboot for all changes to take effect${NC}"
echo "Would you like to reboot now? (y/n)"
read -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi 