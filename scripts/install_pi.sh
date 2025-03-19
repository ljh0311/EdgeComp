#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}Baby Monitor - Raspberry Pi Installer${NC}"
echo -e "${GREEN}====================================${NC}"
echo

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo -e "${RED}Error: This script is intended for Raspberry Pi systems only.${NC}"
    echo "Current platform: $(uname -a)"
    exit 1
fi

# Check for Python 3.11
echo "Checking Python version..."
if ! command -v python3.11 &> /dev/null; then
    echo -e "${YELLOW}Python 3.11 not found. Installing...${NC}"
    
    # Add deadsnakes PPA for Python 3.11
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    
    # Install Python 3.11
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# Verify Python 3.11 installation
python3.11 --version || {
    echo -e "${RED}Failed to install Python 3.11${NC}"
    exit 1
}

echo -e "${GREEN}Installing Baby Monitor System dependencies for Raspberry Pi...${NC}"

# Update package list and upgrade existing packages
echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libatlas-base-dev \
    libjasper-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
    libilmbase-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    python3-picamera2 \
    python3-opencv \
    libopencv-dev

# Create virtual environment with Python 3.11
echo "Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies optimized for Pi
echo "Installing Python packages..."
pip install \
    eventlet==0.33.3 \
    flask-socketio==5.3.6 \
    werkzeug==2.3.7 \
    python-engineio==4.5.1 \
    python-socketio==5.8.0 \
    opencv-python-headless==4.5.3.56 \
    numpy==1.21.0 \
    --extra-index-url https://www.piwheels.org/simple \
    torch==1.10.0 \
    torchaudio==0.10.0 \
    librosa==0.8.1 \
    sounddevice==0.4.3 \
    soundfile==0.10.3.post1 \
    flask==2.0.1 \
    psutil==5.8.0

# Setup camera module
echo "Setting up camera module..."
sudo modprobe bcm2835-v4l2

# Add module to load at boot
if ! grep -q "bcm2835-v4l2" /etc/modules-load.d/modules.conf 2>/dev/null; then
    echo "bcm2835-v4l2" | sudo tee -a /etc/modules-load.d/modules.conf
fi

# Create models directory
echo "Creating models directory..."
mkdir -p models/emotion

# Create a startup script
echo "Creating startup script..."
cat > start_monitor.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export PYTHONPATH=.
export EVENTLET_NO_GREENDNS=yes
python run_monitor.py --mode dev --camera_id 0 --debug
EOF

chmod +x start_monitor.sh

echo -e "${GREEN}Installation complete!${NC}"
echo
echo "To start the system:"
echo "1. Run: ./start_monitor.sh"
echo "2. Open http://localhost:5000 in your web browser"
echo
echo -e "${YELLOW}Note: Make sure to enable the camera interface using:${NC}"
echo "sudo raspi-config"
echo "Navigate to: Interface Options > Camera > Enable"
echo
