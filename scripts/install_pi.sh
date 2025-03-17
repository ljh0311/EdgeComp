#!/bin/bash

# Exit on error
set -e

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "Error: This script is intended for Raspberry Pi systems only."
    echo "Current platform: $(uname -a)"
    exit 1
fi

echo "Installing Baby Monitor System dependencies for Raspberry Pi..."

# Update package list and upgrade existing packages
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
    libx264-dev

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies optimized for Pi
echo "Installing Python packages..."
pip install \
    numpy==1.21.0 \
    opencv-python-headless==4.5.3.56 \
    --extra-index-url https://www.piwheels.org/simple \
    torch==1.10.0 \
    torchaudio==0.10.0 \
    librosa==0.8.1 \
    sounddevice==0.4.3 \
    soundfile==0.10.3.post1 \
    flask==2.0.1 \
    flask-socketio==5.1.1 \
    psutil==5.8.0

# Create models directory
echo "Creating models directory..."
mkdir -p src/babymonitor/detectors/models/emotion

# Download models
echo "Downloading models..."
python3 -c "
from babymonitor.detectors import PersonDetector, EmotionDetector
print('Initializing person detector...')
detector = PersonDetector()
print('Initializing emotion detector...')
emotion = EmotionDetector()
"

echo "Installation complete!"
echo "To start the system:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the baby monitor: python scripts/run_babymonitor.py"
echo "3. For emotion recognition only: python scripts/run_emotion_recognition.py"
