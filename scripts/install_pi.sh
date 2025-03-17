#!/bin/bash

echo "🔧 Installing Baby Monitor System for Raspberry Pi 400..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-picamera2 \
    libportaudio2 \
    portaudio19-dev \
    python3-opencv \
    libopencv-dev \
    python3-venv \
    git

# Setup camera module
echo "📸 Setting up camera..."
sudo modprobe bcm2835-v4l2
# Add module to load at boot
if ! grep -q "bcm2835-v4l2" /etc/modules; then
    echo "bcm2835-v4l2" | sudo tee -a /etc/modules
fi

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install wheel
pip install numpy==1.21.0  # Specific version for Pi compatibility
pip install -r requirements.txt

# Create models directory if it doesn't exist
mkdir -p models

# Download YOLOv8 nano model if not exists
if [ ! -f "models/yolov8n.pt" ]; then
    echo "📥 Downloading YOLOv8 nano model..."
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
fi

# Set up environment variables
echo "⚙️ Setting up environment variables..."
echo 'export PYTHONPATH=$PYTHONPATH:$(pwd)' >> venv/bin/activate
echo 'export USE_CUDA=0' >> venv/bin/activate

# Create log directory
mkdir -p logs

echo "✅ Installation complete!"
echo "To start the system:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the system: python src/main.py" 