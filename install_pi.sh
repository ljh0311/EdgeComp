#!/bin/bash

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-picamera2 \
    libportaudio2 \
    portaudio19-dev \
    python3-opencv \
    libopencv-dev

echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setting up camera..."
sudo modprobe bcm2835-v4l2

echo "Installation complete! Run 'source venv/bin/activate' to activate the virtual environment" 