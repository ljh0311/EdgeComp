# Edge Computing Baby Monitor

An intelligent baby monitoring system that combines real-time cry analysis and person detection using edge computing technology.

## Current Status

### ✅ Currently Working
- Real-time video capture from camera (both Windows and Raspberry Pi)
- Person detection using YOLOv8
- Pose estimation with hand and leg tracking
- Position detection (standing, sitting, lying down)
- Basic logging system
- Cross-platform compatibility (Windows/Raspberry Pi)

### 🚧 Under Development
- Baby-specific detection and tracking
- Cry detection and analysis
- Audio processing pipeline
- Night vision implementation
- User interface
- Temperature and humidity monitoring
- Data storage and analysis features

## Overview

This project implements a smart baby monitor that helps parents and caregivers by:
- Detecting and tracking the baby's position in real-time
- Analyzing baby cries to identify potential causes (hunger, discomfort, tiredness)
- Monitoring room occupancy through person detection
- Operating effectively in various lighting conditions, including night time
- Processing all data locally on the edge device for enhanced privacy

## Features

### Implemented Features
- **Person Detection**
  - Real-time person detection using YOLOv8
  - Position classification (standing/sitting/lying)
  - Body part tracking (hands and legs)
  - Confidence score display

### Planned Features
- **Baby Monitoring**
  - Baby-specific detection and tracking
  - Movement pattern analysis
  - Safety zone monitoring

- **Audio Analysis**
  - Cry detection and pattern recognition
  - Classification of different types of cries
  - Noise filtering

## Hardware Requirements

- Raspberry Pi 4 (Edge Computing Device)
- High-resolution camera with night vision
- Sensitive microphone with noise cancellation
- Power supply units
- SD card for storage
- Protective casing
- Optional: Temperature and humidity sensors

## Software Dependencies

- Python 3.x
- OpenCV
- PyAudio
- Ultralytics (YOLOv8)
- Torch
- YOLOv8 Pose Estimation Model

## Installation

### Automatic Installation
```bash
# For Windows
python requirements.py

# For Raspberry Pi
python3 requirements.py
```

### Manual Installation
### Windows
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python numpy
pip install pipwin
pipwin install pyaudio
```

### Raspberry Pi
```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-picamera2 python3-pyaudio portaudio19-dev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install ultralytics opencv-python numpy pyaudio
```

## Usage

1. Clone the repository
2. Install dependencies as per your platform
3. Run the main script:
```bash
python main.py
```

## Project Structure

```
project/
├── src/
│   ├── video_processing/  # Video and detection modules
│   ├── audio_processing/  # Audio and cry analysis (planned)
│   ├── models/           # AI model files
│   └── utils/            # Utility functions
├── tests/                # Test files
├── docs/                 # Documentation
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details

## Acknowledgments

This project uses research and datasets from:
- [Infant Cry Audio Corpus](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus)
- [Cry Classification Research](https://www.kaggle.com/code/warcoder/classifying-infant-cry-type/notebook)