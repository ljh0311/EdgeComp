# Edge Computing Baby Monitor

An intelligent baby monitoring system that leverages edge computing technology to provide real-time monitoring capabilities with advanced detection features and privacy-focused design.

## Overview

The Edge Computing Baby Monitor is designed to provide intelligent monitoring capabilities while processing all data locally on edge devices. It combines computer vision, audio processing, and machine learning to offer comprehensive monitoring features without relying on cloud services.

## Features

### ✅ Currently Working

- Real-time video capture and processing (Windows/Raspberry Pi)
- Person detection using YOLOv8 nano model
- Motion analysis and fall detection
- Web-based monitoring interface
- Real-time alerts and notifications
- Cross-platform support
- Resource-optimized processing

### 🚧 Under Development

- Audio processing and emotion detection
- Enhanced motion analysis
- Night vision capabilities
- Environmental monitoring
- Multi-camera support
- Mobile application interface

## System Architecture

```
[Camera/Audio Input] → [Edge Device (RPi/PC)]
         ↓
[Processing Layer]
- Person Detection (YOLOv8)
- Motion Analysis
- Audio Processing
         ↓
[Communication Layer]
- Socket.IO
- Event System
         ↓
[Interface Layer]
- Web Interface (Flask)
- Alert Management
```

## Technical Stack

- **Edge Processing**: Python, OpenCV, PyTorch
- **AI Models**: YOLOv8 nano (person detection)
- **Web Interface**: Flask, Socket.IO, Bootstrap
- **Video Processing**: OpenCV, V4L2/DirectShow
- **Development**: Python 3.8+, pip

## Installation

### Prerequisites

- Python 3.8 or higher
- Camera device (USB webcam/Raspberry Pi camera)
- Microphone (for audio features)
- Edge device (PC/Raspberry Pi)

### Windows Setup

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
python requirements.py
```

### Raspberry Pi Setup

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-picamera2 python3-pyaudio portaudio19-dev

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
python3 requirements.py
```

## Usage

1. Start the application:

```bash
python src/main.py
```

2. Access the web interface:

```
http://localhost:5000
```

## Configuration

### Environment Variables

- `CAMERA_INDEX`: Camera device index (default: 0)
- `USE_CUDA`: Enable CUDA acceleration (default: 0)
- `LOG_LEVEL`: Logging level (default: INFO)

### Web Interface Settings

- Detection confidence threshold
- Frame rate control
- Alert sensitivity
- Video quality settings

## Development

### Project Structure

```
project/
├── src/
│   ├── audio/          # Audio processing
│   ├── detectors/      # Detection models
│   ├── web/           # Web interface
│   ├── utils/         # Utilities
│   └── main.py        # Core system
├── models/            # AI models
├── requirements.txt   # Dependencies
└── README.md         # Documentation
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- Flask and Socket.IO teams
