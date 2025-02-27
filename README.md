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
- Environmental monitoring
- Multi-camera support
- Mobile application interface

## System Requirements

### Windows

- Windows 10 or later
- Python 3.8 or higher
- Webcam
- Microphone
- 4GB RAM minimum (8GB recommended)
- Intel Core i3/AMD Ryzen 3 or better

### Raspberry Pi 400

- Raspberry Pi OS (32-bit or 64-bit)
- Python 3.8 or higher
- Raspberry Pi Camera or USB webcam
- USB microphone
- 4GB RAM
- Active cooling recommended

## Installation

### Windows Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/edge-computing-baby-monitor.git
cd edge-computing-baby-monitor
```

2. Create and activate virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Download YOLOv8 model:

```bash
mkdir models
# Download YOLOv8 nano model to models directory
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o models/yolov8n.pt
```

### Raspberry Pi 400 Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/edge-computing-baby-monitor.git
cd edge-computing-baby-monitor
```

2. Run the installation script:

```bash
chmod +x install_pi.sh
./install_pi.sh
```

The installation script will:

- Install system dependencies
- Set up the camera module
- Create Python virtual environment
- Install optimized Python packages
- Download required models
- Configure environment variables

## Usage

### Running on Windows

1. Activate the virtual environment:

```bash
.\venv\Scripts\activate
```

2. Run in production mode (status and alerts only):

```bash
python src/main.py
```

3. Run in developer mode (with video feed and waveform):

```bash
python src/main.py --dev
```

### Running on Raspberry Pi 400

1. Activate the virtual environment:

```bash
source venv/bin/activate
```

2. Run in production mode:

```bash
python src/main.py
```

3. Run in developer mode:

```bash
python src/main.py --dev
```

### Accessing the Web Interface

The web interface will be available at:

```
http://localhost:5000
```

For remote access within your local network, use your device's IP address:

```
http://<device-ip>:5000
```

## Configuration

### Environment Variables

- `CAMERA_INDEX`: Camera device index (default: 0)
- `USE_CUDA`: Enable CUDA acceleration on Windows (default: 0)
- `LOG_LEVEL`: Logging level (default: INFO)

### Performance Settings

#### Windows

- Frame rate: 30 FPS
- Detection interval: 1 second
- Full visualization enabled

#### Raspberry Pi 400

- Frame rate: 15 FPS
- Detection interval: 2 seconds
- Visualization disabled
- Frame skipping enabled
- Reduced audio chunk size
- Limited thread pool

## Troubleshooting

### Windows

1. Camera not detected:
   - Check camera connection
   - Try different `CAMERA_INDEX` values
   - Verify camera works in other applications

2. Performance issues:
   - Close resource-intensive applications
   - Reduce resolution in config.py
   - Disable developer mode

### Raspberry Pi 400

1. Camera issues:
   - Run `sudo modprobe bcm2835-v4l2`
   - Check camera connection
   - Verify camera module is enabled in raspi-config

2. Performance optimization:
   - Monitor temperature: `vcgencmd measure_temp`
   - Check CPU usage: `top`
   - Ensure active cooling
   - Reduce resolution if needed

3. Audio issues:
   - Check audio device: `arecord -l`
   - Test microphone: `arecord -D plughw:1,0 test.wav`
   - Verify audio permissions

## Support

For issues and support:

1. Check the troubleshooting guide
2. Review logs in `logs` directory
3. Open an issue on GitHub
4. Contact the development team

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- Flask and Socket.IO teams
