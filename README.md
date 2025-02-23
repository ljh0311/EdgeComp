# Edge Computing Baby Monitor

An intelligent baby monitoring system that combines real-time cry analysis and person detection using edge computing technology, with a web-based interface for easy monitoring.

## Current Status

### ✅ Currently Working
- Real-time video capture from camera (both Windows and Raspberry Pi)
- Person detection using YOLOv8
- Pose estimation with hand and leg tracking
- Position detection (standing, sitting, lying down)
- Web interface with real-time video streaming
- Cross-platform compatibility (Windows/Raspberry Pi)
- Basic logging system

### 🚧 Under Development
- Baby-specific detection and tracking
- Cry detection and analysis
- Audio processing pipeline
- Night vision implementation
- Temperature and humidity monitoring
- Data storage and analysis features

## Project Structure

```
project/
├── src/                      # Source code
│   ├── audio/               # Audio processing modules
│   ├── detectors/           # Detection algorithms
│   ├── static/              # Web static files
│   ├── templates/           # Web templates
│   ├── ui/                  # UI components
│   ├── utils/              # Utility functions
│   ├── main.py             # Core monitoring logic
│   └── web_app.py          # Flask web application
├── models/                  # AI model files
├── run.py                  # Main entry point
├── requirements.txt        # Python dependencies
├── requirements.py         # Platform-specific setup
├── install_pi.sh          # Raspberry Pi installation script
└── .env                   # Environment configuration
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Camera device (USB webcam or Raspberry Pi camera)
- Microphone (for future audio features)

### Windows Installation
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
python requirements.py
```

### Raspberry Pi Installation
```bash
# Run the installation script
chmod +x install_pi.sh
./install_pi.sh

# Or manually:
sudo apt update
sudo apt install -y python3-picamera2 python3-pyaudio portaudio19-dev
python3 -m venv venv
source venv/bin/activate
python3 requirements.py
```

## Running the Application

1. Activate the virtual environment:
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # Raspberry Pi
   source venv/bin/activate
   ```

2. Start the application:
   ```bash
   python run.py
   ```

3. Access the web interface at `http://localhost:5000`

## Configuration

The system can be configured through the following methods:

1. Environment Variables (`.env` file):
   - `CAMERA_INDEX`: Camera device index (default: 0)
   - `MODEL_PATH`: Path to YOLOv8 model (default: "yolov8n.pt")
   - `LOG_LEVEL`: Logging level (default: "INFO")

2. Web Interface Settings:
   - Detection confidence threshold
   - Position detection sensitivity
   - Video quality and frame rate
   - Alert settings

## Features and Usage

### Video Monitoring
- Real-time video feed with person detection
- Position tracking (standing/sitting/lying)
- Pose estimation with body part tracking
- Detection confidence display

### Web Interface
- Live video stream viewing
- Detection settings adjustment
- System status monitoring
- Alert configuration

### Logging
- System events and detections are logged to `baby_monitor.log`
- Configurable log levels for debugging

## Troubleshooting

1. Camera Issues:
   - Check camera connection
   - Verify camera index in `.env` file
   - Ensure camera permissions are set

2. Performance Issues:
   - Lower video resolution in web interface settings
   - Adjust detection frequency
   - Check system resource usage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details