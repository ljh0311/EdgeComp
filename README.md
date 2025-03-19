# Baby Monitor System

A comprehensive baby monitoring solution with real-time person detection, emotion recognition, and web interface.

![Baby Monitor System](docs/images/screenshot.png)

## Features

- **Person Detection**: Detects faces and bodies using Haar Cascade classifiers
- **Emotion Recognition**: Identifies baby emotions (crying, laughing, babbling, silence) from audio
- **Web Interface**: Real-time monitoring dashboard with alerts and statistics
- **Multi-threading**: Efficient processing of video and audio streams
- **Alerting System**: Notifications when crying is detected or when a person enters/leaves the frame
- **Multiple Launch Modes**: Normal, Developer, and Local GUI modes for different use cases
- **Dark Theme UI**: Modern dark-themed interface with responsive design

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- SoundDevice
- Flask
- SocketIO
- PyQt5 (for local GUI mode)

## Installation

### Installation Scripts

All installation and setup scripts have been reorganized into the `scripts/install` directory:

- **Windows Installation**: Use `scripts/install/install.bat`
- **Linux/Raspberry Pi Installation**: Use `scripts/install/install.sh` or `scripts/install/install_pi.sh`
- **GUI Installation**: Run `python scripts/install/install.py` for the graphical installer

### Maintenance Scripts

Various maintenance and utility scripts are available in the `scripts` directory:

- **Windows Fix Utility**: `scripts/fix_windows.bat`
- **Raspberry Pi Fix Utility**: `scripts/fix_raspberry.sh`
- **Metrics Fix Utility**: `scripts/fix_metrics.bat`
- **Restart Monitor**: `scripts/restart_baby_monitor.bat`

## Usage

### Launch Modes

The Baby Monitor System now supports three launch modes:

#### Normal Mode (Default)

Standard mode for end users, showing the main dashboard with camera feed and metrics, but no access to development tools.

```bash
python main.py --mode normal
```

#### Developer Mode

Shows metrics page with access to all development tools, logs, and settings. Use this mode for debugging and development.

```bash
python main.py --mode dev
```

#### Local GUI Mode

Runs the local GUI version of the baby monitor using PyQt5, without the web interface.

```bash
python main.py --mode local
```

### Command Line Arguments

- `--mode`: Launch mode (`normal`, `dev`, `local`)
- `--threshold`: Detection threshold (default: 0.5)
- `--camera_id`: Camera ID (default: 0)
- `--input_device`: Audio input device ID (default: None)
- `--host`: Web interface host (default: 0.0.0.0)
- `--port`: Web interface port (default: 5000)
- `--debug`: Enable debug mode

### Legacy Usage (Deprecated)

The following commands are still supported for backward compatibility but are deprecated:

```bash
python scripts/run.py --mode web  # Web interface only
python scripts/run.py --mode person  # Person detection only
python scripts/run.py --mode emotion  # Emotion recognition only
python scripts/run.py --mode full  # Full system
```

## Troubleshooting

### Camera Issues

If you encounter camera access issues, try:

```bash
python main.py --camera_id 1
```

Different camera IDs (0, 1, 2, etc.) can be tried if your default camera is not working.

### Audio Device Issues

List available audio devices:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Then specify a device:

```bash
python main.py --input_device 1
```

### Repair Tools

The system includes built-in repair tools accessible from the web interface:

1. Navigate to the "Repair Tools" page
2. Select the component to repair (Camera, Audio, or System)
3. Click the repair button to attempt to fix the issue

## Project Structure

```
baby-monitor-system/
├── main.py                  # Main entry point with multiple launch modes
├── src/
│   ├── babymonitor/
│   │   ├── __init__.py
│   │   ├── camera.py        # Camera handling
│   │   ├── audio_processor.py # Audio processing
│   │   ├── detectors/
│   │   │   ├── __init__.py
│   │   │   ├── base_detector.py
│   │   │   ├── person_detector.py
│   │   │   └── emotion_detector.py
│   │   └── web/
│   │       ├── server.py    # Web server implementation
│   │       ├── static/
│   │       │   ├── css/     # Stylesheets including dark theme
│   │       │   ├── js/      # JavaScript for UI interactions
│   │       │   └── img/     # Images and icons
│   │       └── templates/   # HTML templates for web interface
├── scripts/
│   └── run.py               # Legacy script (for backward compatibility)
├── models/                  # Pre-trained models
├── logs/                    # System logs
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## Development

### Adding New Detectors

New detectors should inherit from the `BaseDetector` class and implement the required methods:

```python
from babymonitor.detectors.base_detector import BaseDetector

class MyDetector(BaseDetector):
    def __init__(self, threshold=0.5):
        super().__init__(threshold=threshold)
        
    def process_frame(self, frame):
        # Process frame and return results
        return {"frame": processed_frame, "detections": []}
```

### Web Interface Customization

The web interface uses Flask and SocketIO. Templates are in `src/babymonitor/web/templates/` and static files in `src/babymonitor/web/static/`.

### Developer Mode Features

When running in developer mode, you have access to:

- **Dev Tools**: Simulate detections, emotions, and clear metrics
- **Logs**: View and filter system logs
- **Settings**: Configure camera, detection, audio, and system settings
- **Metrics**: Detailed system performance metrics with dark theme

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenCV for computer vision capabilities
- Flask for the web framework
- SocketIO for real-time communication
- PyQt5 for the local GUI interface
