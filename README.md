# Baby Monitor System

A comprehensive baby monitoring solution with real-time person detection, emotion recognition, and web interface.

![Baby Monitor System](docs/images/screenshot.png)

## Features

- **Person Detection**: Detects people using YOLOv8 object detection model
- **Emotion Recognition**: Identifies baby emotions (crying, laughing, babbling, silence) from audio
- **Web Interface**: Real-time monitoring dashboard with alerts and statistics
- **Multi-threading**: Efficient processing of video and audio streams
- **Alerting System**: Notifications when crying is detected or when a person enters/leaves the frame
- **Multiple Launch Modes**: Normal, Developer, and Local GUI modes for different use cases
- **Dark Theme UI**: Modern dark-themed interface with responsive design

## Requirements

- Python 3.8-3.12
- OpenCV
- NumPy
- SoundDevice
- Flask
- SocketIO
- PyQt5 (for local GUI mode)

## Setup Instructions

### Windows Setup

1. **Prerequisites**:
   - Install [Python 3.8-3.12](https://www.python.org/downloads/windows/)
   - Install [Git for Windows](https://gitforwindows.org/)
   - Install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

2. **Clone the repository**:

   ```powershell
   git clone https://github.com/yourusername/baby-monitor-system.git
   cd baby-monitor-system
   ```

3. **Run the Windows installer**:

   ```powershell
   .\scripts\install\install.bat
   ```

4. **If you encounter any issues, run the fix script**:

   ```powershell
   .\scripts\fix_windows.bat
   ```

### macOS Setup

1. **Prerequisites**:
   - Install [Python 3.8-3.12](https://www.python.org/downloads/macos/)
   - Install [Homebrew](https://brew.sh/)
   - Install required system libraries:

     ```bash
     brew install portaudio
     ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/baby-monitor-system.git
   cd baby-monitor-system
   ```

3. **Run the installer**:

   ```bash
   bash scripts/install/install.sh
   ```

### Linux/Raspberry Pi Setup

1. **Prerequisites**:
   - Install required system libraries:

     ```bash
     sudo apt update
     sudo apt install -y python3-pip python3-venv portaudio19-dev libopencv-dev
     ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/baby-monitor-system.git
   cd baby-monitor-system
   ```

3. **Run the installer**:
   - For regular Linux:

     ```bash
     bash scripts/install/install.sh
     ```

   - For Raspberry Pi:

     ```bash
     bash scripts/install/install_pi.sh
     ```

### GUI Installer (All Platforms)

For a guided installation with a graphical interface:

1. Make sure you have PyQt5 installed:

   ```bash
   pip install PyQt5
   ```

2. Run the GUI installer:

   ```bash
   # Windows
   python scripts\install\install.py
   
   # macOS/Linux
   python scripts/install/install.py
   ```

## Launch Instructions

### Windows

1. **Start the Baby Monitor**:
   - Using the desktop shortcut created during installation, or
   - Using the command line:

     ```powershell
     cd C:\path\to\baby-monitor-system
     .\start.bat
     ```

2. **For development mode**:

   ```powershell
   cd C:\path\to\baby-monitor-system
   .\start.bat --mode dev
   ```

### macOS/Linux

1. **Start the Baby Monitor**:

   ```bash
   cd /path/to/baby-monitor-system
   bash start.sh
   ```

2. **For development mode**:

   ```bash
   cd /path/to/baby-monitor-system
   bash start.sh --mode dev
   ```

### Raspberry Pi

1. **Start the Baby Monitor**:

   ```bash
   cd /path/to/baby-monitor-system
   bash start_pi.sh
   ```

## Launch Modes

The Baby Monitor System supports three launch modes:

### Normal Mode (Default)

Standard mode for end users, showing the main dashboard with camera feed and metrics, but no access to development tools.

```bash
# Windows
.\start.bat --mode normal

# macOS/Linux
bash start.sh --mode normal
```

### Developer Mode

Shows metrics page with access to all development tools, logs, and settings. Use this mode for debugging and development.

```bash
# Windows
.\start.bat --mode dev

# macOS/Linux
bash start.sh --mode dev
```

### Local GUI Mode

Runs the local GUI version of the baby monitor using PyQt5, without the web interface.

```bash
# Windows
.\start.bat --mode local

# macOS/Linux
bash start.sh --mode local
```

## Command Line Arguments

- `--mode`: Launch mode (`normal`, `dev`, `local`)
- `--threshold`: Detection threshold (default: 0.5)
- `--camera_id`: Camera ID (default: 0)
- `--input_device`: Audio input device ID (default: None)
- `--host`: Web interface host (default: 0.0.0.0)
- `--port`: Web interface port (default: 5000)
- `--debug`: Enable debug mode

## Troubleshooting

### Common Issues

#### Camera Issues

If you encounter camera access issues:

1. **Try a different camera ID**:

   ```bash
   # Windows
   .\start.bat --camera_id 1
   
   # macOS/Linux
   bash start.sh --camera_id 1
   ```

2. **Check camera permissions**:
   - On Windows: Ensure the app has permission in Privacy settings
   - On macOS: Check System Preferences > Security & Privacy > Camera
   - On Linux: Ensure the user has proper permissions to access the camera device

3. **Run the fix script**:

   ```bash
   # Windows
   .\scripts\fix_windows.bat
   
   # macOS/Linux
   bash scripts/fix_raspberry.sh
   ```

#### Audio Device Issues

1. **List available audio devices**:

   ```bash
   # Windows
   python -c "import sounddevice as sd; print(sd.query_devices())"
   
   # macOS/Linux
   python3 -c "import sounddevice as sd; print(sd.query_devices())"
   ```

2. **Specify a specific device**:

   ```bash
   # Windows
   .\start.bat --input_device 1
   
   # macOS/Linux
   bash start.sh --input_device 1
   ```

#### Python Version Issues

The Baby Monitor System supports Python 3.8 through 3.12. If you have multiple Python versions installed:

1. **Force a specific Python version in Windows**:

   ```powershell
   py -3.11 scripts\install\install.py
   ```

2. **Force a specific Python version in macOS/Linux**:

   ```bash
   python3.11 scripts/install/install.py
   ```

### Web Interface Issues

1. **Default Access**:
   - The web interface is available at <http://localhost:5000> by default

2. **Change port if blocked**:

   ```bash
   # Windows
   .\start.bat --port 8080
   
   # macOS/Linux
   bash start.sh --port 8080
   ```

3. **Access from other devices**:
   - Use the machine's IP address: <http://192.168.1.x:5000> (replace with your actual IP)
   - Make sure firewall allows connections to the port

## Maintenance Scripts

Various maintenance and utility scripts are available:

- **Windows Fix Utility**: `scripts/fix_windows.bat`
- **Raspberry Pi Fix Utility**: `scripts/fix_raspberry.sh`
- **Metrics Fix Utility**: `scripts/fix_metrics.bat`
- **Restart Monitor**: `scripts/restart_baby_monitor.bat`

## Updates and Fixes

### Recent Updates

1. **Installation Scripts Organization**:
   - Reorganized all installation scripts to `scripts/install` directory
   - Added GUI installer with PyQt5 interface

2. **Enhanced Person Detection**:
   - Upgraded to YOLOv8 for faster and more accurate person detection
   - Improved tracking stability with multi-frame history

3. **Emotion Detection Improvements**:
   - Better crying detection accuracy
   - Added percentage-based emotion distribution display
   - Real-time emotion event logging

4. **Metrics Dashboard**:
   - Redesigned dark theme interface
   - Added real-time charts for performance monitoring
   - Improved detection event logging
   - Added export functionality for alerts

5. **Multi-platform Support**:
   - Added compatibility for Python 3.8-3.12
   - Enhanced error handling for different platforms
   - Improved setup scripts for Windows, macOS, and Raspberry Pi

## Project Structure

```
baby-monitor-system/
├── main.py                    # Main entry point with multiple launch modes
├── start.bat                  # Windows startup script
├── start.sh                   # macOS/Linux startup script
├── src/
│   ├── babymonitor/
│   │   ├── __init__.py
│   │   ├── camera.py          # Camera handling
│   │   ├── audio_processor.py # Audio processing
│   │   ├── detectors/
│   │   │   ├── __init__.py
│   │   │   ├── base_detector.py
│   │   │   ├── person_detector.py
│   │   │   └── emotion_detector.py
│   │   └── web/
│   │       ├── server.py      # Web server implementation
│   │       ├── static/
│   │       │   ├── css/       # Stylesheets including dark theme
│   │       │   ├── js/        # JavaScript for UI interactions
│   │       │   └── img/       # Images and icons
│   │       └── templates/     # HTML templates for web interface
├── scripts/
│   ├── install/
│   │   ├── install.bat        # Windows installer
│   │   ├── install.sh         # macOS/Linux installer
│   │   ├── install_pi.sh      # Raspberry Pi installer
│   │   ├── install.py         # GUI installer
│   │   └── setup.py           # Core setup script
│   ├── fix_windows.bat        # Windows fix script
│   ├── fix_raspberry.sh       # Raspberry Pi fix script
│   ├── fix_metrics.bat        # Metrics fix script
│   └── restart_baby_monitor.bat # Service restart script
├── models/                    # Pre-trained models
├── logs/                      # System logs
├── config/                    # Configuration files
├── tests/                     # Unit tests
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenCV for computer vision capabilities
- YOLOv8 for object detection
- Flask for the web framework
- SocketIO for real-time communication
- PyQt5 for the local GUI interface

## Start Scripts

The Baby Monitor System includes several enhanced start scripts for different platforms:

### Windows

- **start.bat**: The main startup script for Windows. It supports all command-line arguments, displays helpful status information, and provides proper error handling. This replaces the older `start_monitor.bat` which was a simpler script with hardcoded parameters.

  Usage examples:

  ```powershell
  # Simple startup (uses default camera ID 0 and normal mode)
  .\start.bat
  
  # Developer mode with camera ID 1
  .\start.bat --mode dev --camera_id 1
  
  # Full options example
  .\start.bat --mode dev --camera_id 0 --port 8080 --host 127.0.0.1 --debug
  ```

### macOS/Linux

- **start.sh**: The main startup script for macOS and Linux platforms. It supports all command-line arguments and has similar functionality to the Windows script.

  Usage examples:

  ```bash
  # Make the script executable
  chmod +x start.sh
  
  # Simple startup
  ./start.sh
  
  # Developer mode with camera ID 1
  ./start.sh --mode dev --camera_id 1
  ```

### Raspberry Pi

- **start_pi.sh**: A specialized startup script for Raspberry Pi with performance optimizations. It includes options to manage CPU governor settings, GPU memory allocation, and automatically reduces resolution for better performance.

  Usage examples:

  ```bash
  # Make the script executable
  chmod +x start_pi.sh
  
  # Start with optimizations enabled (default)
  ./start_pi.sh
  
  # Start without optimizations
  ./start_pi.sh --no-optimize
  
  # Start in developer mode
  ./start_pi.sh --mode dev
  ```

> **Note about start_monitor.bat**: The original `start_monitor.bat` was a simple script that started the monitor in developer mode with hardcoded parameters. The new `start.bat` script provides the same functionality but with much more flexibility, proper command-line argument handling, and better user feedback.
