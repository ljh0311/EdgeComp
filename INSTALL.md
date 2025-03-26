# Baby Monitor System - Installation Guide

This document provides instructions for installing the Baby Monitor System on different platforms.

## Quick Start

For most users, the simplest way to install is to run:

```bash
python scripts/install/install.py
```

This will launch the graphical installer, which will guide you through the installation process.

## Installation Options

The installer supports several options:

- `--no-gui`: Run installation without the graphical interface
- `--skip-models`: Skip downloading detection models
- `--skip-shortcut`: Skip creating desktop shortcut
- `--mode [normal|dev]`: Set the operation mode (normal for regular users, dev for developers)

Example:
```bash
python scripts/install/install.py --no-gui --mode dev
```

## Platform-Specific Installation

### Windows

1. Ensure you have Python 3.8-3.11 installed
2. Clone or download this repository
3. Run the installer:
   ```
   python scripts/install/install.py
   ```
   Or use the Windows-specific batch file:
   ```
   scripts\install\install.bat
   ```
4. Follow the on-screen instructions

### Linux

1. Ensure you have Python 3.8-3.11 installed
2. Clone or download this repository
3. Run the installer:
   ```
   python scripts/install/install.py
   ```
   Or use the Linux-specific shell script:
   ```
   bash scripts/install/install.sh
   ```
4. Follow the on-screen instructions

### Raspberry Pi

For Raspberry Pi devices, you have two options:

#### Option 1: Graphical Installer
```
python scripts/install/install.py
```

#### Option 2: Command-Line Installer
```
bash scripts/install/install_pi.sh
```

This script will:
- Install required system dependencies
- Set up the camera module
- Create a Python virtual environment
- Install Python dependencies optimized for Raspberry Pi
- Download detection models
- Create configuration files
- Create a desktop shortcut

## Manual Installation

If you prefer to install manually, follow these steps:

1. Create a Python virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the setup script:
   ```
   python scripts/install/setup.py --no-gui
   ```

## Configuration Options

The Baby Monitor System supports the following configuration options:

### Operation Modes
- **Normal Mode**: Simplified interface for regular users
- **Developer Mode**: Advanced features and debugging options

### Camera Configuration
- Camera ID: Select from available cameras (0, 1, 2, etc.)
- Resolution: Choose from 640x480, 1280x720, 1920x1080
- FPS: Camera frame rate (default: 30)

### Web Interface
- Port: Web interface port (default: 5000)
- Host: Web interface host (default: 0.0.0.0)

### Detection
- Person threshold: Detection sensitivity (0.3-0.7)
- Emotion threshold: Emotion detection sensitivity (0.3-0.7)

### Emotion Models
- Basic Emotion: Default emotion model
- Enhanced Emotion: More detailed emotion classification
- SpeechBrain Emotion: Audio-based emotion detection
- Cry Detection: Specialized for detecting baby crying

## Troubleshooting

### Camera Issues

If you encounter camera access issues, try:

```
python main.py --camera_id 1
```

Different camera IDs (0, 1, 2, etc.) can be tried if your default camera is not working.

### Audio Device Issues

List available audio devices:

```
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Then specify a device:

```
python main.py --input_device 1
```

### Installation Fails

If installation fails, try:

1. Check that you have the required permissions
2. Ensure you have Python 3.8 or newer (but not newer than 3.11) installed
3. Try running the installer with administrator/sudo privileges
4. Check the logs in the `logs` directory for more information

## After Installation

After installation, you can start the Baby Monitor System using:

1. The desktop shortcut (if created during installation)
2. Command line:
   - Windows: `venv\Scripts\python -m src.run_server --mode normal`
   - Linux/Mac: `venv/bin/python -m src.run_server --mode normal`

For more detailed information, please refer to the README.md file.

## Directory Structure

After installation, the system will have the following structure:

```
Baby Monitor System/
├── config/               # Configuration files
├── data/                 # Application data
│   └── cameras/          # Camera management data
├── logs/                 # System logs
├── models/               # AI models
│   ├── person/           # Person detection models
│   └── emotion/          # Emotion recognition models
│       ├── speechbrain/  # SpeechBrain models
│       ├── emotion2/     # Enhanced emotion models
│       └── cry_detection/# Cry detection models
├── scripts/              # Utility scripts
│   └── install/          # Installation scripts
├── src/                  # Source code
│   └── babymonitor/      # Main application code
├── venv/                 # Python virtual environment
├── .env                  # Environment configuration
├── main.py               # Main entry point
└── README.md             # Project documentation
```

For questions or issues, please refer to the project documentation or open an issue on the project repository. 