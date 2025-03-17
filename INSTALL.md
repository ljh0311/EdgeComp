# Baby Monitor System - Installation Guide

This document provides instructions for installing the Baby Monitor System on different platforms.

## Quick Start

For most users, the simplest way to install is to run:

```bash
python install.py
```

This will launch the graphical installer, which will guide you through the installation process.

## Installation Options

The installer supports several options:

- `--no-gui`: Run installation without the graphical interface
- `--skip-models`: Skip downloading detection models
- `--skip-shortcut`: Skip creating desktop shortcut

Example:
```bash
python install.py --no-gui
```

## Platform-Specific Installation

### Windows

1. Ensure you have Python 3.8 or newer installed
2. Run the installer:
   ```
   python install.py
   ```
3. Follow the on-screen instructions

### Linux

1. Ensure you have Python 3.8 or newer installed
2. Run the installer:
   ```
   python install.py
   ```
3. Follow the on-screen instructions

### Raspberry Pi

For Raspberry Pi devices, you have two options:

#### Option 1: Graphical Installer
```
python install.py
```

#### Option 2: Command-Line Installer
```
bash install_pi.sh
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
   python setup.py --no-gui
   ```

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
2. Ensure you have Python 3.8 or newer installed
3. Try running the installer with administrator/sudo privileges
4. Check the logs in the `logs` directory for more information

## After Installation

After installation, you can start the Baby Monitor System using:

1. The desktop shortcut (if created during installation)
2. Command line: `python main.py`

For more information, please refer to the README.md file. 