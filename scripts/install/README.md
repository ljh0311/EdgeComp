# Baby Monitor System - Installation Scripts

This directory contains the installation and setup scripts for the Baby Monitor System.

## Available Scripts

### Main Installer Scripts

- `install.py` - Main Python installation script with GUI interface
- `setup.py` - Core setup functionality that handles all installation tasks

### Platform-Specific Installers

- `install.bat` - Windows-specific installation wrapper
- `install.sh` - Linux/Mac-specific installation wrapper
- `install_pi.sh` - Raspberry Pi optimized installation script

## Installation Methods

### 1. Graphical Installation (Recommended)

Most users should use the graphical installer from the project root:

```bash
python scripts/install/install.py
```

Or if you're already in the install directory:

```bash
python install.py
```

This provides a user-friendly interface with:

- Selection of components to install
- Configuration options for camera, web interface, etc.
- Operation mode selection (Normal/Developer)
- Progress tracking

### 2. Command-Line Installation

For automated or headless installations:

```bash
python scripts/install/install.py --no-gui
```

Options:

- `--no-gui`: Run in command-line mode
- `--skip-models`: Skip downloading detection models
- `--skip-shortcut`: Skip creating desktop shortcut
- `--mode [normal|dev]`: Set operation mode

### 3. Platform-Specific Installations

#### Windows

```
scripts\install\install.bat
```

#### Linux/Mac

```
bash scripts/install/install.sh
```

#### Raspberry Pi

```
bash scripts/install/install_pi.sh
```

## Setup Process

The installation process includes:

1. **Environment Verification**
   - Checking Python version (3.8-3.11 supported)
   - Creating necessary directories

2. **Package Installation**
   - Installing required Python packages with correct versions
   - Platform-specific optimizations

3. **Model Setup**
   - Downloading person detection models (YOLO, Haar Cascades)
   - Setting up emotion recognition models

4. **Camera Management**
   - Detecting available cameras
   - Creating camera configuration

5. **Emotion History**
   - Setting up emotion history logging
   - Creating initial configuration

6. **Configuration**
   - Creating configuration files for cameras, audio, detection, emotions
   - Setting up environment variables

7. **Shortcut Creation**
   - Creating desktop shortcuts for easy access

## Configuration Files

The setup process creates the following configuration files:

- `.env` - Environment configuration
- `config/camera.json` - Camera configuration
- `config/audio.json` - Audio device configuration
- `config/detection.json` - Detection settings
- `config/emotion.json` - Emotion recognition configuration

## Directory Structure

The setup creates the following directory structure:

```
Baby Monitor System/
├── config/               # Configuration files
├── data/                 # Application data
│   └── cameras/          # Camera management data
├── logs/                 # System logs
├── models/               # AI models
│   ├── person/           # Person detection models
│   └── emotion/          # Emotion recognition models
├── venv/                 # Python virtual environment
```

## Troubleshooting

If you encounter issues during installation:

1. Check that you have Python 3.8-3.11 installed
2. Check the logs in the `logs` directory
3. Try running with the `--mode dev` option for more verbose output
4. For camera issues, use `--camera_id 0` or other numbers to try different cameras

## For Developers

To extend the installer:

- Add new dependencies in `setup.py` in the `get_package_versions()` function
- Add new configuration options in the `ConfigurationPage` class
- Implement new setup steps in the `setup_` functions
