# Baby Monitor System

A comprehensive baby monitoring solution with video, audio, and AI-based detection capabilities.

## Features

- Real-time video monitoring with person detection
- Audio monitoring with emotion detection
- User-friendly GUI interface
- Comprehensive metrics and statistics
- Developer tools for testing and diagnostics

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV
- PyQt5
- PyTorch
- Transformers
- SoundDevice
- Librosa

### Setup

1. Clone this repository:
```
git clone https://github.com/yourusername/babymonitor.git
cd babymonitor
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac
```

3. Install required packages:
```
pip install -r requirements.txt
```

4. Set up models using one of these methods:
   - Use our convenient model manager scripts:
     ```
     # Windows
     model_manager.bat
     
     # Linux/macOS
     ./model_manager.sh
     ```
   - Or download/train models directly:
     ```
     # Download pretrained models (recommended)
     python setup.py --download
     
     # Train models (requires significant computational resources)
     python setup.py --train
     ```

## Usage

### Basic Usage

To launch the baby monitor with the default GUI interface in normal mode:

```
python -m babymonitor.launcher
```

### Developer Mode

To run the baby monitor in developer mode with additional testing tools:

```
python -m babymonitor.launcher --mode dev
```

This enables:
- Model testing (processing of user-selected data/microphone input)
- Microphone testing (recording and processing live microphone input)
- Camera testing
- System diagnostics
- Performance benchmarks

### Command Line Options

The launcher supports various command-line options:

```
python -m babymonitor.launcher --help
```

Main options:
- `--interface`, `-i`: Choose between GUI (PyQt) or web interface
- `--mode`, `-m`: Select operation mode (normal or dev)
- `--camera`, `-c`: Specify camera device index
- `--audio-device`, `-a`: Specify audio device index
- `--debug`, `-d`: Enable debug logging
- `--person-model`: Select person detection model
- `--emotion-model`: Select emotion detection model

Example:
```
python -m babymonitor.launcher -i gui -m dev -c 0 -d
```

## GUI Components

### Dashboard

The main dashboard displays:
- Live video feed with person detection
- System status information
- Emotion detection results
- Quick actions

### Metrics

The metrics tab shows:
- Performance statistics
- Detection counts
- Emotion distribution
- System resource usage

### Emotion Detection

The emotion detection tab provides:
- Real-time audio analysis
- Emotion visualization
- Recording and playback capabilities
- Audio file analysis

### Developer Tools (Dev Mode)

The developer tools tab (available in dev mode) includes:
- Model testing tools
- Microphone testing
- Camera diagnostics
- System performance benchmarks

## Model Management

The Baby Monitor System relies on AI models for its operation. Our enhanced model management tools make it easy to check, download, and train these models.

### Using the Model Manager

We provide dedicated scripts for convenient model management:

#### Windows:
```
model_manager.bat
```

#### Linux/macOS:
```
./model_manager.sh
```

These scripts allow you to:
1. Check the status of required models
2. Download pretrained models (recommended)
3. Train specific models from scratch
4. Train all models at once

### Command-Line Model Management

You can also manage models directly using command-line arguments with `setup.py`:

```
# Check model status
python setup.py --models

# Download missing models
python setup.py --download

# Download a specific model
python setup.py --download --specific-model emotion_model

# Train models
python setup.py --train

# Train a specific model
python setup.py --train --specific-model wav2vec2_model
```

### Required Models

The system requires these AI models:
- **Person Detection**: Uses YOLOv8n for detecting people in video feed
- **Emotion Detection**: Uses audio analysis to detect emotions like crying, laughing
- **Speech Recognition**: Uses wav2vec2 models for more nuanced audio analysis

## Troubleshooting

### Camera Issues

If the camera doesn't work properly:
1. Make sure no other application is using the camera
2. Try a different camera index with `-c` option
3. Check the camera connections
4. Update your camera drivers

### Audio Issues

If audio detection doesn't work properly:
1. Verify your microphone is properly connected and working
2. Use the `-a` option to select a different audio device
3. Run microphone testing in developer mode
4. Check system audio settings

### Model Issues

If you're having issues with the AI models:
1. Verify models are properly installed with `python setup.py --models`
2. Try downloading pre-trained models with `python setup.py --download`
3. Use the model manager script for simplified management
4. Check your Python version is 3.8 or higher
5. Make sure you have adequate disk space (~500MB for all models)

### Performance Problems

If the application runs slowly:
1. Close other resource-intensive applications
2. Try a lighter person detection model (`--person-model yolov8n`)
3. Run in developer mode and use the diagnostics tools

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 for object detection
- Wav2Vec2 for audio emotion detection
- PyQt5 for the GUI framework

## New Features

### Microphone Selection
The application now supports selecting different microphones for sound/emotion detection. Here's how to use this feature:

1. Go to the "Repair Tools" page
2. In the "Sound Settings" section, you'll find a dropdown menu labeled "Microphone" or "Audio Input Device"
3. Select your preferred microphone from the list
4. Click the "Apply Microphone Selection" button to use the selected microphone

### Sound Quality Improvement Tips
To improve sound quality for better emotion recognition:

1. Position the microphone closer to the sound source
2. Reduce background noise in the environment
3. Use a higher quality microphone if available
4. Ensure the microphone is not obstructed or covered
5. Adjust system volume levels if detection is too sensitive

## API Endpoints

The application now provides the following API endpoints:

### Emotion Model Endpoints
- `GET /api/emotion/models` - Get available emotion models
- `GET /api/emotion/model/<model_id>` - Get info about a specific model
- `POST /api/emotion/switch_model` - Switch to a different model
- `POST /api/emotion/test_audio` - Test audio system
- `POST /api/emotion/restart_audio` - Restart audio system

### Audio Microphone Endpoints
- `GET /api/audio/microphones` - Get available microphones
- `GET /api/audio/microphones/<microphone_id>` - Get info about a specific microphone
- `POST /api/audio/set_microphone` - Set the active microphone

### System Endpoints
- `GET /api/system/check` - Check system status
- `GET /api/system/info` - Get detailed system information
- `POST /api/system/restart` - Restart the system
- `POST /api/system/stop` - Stop the system

## About Backup Folders

The folders named `backup_YYYYMMDD_HHMMSS` (e.g., `backup_20250327_011217`) are automatically generated by the system's backup mechanism. They contain copies of the system's state at the time specified in their name.

### Managing Backups

- **Keep** at least the 3 most recent backups for recovery purposes
- **Delete** older backups automatically using the cleanup option
- **Restore** from a backup if you encounter issues

## Utility Scripts

The Baby Monitor System includes several utility scripts to help manage and maintain your installation.

### Backup and Restore Tool

The system includes a comprehensive backup and restore utility:

```
tools\backup_restore.bat    # Windows
# OR
bash tools/backup/restore.sh  # Linux/MacOS
```

Options:
1. **List backup folders** - View all available backups
2. **Create a new backup** - Create a timestamped backup of your current system
3. **Restore from backup** - Restore your system to a previous state
4. **Clean up old backups** - Remove old backups, keeping only the most recent ones

### Platform-Specific Tools

#### Raspberry Pi Optimization

If you're running the Baby Monitor on a Raspberry Pi, you can optimize its performance:

```bash
# Run as root for full optimizations
sudo bash tools/system/optimize_raspberry_pi.sh
```

This script:
- Sets the CPU governor to performance mode
- Optimizes GPU memory allocation
- Improves swap configuration

### Testing Tools

#### API Testing

Test the Baby Monitor's API endpoints:

```
python tests/updated/api_test.py
```

Add additional parameters to test a different host or port:
```
python tests/updated/api_test.py [host] [port]
```

### Directory Structure

The utility scripts have been organized into the following directory structure:

```
baby_monitor_system/
├── tools/
│   ├── backup/                # Backup and restore utilities
│   │   ├── list_backups.py    # Script to list available backups
│   │   ├── create_backup.py   # Script to create a new backup
│   │   ├── restore_backup.py  # Script to restore from a backup
│   │   └── cleanup_backups.py # Script to clean up old backups
│   └── system/                # System utilities
│       └── optimize_raspberry_pi.sh # Performance optimizations for Raspberry Pi
├── tests/
│   └── updated/               # Updated test scripts
│       └── api_test.py        # API testing utility
├── model_manager.bat          # Windows model management utility
└── model_manager.sh           # Linux/macOS model management utility
```

## Installation

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Set up required models: `model_manager.bat` or `./model_manager.sh`
4. Run the application: `python app.py`
5. Access the application at `http://localhost:5000`

## Requirements

- Python 3.8+
- PyAudio
- Flask
- OpenCV
- PyTorch
- psutil

## Troubleshooting

If you encounter "Loading..." messages in the Repair Tools interface, ensure:

1. The Flask server is running
2. The API endpoints are properly configured
3. PyAudio is installed for microphone detection
4. OpenCV is installed for camera functions
5. Required models are downloaded/trained properly
