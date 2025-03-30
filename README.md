# Baby Monitor System

An advanced baby monitoring system with AI-powered emotion detection, camera management, and web interface.

## Features

- **Real-time Video Monitoring**: Stream video from multiple cameras
- **Emotion Detection**: AI-powered detection of baby emotions (crying, laughing, etc.)
- **Audio Monitoring**: Listen to audio from the baby's room
- **Motion Detection**: Get alerts when motion is detected
- **Emotion History**: Track and visualize baby's emotions over time
- **Multi-camera Support**: Connect and manage multiple cameras
- **Web Interface**: Access the monitoring system from any device with a web browser
- **Raspberry Pi Optimized**: Runs efficiently on Raspberry Pi devices

## Quick Start

1. Ensure you have Python 3.8-3.11 installed
2. Run the installer:

   ```bash
   python setup.py
   ```

3. Start the system:

   ```bash
   python main.py --mode normal
   ```

4. Open your browser and go to: `http://localhost:5000`

For detailed installation instructions, please refer to [INSTALL.md](INSTALL.md).

## Starting the Baby Monitor

The Baby Monitor System supports different operating modes:

### Normal Mode (for regular users)

```bash
python main.py --mode normal
```

This mode provides a simplified interface designed for parents and caregivers, hiding technical details and focusing on essential baby monitoring features.

### Developer Mode

```bash
python main.py --mode dev
```

Developer mode provides access to advanced features, system metrics, and diagnostic tools that aren't needed by regular users.

### Additional Command Line Options

```bash
python main.py [--mode {normal|dev|local}] [OPTIONS]

Options:
  --threshold THRESHOLD   Detection threshold (default: 0.5)
  --camera_id CAMERA_ID   Camera ID (default: 0)
  --input_device DEVICE   Audio input device ID
  --host HOST             Host for web interface (default: 0.0.0.0)
  --port PORT             Port for web interface (default: 5000)
  --debug                 Enable debug mode
```

## Project Structure

```
Baby Monitor System/
├── config/               # Configuration files
├── data/                 # Application data
├── logs/                 # System logs
├── models/               # AI models
├── scripts/              # Utility scripts
│   └── install/          # Installation scripts
├── src/                  # Source code
│   └── babymonitor/      # Main application code
├── main.py               # Main entry point
└── README.md             # This file
```

## System Requirements

- **Python**: 3.8-3.11
- **OS**: Windows, Linux, macOS, or Raspberry Pi OS
- **Memory**: 2GB RAM minimum (4GB recommended)
- **Storage**: 1GB free space minimum
- **Camera**: USB webcam or Raspberry Pi Camera Module
- **Microphone**: Required for audio monitoring

## Configuration

The system can be configured by:

1. Using the configuration UI in repair tools
2. Editing the `.env` file
3. Editing JSON configuration files in the `config` directory

For more details, see the Configuration section in [INSTALL.md](INSTALL.md).

## Scripts and Utilities

The `scripts` directory contains various utilities for:

- Installation and setup
- System maintenance
- Performance monitoring
- Troubleshooting

For more information, see [scripts/README.md](scripts/README.md).

## Development

To set up a development environment:

1. Install the system with developer mode:

   ```bash
   python setup.py --mode dev
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
3. Start the system with debug mode:

   ```bash
   python main.py --mode dev --debug
   ```

## Troubleshooting

For common issues and solutions, please refer to the Troubleshooting section in [INSTALL.md](INSTALL.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- TensorFlow and PyTorch for AI models
- Flask for the web interface
- All contributors to the project

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

The folders named `backup_YYYYMMDD_HHMMSS` (e.g., `backup_20250327_011217`) are automatically generated by the system's backup mechanism. They are created during:

1. System updates
2. Major configuration changes
3. Scheduled automatic backups (daily/weekly)

These folders contain copies of the system's state at the time specified in their name.

### Managing Backups

- **Keep** at least the 3 most recent backups for recovery purposes
- **Delete** older backups by running the cleanup script:

  ```
  python tools/cleanup_backups.py --keep 3
  ```

- **Restore** from a backup if needed:

  ```
  python tools/restore_backup.py --backup backup_20250327_011217
  ```

**Note:** The cleanup script will automatically identify and remove recursively nested backup folders, which can sometimes occur due to system errors.

## Installation

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Access the application at `http://localhost:5000`

## Requirements

- Python 3.6+
- PyAudio
- Flask
- OpenCV
- psutil

## Troubleshooting

If you encounter "Loading..." messages in the Repair Tools interface, ensure:

1. The Flask server is running
2. The API endpoints are properly configured
3. PyAudio is installed for microphone detection
4. OpenCV is installed for camera functions
