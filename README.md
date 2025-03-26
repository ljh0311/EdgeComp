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
   python scripts/install/install.py
   ```
3. Start the system:
   ```bash
   python -m src.run_server
   ```
4. Open your browser and go to: `http://localhost:5000`

For detailed installation instructions, please refer to [INSTALL.md](INSTALL.md).

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
   python scripts/install/install.py --mode dev
   ```
2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
3. Start the system with debug mode:
   ```bash
   python -m src.run_server --mode dev --debug
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
