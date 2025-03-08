# Baby Monitor System

An intelligent baby monitoring system that combines computer vision and audio processing for comprehensive baby monitoring, featuring emotion detection, person detection, and real-time alerts.

## Project Structure

```
edge_comp/
├── models/                  # Model files and weights
│   ├── yolov8n.pt         # YOLO model for person detection
│   ├── hubert/            # HuBERT model files for emotion detection
│   └── README.md          # Model documentation
├── src/                    # Source code
│   ├── audio/             # Audio processing modules
│   │   └── audio_processor.py
│   ├── camera/            # Camera handling
│   │   └── camera.py
│   ├── detectors/         # Detection modules
│   │   ├── motion_mog2.py
│   │   ├── sound_hubert.py
│   │   └── vision_yolo.py
│   ├── emotion/           # Emotion recognition
│   ├── utils/             # Utility functions
│   │   ├── config.py      # System configuration
│   │   └── system_monitor.py
│   ├── web/              # Web interface
│   ├── ui/               # Desktop UI components
│   └── main.py           # Main application
├── training/             # Training scripts and notebooks
│   └── speechbrain-finetune.ipynb
├── config/              # Configuration files
├── data/               # Data files
├── tests/             # Test files
├── requirements.txt    # Python dependencies
├── setup.py           # Package setup
└── .env              # Environment variables (create from template)
```

## Features

### ✅ Currently Implemented

- **Video Monitoring**
  - Real-time video capture and streaming
  - Support for multiple camera resolutions
  - Automatic camera detection and selection

- **Person Detection**
  - YOLOv8 nano model integration
  - Real-time person tracking
  - Fall detection capabilities

- **Audio Processing**
  - Real-time audio capture
  - Waveform visualization
  - Multiple audio input device support

- **Emotion Detection**
  - HuBERT-based emotion recognition
  - Support for multiple emotion models
  - Real-time emotion classification

- **User Interface**
  - Modern desktop UI with Tkinter
  - Web-based monitoring interface
  - Real-time status updates
  - Configurable settings panel

### 🚧 Under Development

- Enhanced motion analysis
- Multi-camera support
- Mobile application interface
- Environmental monitoring
- Custom emotion model training

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- Microphone
- Git (for version control)

### Quick Start

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd edge_comp
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -e .
   ```

4. **Run System Setup**

   ```bash
   # Run the setup script to configure the system
   python src/setup.py
   
   # This will:
   # - Check system requirements
   # - Install/update required packages
   # - Configure camera and audio devices
   # - Create necessary directories
   # - Generate system configuration file
   ```

5. **Start the Application**

   ```bash
   # Start the web interface
   python src/web/web_app.py
   
   # Or start the main application
   python src/main.py
   
   # For development mode with additional logging and visualizations
   python src/main.py --dev
   ```

### Running Individual Components

1. **System Setup and Configuration**

   ```bash
   python src/setup.py
   ```

   - Configures system and hardware
   - Creates required directories
   - Generates `config/system_config.json`
   - Shows recommended model based on hardware

2. **Web Interface Only**

   ```bash
   python src/web/web_app.py
   ```

   - Access at `http://localhost:5000`
   - Real-time monitoring interface
   - Camera and audio controls

3. **Main Application**

   ```bash
   # Normal mode
   python src/main.py
   
   # Development mode
   python src/main.py --dev
   ```

   - Full application with all features
   - Emotion detection and monitoring
   - System resource tracking

### Configuration Files

After running `setup.py`, the following files will be created:

- `config/system_config.json`: System configuration and hardware settings
- `logs/baby_monitor.log`: Application logs
- `models/`: Downloaded model files
- `data/`: Application data directory

## Usage

### Desktop Application

1. **Start the Application**

   ```bash
   python src/main.py
   ```

2. **Camera Controls**
   - Select camera from dropdown
   - Choose resolution
   - Toggle camera feed

3. **Audio Controls**
   - Enable/disable audio monitoring
   - Select emotion detection model
   - View real-time waveform

### Web Interface

Access the monitoring interface at:

- Local: `http://localhost:5000`
- Network: `http://<device-ip>:5000`

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Check style
flake8 src/
```

## Troubleshooting

### Common Issues

1. **Camera Not Found**
   - Check camera connection
   - Verify camera index in `.env`
   - Test camera in other applications

2. **Audio Issues**
   - Check microphone connection
   - Verify audio input device
   - Check system permissions

3. **Model Loading Errors**
   - Check internet connection for first run
   - Verify model files in `models/` directory
   - Check CUDA configuration if using GPU

### Logs

- Application logs: `baby_monitor.log`
- Check logs for detailed error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- SpeechBrain project
- OpenCV community
- Flask and Socket.IO teams
