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
   pip install -r requirements.txt
   ```

4. **Environment Setup**
   ```bash
   # Copy template and edit as needed
   cp .env.template .env
   ```

5. **Run the Application**
   ```bash
   # Normal mode
   python src/main.py
   
   # Developer mode (with additional visualizations)
   python src/main.py --dev
   ```

### Model Setup

The required models will be downloaded automatically when you first run the application:

- **YOLOv8n**: Person detection model
- **HuBERT**: Emotion recognition model

You can also manually download them by following instructions in `models/README.md`.

## Configuration

### Environment Variables

Create a `.env` file with the following options:

```env
CAMERA_INDEX=0           # Camera device index
USE_CUDA=0              # Enable CUDA (1) or CPU (0)
LOG_LEVEL=INFO          # Logging level
```

### System Configuration

Main configuration settings are in `src/utils/config.py`:

- Camera settings (resolution, FPS)
- Model paths and parameters
- Audio processing settings
- Detection thresholds
- Performance options

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
