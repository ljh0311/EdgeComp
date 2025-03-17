# Baby Monitor System

An intelligent baby monitoring system that combines computer vision and audio processing for comprehensive baby monitoring, featuring emotion detection, person detection, and real-time alerts.

## Project Structure

```
edge_comp/
├── models/                  # Model files and weights
│   ├── yolov8n.pt         # YOLO model for person detection
│   ├── emotion_model.pt   # Custom emotion detection model
│   └── audio_model.pt     # Custom audio processing model
├── src/                    # Source code
│   ├── audio/             # Audio processing modules
│   ├── camera/            # Camera handling
│   ├── detectors/         # Detection modules
│   ├── emotion/           # Emotion recognition
│   ├── utils/             # Utility functions
│   ├── web/              # Web interface
│   ├── ui/               # Desktop UI components
│   └── main.py           # Main application
├── training/             # Training scripts and notebooks
├── config/              # Configuration files
├── data/               # Data files
├── tests/             # Test files
├── requirements.txt    # Python dependencies
├── setup.py           # Package setup
└── .env              # Environment variables (create from template)
```

## Installation

### Prerequisites

1. **System Requirements**
   - Python 3.8 or higher
   - Webcam
   - Microphone
   - Git
   - CUDA-capable GPU (optional, for better performance)

2. **Required Software**
   - Python 3.8+: Download from [python.org](https://www.python.org/downloads/)
   - Git: Download from [git-scm.com](https://git-scm.com/downloads)
   - Visual Studio Build Tools (Windows only): Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd edge_comp
   ```

2. **Run Setup Script**
   ```bash
   python setup.py
   ```
   This will automatically:
   - Create a virtual environment
   - Install required dependencies
   - Download available models (YOLOv8)
   - Create necessary directories
   - Setup basic configuration

3. **Manual Model Setup**
   Some models need to be downloaded/setup manually:
   
   a) **Emotion Detection Model (emotion_model.pt)**
      - Download from our shared drive: [link-to-shared-drive]
      - Place in the `models/` directory
   
   b) **Audio Processing Model (audio_model.pt)**
      - Download from our shared drive: [link-to-shared-drive]
      - Place in the `models/` directory

4. **Environment Configuration**
   Create a `.env` file in the project root:
   ```
   CAMERA_INDEX=0
   AUDIO_DEVICE_INDEX=0
   GPU_ENABLED=true
   MODEL_PATH=models/
   LOG_LEVEL=INFO
   ```

### Running the Application

1. **Activate Virtual Environment**
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Start the Application**
   ```bash
   # Run the Baby Monitor Example (uses lightweight detector by default)
   python src/examples/baby_monitor_example.py
   
   # Use YOLOv8 detector instead
   python src/examples/baby_monitor_example.py --detector yolov8
   
   # Adjust camera and resolution
   python src/examples/baby_monitor_example.py --camera 0 --resolution 640x480 --threads 4
   ```

3. **Access the Web Interface**
   - Open a browser and navigate to: http://localhost:5000
   - You can switch between detectors in the web interface

## Detector Options

### Lightweight Detector (Default)
- Optimized for CPU usage
- Ideal for resource-constrained devices like Raspberry Pi
- Lower resource usage but still good accuracy
- Uses TensorFlow Lite for efficient inference

### YOLOv8 Detector
- Higher accuracy person detection
- Requires more computational resources
- Better for desktop/laptop with decent CPU or GPU
- Supports GPU acceleration if available

## Component Testing

### Testing Camera
```bash
python src/camera/test_camera.py
```
- Verifies camera connection
- Tests different resolutions
- Checks frame rate

### Testing Audio
```bash
python src/audio/test_audio.py
```
- Verifies microphone connection
- Tests audio capture
- Checks sampling rate

### Testing Models
```bash
python src/detectors/test_models.py
```
- Verifies model loading
- Tests inference speed
- Validates detection accuracy

## Troubleshooting

### Common Issues and Solutions

1. **Model Loading Errors**
   - Ensure all model files are in the `models/` directory
   - Check file permissions
   - Verify CUDA installation if using GPU

2. **Camera Issues**
   - Update `CAMERA_INDEX` in `.env` file
   - Check USB connection
   - Verify camera permissions

3. **Audio Issues**
   - Update `AUDIO_DEVICE_INDEX` in `.env` file
   - Check microphone connection
   - Verify audio permissions

4. **CUDA/GPU Issues**
   - Install CUDA Toolkit (version 11.0+)
   - Install cuDNN
   - Set `GPU_ENABLED=false` in `.env` for CPU fallback

### Log Files
- Application logs: `logs/baby_monitor.log`
- Error logs: `logs/error.log`
- Performance logs: `logs/performance.log`

## Development

### Code Style
```bash
# Format code
black src/

# Check style
flake8 src/

# Run tests
pytest tests/
```

### Creating New Features
1. Create feature branch
2. Implement changes
3. Add tests
4. Run style checks
5. Submit pull request

## License and Credits

This project is licensed under the MIT License.

### Acknowledgments
- YOLOv8 by Ultralytics
- SpeechBrain project
- OpenCV community
- Flask and Socket.IO teams

### Model Credits
- Person Detection: YOLOv8 by Ultralytics
- Emotion Detection: Custom trained model based on HuBERT
- Audio Processing: Custom trained model based on Wav2Vec2
