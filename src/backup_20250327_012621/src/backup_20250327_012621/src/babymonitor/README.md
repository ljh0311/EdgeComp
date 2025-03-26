# Baby Monitor System

A comprehensive baby monitoring system with advanced detection capabilities.

## Overview

The Baby Monitor System is a Python-based application that provides real-time monitoring of babies using computer vision and audio processing. It includes features such as person detection, motion detection, emotion detection, and more.

## Features

- **Person Detection**: Detect people in the camera feed using YOLOv8 or lightweight TensorFlow Lite models.
- **Person Tracking**: Track people across frames to maintain identity and detect movement patterns.
- **Motion Detection**: Detect motion in the camera feed and identify potential falls or unusual movements.
- **Emotion Detection**: Detect emotions from audio input to identify crying or distress.
- **Web Interface**: Access the monitoring system through a web interface from any device.
- **Asynchronous Model Loading**: Load models in the background to prevent blocking the main thread.
- **Adaptive Processing**: Adjust processing based on system load to maintain performance.
- **Frame Skipping**: Skip frames to maintain FPS on resource-constrained devices.

## Project Structure

- **alerts/**: Alert system for notifications.
- **audio/**: Audio processing and emotion detection.
- **camera/**: Camera utilities and video stream handling.
- **core/**: Core functionality and shared components.
- **detectors/**: Various detectors for person detection, motion detection, etc.
- **emotion/**: Emotion detection models and processing.
- **models/**: Model files and model-related utilities.
- **utils/**: Utility functions and classes.
- **web/**: Web interface for accessing the monitoring system.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/babymonitor.git
   cd babymonitor
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download models:

   ```bash
   python -m src.babymonitor.utils.setup_models
   ```

## Usage

### Running the Baby Monitor System

```bash
python -m src.babymonitor.main
```

### Command-line Arguments

- `--config`: Path to configuration file.
- `--debug`: Enable debug logging.

### Testing Detectors

```bash
python -m src.babymonitor.utils.test_detectors --detector tracker --camera 0
```

### Testing Model Loader

```bash
python -m src.babymonitor.utils.test_model_loader --model yolov8n.pt --type yolov8
```

## Configuration

The system can be configured through the `config.py` file or by providing a JSON configuration file. See the [Configuration Documentation](src/babymonitor/utils/README.md#configuration) for more details.

## Performance Optimization

The system includes several performance optimizations:

- **Frame Skipping**: Skip frames to maintain FPS on resource-constrained devices.
- **Resolution Scaling**: Process frames at a lower resolution for faster detection.
- **Asynchronous Model Loading**: Load models in the background to prevent blocking the main thread.
- **Adaptive Processing**: Adjust processing based on system load to maintain performance.
- **Result Caching**: Cache detection results to avoid redundant processing.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
