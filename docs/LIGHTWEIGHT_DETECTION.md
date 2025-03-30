# Lightweight Detection System

This document explains how to use the lightweight detection system implemented in this project. The system is based on TensorFlow Lite and is designed to be efficient and run well on resource-constrained devices like Raspberry Pi.

## Overview

The lightweight detection system provides an alternative to the more resource-intensive YOLOv8-based detection system. It uses TensorFlow Lite for efficient inference and is optimized for performance on devices with limited computational resources.

## Setup

### 1. Install Dependencies

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

For Raspberry Pi, you'll need to install TensorFlow Lite Runtime instead of full TensorFlow:

```bash
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl
```

Note: The exact URL may vary depending on your Python version and Raspberry Pi model. Check the [TensorFlow Lite Runtime releases](https://github.com/google-coral/pycoral/releases) for the appropriate version.

### 2. Get a CPU-Compatible Model

You have several options to get a CPU-compatible TFLite model:

#### Option 1: Download a Pre-trained Model (Recommended)

Run the provided script to download a pre-trained TFLite model:

```bash
python scripts/download_tflite_model.py
```

This script will:

- Download a pre-trained MobileNet SSD v1 model from TensorFlow
- Save it to your models directory as person_detection_model.tflite
- Create a person_labels.txt file

#### Option 2: Convert a Model

If you have TensorFlow installed, you can create a simple model:

```bash
python scripts/convert_to_cpu_model.py
```

This script will:

- Create a simple MobileNetV2-based model
- Convert it to TFLite format
- Save it to your models directory

#### Option 3: Use the Detector Factory (Automatic)

The detector factory will automatically attempt to download a CPU-compatible model if it detects that the current model contains EdgeTPU operations.

## Usage

### Using the Detector Factory (Recommended)

The easiest way to use the lightweight detector is through the detector factory:

```python
from babymonitor.detectors.detector_factory import DetectorFactory, DetectorType

# Create a lightweight detector
detector = DetectorFactory.create_detector(
    DetectorType.LIGHTWEIGHT.value,
    config={
        "threshold": 0.6,
        "resolution": (320, 240),
        "num_threads": 4
    }
)

# Create a video stream
video_stream = DetectorFactory.create_video_stream(
    DetectorType.LIGHTWEIGHT.value,
    config={
        "resolution": (320, 240),
        "framerate": 30,
        "camera_index": 0
    }
)

# Process frames
video_stream.start()
while True:
    frame = video_stream.read()
    results = detector.process_frame(frame)
    
    # Access detection results
    detections = results['detections']
    processed_frame = results['frame']
    fps = results['fps']
    
    # Display the frame
    cv2.imshow('Detections', processed_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
video_stream.stop()
detector.cleanup()
cv2.destroyAllWindows()
```

### Using the Example Script

You can use the provided example script to test the lightweight detector:

```bash
# Run with default settings
python src/examples/detector_example.py

# Specify a different model
python src/examples/detector_example.py --model models/custom_model.tflite

# Adjust detection threshold
python src/examples/detector_example.py --threshold 0.6

# Change resolution
python src/examples/detector_example.py --resolution 320x240

# Specify number of threads
python src/examples/detector_example.py --threads 2
```

### Direct Usage

If you prefer to use the lightweight detector directly:

```python
from babymonitor.detectors.lightweight_detector import LightweightDetector, VideoStream

# Initialize the video stream
video_stream = VideoStream(resolution=(640, 480)).start()

# Initialize the detector
detector = LightweightDetector(
    model_path="models/person_detection_model.tflite",
    label_path="models/person_labels.txt",
    threshold=0.5,
    num_threads=4
)

# Process frames
while True:
    frame = video_stream.read()
    results = detector.process_frame(frame)
    
    # Access detection results
    detections = results['detections']
    
    # Display the frame
    cv2.imshow('Detections', results['frame'])
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
video_stream.stop()
detector.cleanup()
cv2.destroyAllWindows()
```

## Performance Optimization

To optimize performance on resource-constrained devices:

1. **Lower the resolution**: Use a smaller resolution like 320x240 instead of 640x480.
2. **Increase the threshold**: Use a higher confidence threshold (e.g., 0.6 or 0.7) to reduce false positives.
3. **Adjust the number of threads**: Set the `num_threads` parameter to match your CPU core count.
4. **Reduce processing frequency**: If you're integrating with your own application, you can process frames at a lower frequency.

## Troubleshooting

### Camera Issues

If you encounter issues with the camera:

1. Make sure the camera is properly connected.
2. Try a different camera index (use the `camera_index` parameter).
3. Check if the camera is being used by another application.

### Model Issues

If you encounter issues with the model:

1. **EdgeTPU Error**: If you see an error about "edgetpu-custom-op", it means the model contains EdgeTPU operations but EdgeTPU is not available. Run `python scripts/download_tflite_model.py` to download a CPU-compatible model.
2. Make sure the model file exists and is accessible.
3. Check if the labels file matches the model.
4. Try a different model or convert your own model to TensorFlow Lite format.

## Integration with the Baby Monitor System

The lightweight detector is fully integrated with the baby monitor system through the detector factory. You can easily switch between different detector implementations based on your hardware capabilities:

```python
from babymonitor.detectors.detector_factory import DetectorFactory, DetectorType
import platform
import psutil

def is_resource_constrained():
    """Check if the system has limited resources."""
    # Check if running on Raspberry Pi
    if platform.machine() in ('armv7l', 'armv6l'):
        return True
        
    # Check available memory (less than 2GB)
    if psutil.virtual_memory().total < 2 * 1024 * 1024 * 1024:
        return True
        
    return False

# Choose detector based on system resources
if is_resource_constrained():
    detector_type = DetectorType.LIGHTWEIGHT.value
else:
    detector_type = DetectorType.YOLOV8.value

# Create the detector
detector = DetectorFactory.create_detector(detector_type)
```

## Credits

The lightweight detection system is inspired by the BirdRepeller project and uses TensorFlow Lite for efficient inference.
