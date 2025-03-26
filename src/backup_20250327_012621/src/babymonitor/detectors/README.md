# Baby Monitor Detectors

This directory contains various detector implementations for the Baby Monitor System.

## Detector Types

### 1. Lightweight Detector

The `lightweight_detector.py` module provides a TensorFlow Lite-based detector optimized for CPU usage. This detector is ideal for resource-constrained devices like Raspberry Pi.

#### Features

- Efficient TensorFlow Lite inference
- Multi-threading support for improved performance
- Optimized video streaming with frame skipping
- Low memory footprint
- Automatic fallback to CPU-compatible models

#### Performance Optimizations

- Reduced frame resolution for faster processing
- Frame skipping on resource-constrained devices
- Efficient buffer management
- Prioritized detection processing (top detections only)
- Reduced JPEG quality for web streaming

### 2. YOLOv8 Detector

The `person_detector.py` module provides a YOLOv8-based detector for high-accuracy person detection. This detector is ideal for systems with more computational resources.

#### Features

- High accuracy person detection
- Motion analysis (standing, moving, falling)
- GPU acceleration support
- Detailed metrics and performance monitoring

### 3. Person Tracker

The `person_tracker.py` module extends the YOLOv8 detector with tracking capabilities, allowing the system to track individuals across frames.

#### Features

- Person tracking across frames
- Unique ID assignment
- Movement path analysis
- Fall detection

## Detector Factory

The `detector_factory.py` module provides a factory pattern for creating detectors and video streams. This allows the system to easily switch between different detector implementations.

### Usage

```python
from babymonitor.detectors.detector_factory import DetectorFactory, DetectorType

# Create a lightweight detector
detector = DetectorFactory.create_detector(
    DetectorType.LIGHTWEIGHT.value,
    config={
        "threshold": 0.5,
        "resolution": (640, 480),
        "num_threads": 4
    }
)

# Create a video stream
video_stream = DetectorFactory.create_video_stream(
    DetectorType.LIGHTWEIGHT.value,
    config={
        "resolution": (640, 480),
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
```

## Performance Considerations

### Lightweight Detector

- Best for Raspberry Pi and other resource-constrained devices
- Optimized for CPU usage
- Lower accuracy but faster processing
- Typical FPS: 15-30 on Raspberry Pi 4, 30+ on desktop CPU

### YOLOv8 Detector

- Best for desktop/laptop with decent CPU or GPU
- Higher accuracy but more resource-intensive
- GPU acceleration recommended
- Typical FPS: 5-15 on CPU, 30+ on GPU

## Troubleshooting

### Low FPS

- Reduce resolution in the configuration
- Increase frame skipping on resource-constrained devices
- Use the lightweight detector instead of YOLOv8
- Ensure proper camera configuration (MJPG format, appropriate resolution)

### No Metrics in Web Interface

- Check if the detector has the required metrics methods
- Ensure the web app is properly configured
- Check browser console for errors

### Camera Issues

- Try different camera indices
- Check if the camera is being used by another application
- Verify camera permissions

## Recent Improvements

1. **Performance Optimizations**:
   - Added frame skipping for improved performance
   - Optimized buffer management in video streams
   - Reduced processing overhead in the detector

2. **Metrics Support**:
   - Added comprehensive metrics for monitoring
   - Improved error handling for metrics collection
   - Added fallback mechanisms for missing metrics

3. **Web Integration**:
   - Enhanced frame streaming to web clients
   - Added real-time detector switching
   - Improved error handling and recovery

4. **Resource Management**:
   - Better cleanup of resources
   - Reduced memory usage
   - Improved thread management

## Configuration Options

### Lightweight Detector

- `model_path`: Path to the TFLite model file
- `
