# Lightweight Detector Integration Summary

This document summarizes the changes made to integrate the lightweight detector into the baby monitor system.

## Overview

The lightweight detector has been fully integrated into the baby monitor system, providing an efficient alternative to the YOLOv8-based detector for resource-constrained devices like Raspberry Pi. The integration includes:

1. Configuration settings for the lightweight detector
2. Updates to the main program to use the detector factory
3. Updates to the web interface to support switching between detectors
4. A new example script demonstrating how to use the baby monitor system with the lightweight detector

## Changes Made

### 1. Configuration Updates

The `Config` class in `src/babymonitor/config.py` has been updated to include:

- Settings for the lightweight detector (model path, label path, threshold, resolution, threads)
- Auto-selection of the lightweight detector on resource-constrained devices
- Web interface settings (host, port)

```python
# Lightweight detection settings
LIGHTWEIGHT_DETECTION = {
    "model_path": "models/person_detection_model.tflite",
    "label_path": "models/person_labels.txt",
    "threshold": 0.5,
    "resolution": (640, 480),
    "num_threads": 4,
    "camera_index": 0,
}

# Detector selection
DETECTOR_TYPE = "yolov8"
# Auto-select lightweight detector on resource-constrained devices
if platform.machine() in ('armv7l', 'armv6l') or os.environ.get("USE_LIGHTWEIGHT", "0") == "1":
    DETECTOR_TYPE = "lightweight"
```

### 2. Main Program Updates

The `main.py` file has been updated to use the detector factory for creating the person detector:

```python
# Initialize detectors based on configuration
if Config.DETECTOR_TYPE.lower() == "lightweight":
    logger.info("Using lightweight detector")
    person_detector = DetectorFactory.create_detector(
        DetectorType.LIGHTWEIGHT.value,
        config=Config.LIGHTWEIGHT_DETECTION
    )
else:
    logger.info("Using YOLOv8 detector")
    person_detector = PersonDetector(
        model_path=Config.PERSON_DETECTION["model_path"],
        device=Config.PERSON_DETECTION["device"]
    )
```

### 3. Web Interface Updates

The web interface (`src/babymonitor/web/app.py`) has been updated to:

- Initialize the detector based on the configuration
- Add a socket event handler for switching between detectors
- Clean up resources when switching detectors

The HTML template (`src/babymonitor/web/templates/index.html`) has been updated to add a control for switching between detectors:

```html
<div class="control-group">
    <label for="detectorSelect">Detector:</label>
    <select id="detectorSelect" onchange="switchDetector(this.value)">
        <option value="yolov8">YOLOv8 (High Accuracy)</option>
        <option value="lightweight">Lightweight (Low Resource)</option>
    </select>
</div>
```

### 4. Example Script

A new example script (`src/examples/baby_monitor_example.py`) has been created to demonstrate how to use the baby monitor system with the lightweight detector. The script includes:

- Command line arguments for configuring the detector type, threads, resolution, etc.
- Initialization of the web interface and detectors
- Proper cleanup of resources

## Usage

To use the baby monitor system with the lightweight detector:

1. Make sure you have the required dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Download a CPU-compatible model if you don't have one:
   ```bash
   python scripts/download_tflite_model.py
   ```

3. Run the example script:
   ```bash
   python src/examples/baby_monitor_example.py
   ```

4. Access the web interface at http://localhost:5000

5. Use the detector dropdown to switch between YOLOv8 and lightweight detectors in real-time.

## Environment Variables

You can use the following environment variables to control the detector selection:

- `USE_LIGHTWEIGHT=1`: Force the use of the lightweight detector
- `USE_CUDA=1`: Use CUDA for YOLOv8 detector (if available)

## Performance Considerations

- The lightweight detector is best for resource-constrained devices like Raspberry Pi
- The YOLOv8 detector is best for desktop/laptop with decent CPU or GPU
- You can switch between detectors in real-time using the web interface

## Next Steps

1. Test the integration on different devices, especially resource-constrained ones like Raspberry Pi
2. Fine-tune the lightweight detector parameters for optimal performance
3. Consider adding more detector types to the factory (e.g., specialized detectors for specific use cases) 