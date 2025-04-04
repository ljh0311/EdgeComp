# Baby Monitor Examples

This directory contains example scripts demonstrating how to use the baby monitor system.

## Baby Monitor Example

The `baby_monitor_example.py` script demonstrates how to use the baby monitor system with different detector types, including the lightweight detector.

### Usage

```bash
# Use the lightweight detector (default)
python src/examples/baby_monitor_example.py

# Use the YOLOv8 detector
python src/examples/baby_monitor_example.py --detector yolov8

# Adjust the number of threads for the lightweight detector
python src/examples/baby_monitor_example.py --threads 2

# Change the camera resolution
python src/examples/baby_monitor_example.py --resolution 320x240

# Use a different camera
python src/examples/baby_monitor_example.py --camera 1

# Specify a custom host and port
python src/examples/baby_monitor_example.py --host localhost --port 8080
```

### Command Line Arguments

- `--detector`: Type of detector to use (lightweight, yolov8)
- `--host`: Host address to bind to (default: 0.0.0.0)
- `--port`: Port to listen on (default: 5000)
- `--threads`: Number of threads for lightweight detector (default: 4)
- `--camera`: Camera index (default: 0)
- `--resolution`: Camera resolution in WxH format (default: 640x480)

### Accessing the Web Interface

Once the example is running, you can access the web interface by opening a web browser and navigating to:

```
http://localhost:5000
```

The web interface provides:
- Live video feed with person detection
- Controls for switching between detectors
- Status information and alerts

### Performance Considerations

- The lightweight detector is best for resource-constrained devices like Raspberry Pi
- The YOLOv8 detector is best for desktop/laptop with decent CPU or GPU
- You can switch between detectors in real-time using the web interface

## Detector Example

The `detector_example.py` script demonstrates how to use the detector factory to easily switch between different detector implementations.

For more information on the detector example, see the [Detector Example Documentation](../babymonitor/detectors/LIGHTWEIGHT_DETECTION.md) 