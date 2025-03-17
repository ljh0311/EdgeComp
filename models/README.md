# Baby Monitor Models

This directory contains models used by the Baby Monitor System.

## Person Detection Models

### Lightweight Detection (Default)
- `person_detection_model.tflite`: A lightweight TensorFlow Lite model for person detection
- `person_labels.txt`: Labels file for the person detection model

If these files are missing, you can download them by running:
```bash
python scripts/download_tflite_model.py
```

### YOLOv8 Detection (High Accuracy)
- `yolov8n.pt`: YOLOv8 nano model for high-accuracy person detection

## Audio Models
- `cry_detection_model.pth`: Model for detecting baby cries
- `emotion_model.pt`: Model for emotion detection from audio
- `hubert-base-ls960_emotion.pt`: HuBERT-based model for emotion detection

## Model Selection

The system will use the lightweight detector by default. You can switch between detectors in the web interface or by specifying the `--detector` argument when running the example:

```bash
# Use lightweight detector (default)
python src/examples/baby_monitor_example.py

# Use YOLOv8 detector
python src/examples/baby_monitor_example.py --detector yolov8
```

## Performance Considerations

- The lightweight detector is optimized for CPU usage and is ideal for resource-constrained devices like Raspberry Pi
- The YOLOv8 detector provides higher accuracy but requires more computational resources
