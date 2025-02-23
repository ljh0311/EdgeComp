# Model Files

This directory contains the trained models used by the Baby Monitor System.

## Required Models

1. `hubert-base-ls960_emotion.pt` - Emotion detection model (HuBERT-based)
   - Used for detecting emotions in baby sounds
   - Trained on 6 emotion classes: Natural, Anger, Worried, Happy, Fear, Sadness
   - Model accuracy: 77.47%

2. `yolov8n.pt` - Person detection model (YOLOv8 nano)
   - Used for detecting people and their positions
   - Pre-trained on COCO dataset

3. `yolov8n-pose.pt` - Pose estimation model (YOLOv8 nano)
   - Used for detecting body poses and tracking movement
   - Pre-trained on COCO keypoints

## Model Sources

- Emotion detection model: Trained using HuBERT base model on custom emotion dataset
- YOLOv8 models: Downloaded automatically from Ultralytics 