# Baby Emotion Detection Training

This directory contains the training code for the baby emotion detection model used in the Baby Monitor System.

## Model Architecture

The model is based on the HuBERT architecture and is fine-tuned for baby emotion detection. It can recognize the following emotions:
- Happy (playful sounds, laughing)
- Sad (crying)
- Hungry (specific cry patterns)
- Pain/Uncomfortable (distress sounds)
- Tired (sleepy sounds)
- Natural (normal babbling)

## Training Data Requirements

The training script expects data in the following structure:

```
data/
  baby_emotions/
    happy/
      audio1.wav
      audio2.wav
      ...
    sad/
      audio1.wav
      audio2.wav
      ...
    hungry/
      ...
    pain/
      ...
    tired/
      ...
    natural/
      ...
  train_metadata.json
  val_metadata.json
```

### Metadata Format

The metadata JSON files should have the following structure:

```json
{
  "audio_id": {
    "path": "relative/path/to/audio.wav",
    "emotion": "emotion_label",
    "duration": float,
    "sample_rate": int
  },
  ...
}
```

## Training Process

1. Install requirements:
   ```bash
   pip install torch torchaudio transformers pandas numpy
   ```

2. Prepare your dataset following the structure above

3. Run the training:
   ```bash
   python train_emotion_model.py
   ```

The script will:
- Load and preprocess the audio data
- Train the model using the HuBERT architecture
- Save the best model based on validation loss
- Output training metrics and logs

## Model Integration

The trained model will be saved as `hubert-base-ls960_emotion.pt` in the models directory. This model is compatible with the Baby Monitor System's emotion detection module.

To use the trained model:
1. Place the model file in the `models` directory
2. The system will automatically use it for emotion detection
3. The model provides real-time emotion detection with confidence scores

## Dataset Sources

Recommended datasets for training:
- Baby Cry Detection Dataset
- Baby Sound Dataset
- RAVDESS Dataset (for general emotion sounds)
- Custom recorded baby sounds (recommended for best results)

## Performance Metrics

The model should achieve:
- Minimum 75% accuracy on validation set
- Real-time inference (< 100ms per prediction)
- Low false positive rate for critical emotions (crying, pain) 