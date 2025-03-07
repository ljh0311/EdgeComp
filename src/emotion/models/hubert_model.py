"""
HuBERT-based Emotion Recognition Model
===================================
Uses Facebook's HuBERT model for emotion recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import logging
import numpy as np

class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960')
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),  # First layer: 768 -> 128
            nn.ReLU(),           # Activation
            nn.Linear(128, 6)    # Second layer: 128 -> 6 (emotions)
        )
    
    def forward(self, x):
        outputs = self.hubert(x)
        features = outputs.last_hidden_state.mean(dim=1)
        x = self.classifier(features)
        return x

class HuBERTEmotionDetector:
    """Emotion detection using HuBERT model"""
    
    EMOTION_LABELS = ["Natural", "Anger", "Worried", "Happy", "Fear", "Sadness"]
    
    def __init__(self, model_path):
        """
        Initialize the emotion detector.
        
        Args:
            model_path (str): Path to the model weights
        """
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Initialize feature extractor
            self.embedding = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')
            
            # Initialize emotion classifier
            self.model = EmotionClassifier().to(self.device)
            
            # Load model weights
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Create new state dict with proper structure
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("BERT."):
                    # Map BERT keys to hubert
                    new_key = "hubert." + key[5:]
                    new_state_dict[new_key] = value
                elif key.startswith("classifier."):
                    # Map classifier keys to Sequential format
                    layer_num = int(key.split('.')[1])  # Get the layer number
                    if layer_num == 1:
                        new_key = "classifier.0"  # First linear layer
                    elif layer_num == 3:
                        new_key = "classifier.2"  # Second linear layer
                    new_key += key[key.rindex('.'):]  # Add .weight or .bias
                    new_state_dict[new_key] = value
            
            # Load state dict
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            
            self.logger.info("HuBERT emotion detection model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HuBERT emotion detection model: {str(e)}")
            raise
    
    def detect(self, audio_data):
        """
        Detect emotions in audio data.
        
        Args:
            audio_data: Audio data array (sampling rate should be 16kHz)
            
        Returns:
            tuple: (emotion_label, confidence)
        """
        try:
            with torch.no_grad():
                # Ensure audio data is the right shape and length
                audio_data = audio_data.flatten()
                
                # We need at least 1 second of audio (16000 samples)
                min_length = 16000
                if len(audio_data) < min_length:
                    # Pad with zeros if too short
                    padding = np.zeros(min_length - len(audio_data), dtype=np.float32)
                    audio_data = np.concatenate([audio_data, padding])
                elif len(audio_data) > min_length * 2:
                    # If too long, take the middle section
                    start = len(audio_data) // 4
                    end = start + min_length
                    audio_data = audio_data[start:end]
                
                # Preprocess audio
                inputs = self.embedding(
                    audio_data,
                    sampling_rate=16000,
                    padding=True,
                    return_tensors='pt'
                ).input_values.to(self.device)
                
                # Get emotion predictions
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=1)
                
                # Get predicted emotion
                pred_idx = torch.argmax(probs, dim=1)[0]
                confidence = probs[0][pred_idx].item()
                emotion = self.EMOTION_LABELS[pred_idx]
                
                return emotion, confidence
                
        except Exception as e:
            self.logger.error(f"Error detecting emotion: {str(e)}")
            return "Unknown", 0.0 