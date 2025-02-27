"""
Emotion Detection Module
=====================
Handles emotion detection using HuBERT model.
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
        # Create a sequential classifier with correct dimensions
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

class EmotionDetector:
    def __init__(self, config):
        """
        Initialize the emotion detector.
        
        Args:
            config (dict): Configuration for emotion detection
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Initialize feature extractor
            self.embedding = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')
            
            # Initialize emotion classifier
            self.model = EmotionClassifier().to(self.device)
            
            # Load model weights
            state_dict = torch.load(config['model_path'], map_location=self.device)
            
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
            
            self.emotion_labels = ["Natural", "Anger", "Worried", "Happy", "Fear", "Sadness"]
            self.logger.info("Emotion detection model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize emotion detection model: {str(e)}")
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
                emotion = self.emotion_labels[pred_idx]
                
                self.logger.debug(f"Detected emotion: {emotion} with confidence {confidence:.2f}")
                return emotion, confidence
                
        except Exception as e:
            self.logger.error(f"Error in emotion detection: {str(e)}")
            return None, 0.0
    
    def get_emotion_level(self, emotion, confidence):
        """
        Determine alert level based on emotion and confidence.
        
        Args:
            emotion (str): Detected emotion
            confidence (float): Confidence score
            
        Returns:
            str: Alert level ('critical', 'warning', or None)
        """
        if confidence < self.config['confidence_threshold']:
            return None
            
        if emotion in ['Anger', 'Fear', 'Sadness'] and confidence > self.config['critical_threshold']:
            return 'critical'
        elif emotion in ['Worried'] or confidence > self.config['warning_threshold']:
            return 'warning'
        
        return None 