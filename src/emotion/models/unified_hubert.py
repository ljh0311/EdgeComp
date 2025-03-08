"""
Unified HuBERT Emotion Detection
============================
Combines HuBERT detector and model implementation for emotion detection.
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoFeatureExtractor
from .unified_sound_detector import BaseSoundDetector

class EmotionClassifier(nn.Module):
    """HuBERT-based emotion classifier."""
    
    def __init__(self, num_emotions=6):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, num_emotions)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class UnifiedHuBERTDetector(BaseSoundDetector):
    """Unified HuBERT-based emotion detector combining detector and model logic."""
    
    EMOTIONS = ['Natural', 'Anger', 'Worried', 'Happy', 'Fear', 'Sadness']
    
    def __init__(self, config: dict, web_app=None):
        """Initialize the HuBERT emotion detector.
        
        Args:
            config (dict): Configuration dictionary containing model settings
            web_app: Optional web application instance for real-time updates
        """
        super().__init__(config, web_app)
        
        try:
            self._initialize_model(config)
            self.logger.info(f"HuBERT detector initialized on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize HuBERT detector: {str(e)}")
            raise
    
    def _initialize_model(self, config: dict):
        """Initialize the HuBERT model."""
        try:
            # Get model path
            if isinstance(config, dict):
                model_dir = config.get('model_path', '')
            else:
                model_dir = str(config)
                
            if not model_dir:
                raise ValueError("Model path not provided in config")
                
            if os.path.isdir(model_dir):
                model_path = os.path.join(model_dir, "hubert-base-ls960_emotion.pt")
            else:
                model_path = model_dir
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Initialize model components
            self.model = EmotionClassifier(len(self.EMOTIONS))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
            
        except Exception as e:
            self.logger.error(f"Error loading HuBERT model: {str(e)}")
            raise
    
    @property
    def model_name(self) -> str:
        """Return the name of the model"""
        return "HuBERT (High Accuracy)"
    
    @property
    def supported_emotions(self) -> list:
        """Return list of supported emotions"""
        return self.EMOTIONS
    
    def preprocess_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """Preprocess audio data for the model.
        
        Args:
            audio_data: Audio data as numpy array
        
        Returns:
            torch.Tensor: Preprocessed audio features
        """
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)
        
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        
        # Normalize audio
        if audio_data.abs().max() > 1.0:
            audio_data = audio_data / audio_data.abs().max()
        
        # Extract features
        inputs = self.feature_extractor(
            audio_data.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        return inputs.input_values.to(self.device)
    
    def detect(self, audio_data: np.ndarray) -> tuple:
        """Detect emotions in audio data.
        
        Args:
            audio_data: Audio data array (sampling rate should be 16kHz)
            
        Returns:
            tuple: (emotion_label, confidence)
        """
        if not self.is_running:
            return None, 0.0
        
        try:
            # Preprocess audio
            inputs = self.preprocess_audio(audio_data)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                emotion_idx = torch.argmax(probs).item()
                confidence = probs[0][emotion_idx].item()
            
            emotion = self.EMOTIONS[emotion_idx]
            
            # Emit emotion if confidence is high enough
            if confidence > self.config.get('confidence_threshold', 0.5):
                self._emit_emotion(emotion, confidence)
            
            return emotion, confidence
            
        except Exception as e:
            self.logger.error(f"Error during emotion detection: {str(e)}")
            return None, 0.0 