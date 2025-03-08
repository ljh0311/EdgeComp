"""
HuBERT Emotion Detection Model
============================
Handles emotion detection using HuBERT model.
"""

import os
import logging
import torch
import torchaudio
import numpy as np
from transformers import AutoFeatureExtractor

logger = logging.getLogger(__name__)

class EmotionClassifier(torch.nn.Module):
    def __init__(self, num_emotions=6):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.linear1 = torch.nn.Linear(768, 256)
        self.linear2 = torch.nn.Linear(256, num_emotions)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class HuBERTEmotionDetector:
    """HuBERT-based emotion detector."""
    
    EMOTIONS = ['Natural', 'Anger', 'Worried', 'Happy', 'Fear', 'Sadness']
    
    def __init__(self, config, web_app=None):
        """Initialize the HuBERT emotion detector.
        
        Args:
            config (dict): Configuration dictionary containing model settings
            web_app: Optional web application instance for real-time updates
        """
        self.config = config
        self.web_app = web_app
        self.is_running = False
        
        # Load model
        try:
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
            
            # Set device
            device_str = config.get('device', 'cpu') if isinstance(config, dict) else 'cpu'
            self.device = torch.device(device_str)
            
            # Initialize and load model
            self.model = EmotionClassifier(len(self.EMOTIONS))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
            logger.info(f"Successfully loaded HuBERT model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load HuBERT model: {str(e)}")
            raise
    
    @property
    def supported_emotions(self):
        """Return list of supported emotions."""
        return self.EMOTIONS
    
    def start(self):
        """Start the emotion detector."""
        logger.info("Starting HuBERT emotion detector")
        self.is_running = True
    
    def stop(self):
        """Stop the emotion detector."""
        logger.info("Stopping HuBERT emotion detector")
        self.is_running = False
    
    def preprocess_audio(self, audio_data):
        """Preprocess audio data for the model.
        
        Args:
            audio_data: Audio data as numpy array or torch tensor
        
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
            sampling_rate=self.config.get('sampling_rate', 16000),
            return_tensors="pt"
        )
        return inputs.input_values.to(self.device)
    
    def detect(self, audio_data):
        """Detect emotions in audio data.
        
        Args:
            audio_data: Audio data as numpy array or torch tensor
        
        Returns:
            tuple: (emotion, confidence)
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
            
            # Send to web interface if available
            if self.web_app:
                self.web_app.send_emotion(emotion, confidence)
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Error during emotion detection: {str(e)}")
            return None, 0.0 