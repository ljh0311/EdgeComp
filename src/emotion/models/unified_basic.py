"""
Unified Basic Emotion Detection
===========================
Lightweight emotion detection optimized for edge devices.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from .unified_sound_detector import BaseSoundDetector

class LightweightEmotionModel(nn.Module):
    """Ultra-lightweight emotion recognition model for edge devices."""
    
    def __init__(self, input_size=128, num_emotions=4):
        super().__init__()
        # Minimal architecture for maximum efficiency
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_emotions)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class UnifiedBasicDetector(BaseSoundDetector):
    """Unified basic emotion detector optimized for edge devices."""
    
    EMOTIONS = ["angry", "happy", "sad", "neutral"]
    
    def __init__(self, config: dict, web_app=None):
        """Initialize the basic emotion detector.
        
        Args:
            config (dict): Configuration dictionary containing model settings
            web_app: Optional web application instance for real-time updates
        """
        super().__init__(config, web_app)
        
        # Edge optimizations
        self.use_quantization = config.get('use_quantization', True)
        self.feature_cache_size = config.get('feature_cache_size', 10)
        self.feature_cache = {}
        
        # Audio processing settings optimized for Raspberry Pi
        self.n_mfcc = 20  # Reduced number of MFCC features
        self.n_mels = 32  # Reduced number of mel bands
        self.hop_length = 512  # Increased hop length for faster processing
        
        try:
            self._initialize_model(config)
            self.logger.info(f"Basic detector initialized on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Basic detector: {str(e)}")
            raise
    
    def _initialize_model(self, config: dict):
        """Initialize the lightweight model."""
        try:
            model_path = config.get('model_path')
            if not model_path:
                model_path = os.path.join(os.path.dirname(__file__), "models", "basic_emotion.pt")
            
            # Initialize model
            self.model = LightweightEmotionModel(
                input_size=self.n_mfcc * 3,  # MFCC + delta + delta2
                num_emotions=len(self.EMOTIONS)
            )
            
            # Load weights if available
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            
            # Move to device and optimize
            self.model = self.model.to(self.device)
            if self.use_quantization:
                self._quantize_model()
            
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Error loading Basic model: {str(e)}")
            raise
    
    def _quantize_model(self):
        """Apply quantization for edge deployment."""
        try:
            # Use INT8 quantization for maximum efficiency
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},
                dtype=torch.qint8
            )
            self.logger.info("Model quantized to INT8")
        except Exception as e:
            self.logger.error(f"Error during model quantization: {str(e)}")
    
    def _extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract minimal but effective features for emotion recognition.
        
        Args:
            audio_data: Audio data array
            
        Returns:
            np.ndarray: Extracted features
        """
        try:
            # Check feature cache
            audio_hash = hash(audio_data.tobytes())
            if audio_hash in self.feature_cache:
                return self.feature_cache[audio_hash]
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            
            # Compute deltas for temporal information
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Combine features
            features = np.concatenate([
                np.mean(mfcc, axis=1),
                np.mean(mfcc_delta, axis=1),
                np.mean(mfcc_delta2, axis=1)
            ])
            
            # Update cache
            if len(self.feature_cache) >= self.feature_cache_size:
                self.feature_cache.pop(next(iter(self.feature_cache)))
            self.feature_cache[audio_hash] = features
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros(self.n_mfcc * 3)
    
    @property
    def model_name(self) -> str:
        """Return the name of the model"""
        return "Basic NN (Edge Optimized)"
    
    @property
    def supported_emotions(self) -> list:
        """Return list of supported emotions"""
        return self.EMOTIONS
    
    def detect(self, audio_data: np.ndarray) -> tuple:
        """Detect emotions with minimal processing.
        
        Args:
            audio_data: Audio data array (sampling rate should be 16kHz)
            
        Returns:
            tuple: (emotion_label, confidence)
        """
        if not self.is_running:
            return None, 0.0
        
        try:
            # Extract features
            features = self._extract_features(audio_data)
            
            # Convert to tensor
            inputs = torch.FloatTensor(features).to(self.device)
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
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
    
    def __del__(self):
        """Cleanup resources."""
        try:
            # Clear feature cache
            self.feature_cache.clear()
            # Clear CUDA cache if using GPU
            if hasattr(self, 'device') and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Error cleaning up Basic detector: {str(e)}") 