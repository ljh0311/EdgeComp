"""
Unified Wav2Vec2 Emotion Detection
==============================
Optimized Wav2Vec2-based emotion detection for edge devices.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Config
from .unified_sound_detector import BaseSoundDetector

class EdgeOptimizedClassifier(nn.Module):
    """Memory-efficient emotion classifier for edge devices."""
    
    def __init__(self, input_size=768, num_emotions=8):
        super().__init__()
        # Reduced architecture for edge devices
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),  # Reduced from 256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_emotions)
        )
        
    def forward(self, x):
        return self.classifier(x)

class UnifiedWav2Vec2Detector(BaseSoundDetector):
    """Unified Wav2Vec2-based emotion detector optimized for edge devices."""
    
    EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    def __init__(self, config: dict, web_app=None):
        """Initialize the Wav2Vec2 emotion detector.
        
        Args:
            config (dict): Configuration dictionary containing model settings
            web_app: Optional web application instance for real-time updates
        """
        super().__init__(config, web_app)
        
        # Edge device optimizations
        self.use_quantization = config.get('use_quantization', True)
        self.use_int8 = config.get('use_int8', True)
        self.chunk_size = config.get('chunk_size', 4000)  # Smaller chunks for memory efficiency
        
        try:
            self._initialize_model(config)
            self.logger.info(f"Wav2Vec2 detector initialized on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Wav2Vec2 detector: {str(e)}")
            raise
    
    def _initialize_model(self, config: dict):
        """Initialize the Wav2Vec2 model with edge optimizations."""
        try:
            # Load model path
            model_path = config.get('model_path', '')
            if not model_path:
                raise ValueError("Model path not provided in config")
            
            # Initialize base model with reduced size
            wav2vec2_config = Wav2Vec2Config.from_pretrained(
                "facebook/wav2vec2-base",
                num_hidden_layers=6,  # Reduced from 12
                hidden_size=512,      # Reduced from 768
                num_attention_heads=6  # Reduced from 12
            )
            
            self.feature_extractor = Wav2Vec2Model(wav2vec2_config)
            self.classifier = EdgeOptimizedClassifier(
                input_size=512,  # Match reduced hidden size
                num_emotions=len(self.EMOTIONS)
            )
            
            # Load weights if available
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.classifier.load_state_dict(state_dict)
            
            # Move models to device
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.classifier = self.classifier.to(self.device)
            
            # Apply quantization for edge devices
            if self.use_quantization:
                self._quantize_model()
            
            # Set to evaluation mode
            self.feature_extractor.eval()
            self.classifier.eval()
            
        except Exception as e:
            self.logger.error(f"Error loading Wav2Vec2 model: {str(e)}")
            raise
    
    def _quantize_model(self):
        """Apply quantization optimizations for edge deployment."""
        try:
            if self.use_int8:
                # INT8 quantization for maximum memory savings
                self.feature_extractor = torch.quantization.quantize_dynamic(
                    self.feature_extractor,
                    {torch.nn.Linear, torch.nn.Conv1d},
                    dtype=torch.qint8
                )
                self.classifier = torch.quantization.quantize_dynamic(
                    self.classifier,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            else:
                # FP16 quantization for better accuracy
                self.feature_extractor = self.feature_extractor.half()
                self.classifier = self.classifier.half()
                
            self.logger.info(f"Model quantized with INT8={self.use_int8}")
            
        except Exception as e:
            self.logger.error(f"Error during model quantization: {str(e)}")
            raise
    
    @property
    def model_name(self) -> str:
        """Return the name of the model"""
        return "Wav2Vec2 (Edge Optimized)"
    
    @property
    def supported_emotions(self) -> list:
        """Return list of supported emotions"""
        return self.EMOTIONS
    
    def preprocess_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """Preprocess audio data with edge optimizations.
        
        Args:
            audio_data: Audio data as numpy array
        
        Returns:
            torch.Tensor: Preprocessed audio features
        """
        # Convert to tensor
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)
        
        # Ensure correct shape
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        
        # Normalize audio (in-place for memory efficiency)
        if audio_data.abs().max() > 1.0:
            audio_data.div_(audio_data.abs().max())
        
        return audio_data.to(self.device)
    
    def detect(self, audio_data: np.ndarray) -> tuple:
        """Detect emotions with edge-optimized processing.
        
        Args:
            audio_data: Audio data array (sampling rate should be 16kHz)
            
        Returns:
            tuple: (emotion_label, confidence)
        """
        if not self.is_running:
            return None, 0.0
        
        try:
            # Process in chunks for memory efficiency
            chunks = torch.split(audio_data, self.chunk_size)
            features_list = []
            
            # Process each chunk
            with torch.no_grad():
                for chunk in chunks:
                    # Preprocess chunk
                    inputs = self.preprocess_audio(chunk)
                    
                    # Extract features
                    features = self.feature_extractor(inputs).last_hidden_state
                    features = torch.mean(features, dim=1)  # Pool features
                    features_list.append(features)
                
                # Combine chunk features
                combined_features = torch.mean(torch.stack(features_list), dim=0)
                
                # Get predictions
                logits = self.classifier(combined_features)
                probs = F.softmax(logits, dim=-1)
                emotion_idx = torch.argmax(probs).item()
                confidence = probs[emotion_idx].item()
            
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
            # Clear CUDA cache if using GPU
            if hasattr(self, 'device') and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Error cleaning up Wav2Vec2 detector: {str(e)}") 