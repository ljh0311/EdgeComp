"""
Unified Sound Detection Module
==========================
Provides a unified interface for all sound-based emotion detection models.
"""

import logging
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

class BaseSoundDetector(ABC):
    """Base class for all sound-based emotion detectors."""
    
    def __init__(self, config: dict, web_app=None):
        """Initialize the base detector.
        
        Args:
            config (dict): Configuration for the detector
            web_app: Optional web application instance for real-time updates
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.web_app = web_app
        self.is_running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Common audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.block_size = 4000
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model"""
        pass
    
    @property
    @abstractmethod
    def supported_emotions(self) -> List[str]:
        """Return list of supported emotions"""
        pass
    
    def start(self):
        """Start the emotion detector."""
        self.is_running = True
        self.logger.info(f"{self.model_name} started")
    
    def stop(self):
        """Stop the emotion detector."""
        self.is_running = False
        self.logger.info(f"{self.model_name} stopped")
    
    @abstractmethod
    def detect(self, audio_data: np.ndarray) -> Tuple[Optional[str], float]:
        """Detect emotions in audio data.
        
        Args:
            audio_data: Audio data array (sampling rate should be 16kHz)
            
        Returns:
            tuple: (emotion_label, confidence)
        """
        pass
    
    def get_emotion_level(self, emotion: str, confidence: float) -> Optional[str]:
        """Determine alert level based on emotion and confidence.
        
        Args:
            emotion (str): Detected emotion
            confidence (float): Confidence score
            
        Returns:
            str: Alert level ('critical', 'warning', or None)
        """
        if confidence < self.config.get('confidence_threshold', 0.5):
            return None
            
        if emotion.lower() in ['anger', 'fear', 'sadness', 'worried'] and confidence > self.config.get('critical_threshold', 0.7):
            return 'critical'
        elif emotion.lower() in ['worried', 'disgust'] or confidence > self.config.get('warning_threshold', 0.6):
            return 'warning'
        
        return None
    
    def _emit_emotion(self, emotion: str, confidence: float):
        """Emit emotion detection to web interface if available."""
        if self.web_app:
            level = self.get_emotion_level(emotion, confidence)
            self.web_app.emit_emotion(emotion, confidence, level) 