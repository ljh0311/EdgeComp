"""
Sound Detector Module
==================
Provides a high-level interface for sound detection and classification.
Wraps the AudioProcessor for use in the Baby Monitor System.
"""

import logging
import os
from pathlib import Path
import torch
from .audio_processor import AudioProcessor
import numpy as np

class SoundDetector:
    """Sound detector for baby monitoring.
    
    Provides a high-level interface for detecting and classifying sounds
    such as crying, laughing, and babbling.
    """
    
    def __init__(self, config=None):
        """Initialize the sound detector.
        
        Args:
            config: Configuration dictionary for audio processing.
                   If None, default configuration will be used.
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration if none provided
        if config is None:
            config = {
                'chunk_size': 1024,
                'sample_rate': 16000,
                'channels': 1,
                'format': 'paFloat32',
                'use_callback': False,
                'device': 'cpu',
                'model_path': self._get_model_path(),
                'alert_threshold': 0.6,
                'alert_cooldown': 5.0,  # seconds
            }
        
        self.config = config
        self.audio_processor = AudioProcessor(config, self._alert_callback)
        self.is_running = False
        self.last_sound_level = 0
        self.last_sound_status = 'quiet'
        self.last_emotion = 'background'
        self.last_emotion_confidence = 0
        
    def _get_model_path(self):
        """Get the path to the emotion model."""
        # Try to find the model in several possible locations
        possible_paths = [
            # In the models directory at the project root
            Path(__file__).parent.parent.parent.parent / "models" / "emotion_model.pt",
            # In the babymonitor models directory
            Path(__file__).parent.parent / "models" / "emotion_model.pt",
            # In the emotion directory
            Path(__file__).parent.parent / "emotion" / "models" / "emotion_model.pt"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # If no model found, return a default path
        return str(Path(__file__).parent.parent.parent.parent / "models" / "emotion_model.pt")
    
    def _alert_callback(self, emotion, confidence, audio_data):
        """Callback for audio alerts."""
        self.last_emotion = emotion
        self.last_emotion_confidence = confidence
        
        # Calculate sound level (simple RMS)
        if audio_data is not None:
            self.last_sound_level = float(np.sqrt(np.mean(np.square(audio_data))))
        
        # Determine sound status
        if emotion == 'cry' and confidence > self.config.get('alert_threshold', 0.6):
            self.last_sound_status = 'crying'
        elif emotion == 'laugh' and confidence > 0.5:
            self.last_sound_status = 'laughing'
        elif emotion == 'babble' and confidence > 0.5:
            self.last_sound_status = 'babbling'
        elif self.last_sound_level > 0.1:  # Arbitrary threshold for "noise"
            self.last_sound_status = 'noise'
        else:
            self.last_sound_status = 'quiet'
            
        self.logger.debug(f"Sound status: {self.last_sound_status}, emotion: {emotion}, confidence: {confidence:.2f}")
    
    def start(self):
        """Start sound detection."""
        if not self.is_running:
            self.logger.info("Starting sound detector")
            self.audio_processor.start()
            self.is_running = True
    
    def stop(self):
        """Stop sound detection."""
        if self.is_running:
            self.logger.info("Stopping sound detector")
            self.audio_processor.stop()
            self.is_running = False
    
    def get_sound_status(self):
        """Get the current sound status.
        
        Returns:
            dict: Dictionary containing sound status information.
        """
        return {
            'sound_level': self.last_sound_level,
            'sound_status': self.last_sound_status,
            'emotion': self.last_emotion,
            'confidence': self.last_emotion_confidence
        }
    
    def is_active(self):
        """Check if the sound detector is active.
        
        Returns:
            bool: True if the sound detector is active, False otherwise.
        """
        return self.is_running and self.audio_processor.is_active()
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        
    def __del__(self):
        """Clean up resources when the object is deleted."""
        self.cleanup() 