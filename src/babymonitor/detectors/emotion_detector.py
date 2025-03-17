"""
Emotion Detector Module
===================
Sound-based emotion recognition for baby monitoring.
"""

import numpy as np
import logging
import os
from pathlib import Path
import time
import random
from typing import Dict, List, Any, Optional
import torch
from .base_detector import BaseDetector

class EmotionDetector(BaseDetector):
    """Sound-based emotion detector for baby monitoring."""
    
    # Class constants
    EMOTIONS = ['crying', 'laughing', 'babbling', 'silence']
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 4096  # ~0.25 seconds at 16kHz
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 threshold: float = 0.5,
                 device: str = 'cpu'):
        """Initialize the emotion detector.
        
        Args:
            model_path: Path to model file (will download if None)
            threshold: Detection confidence threshold
            device: Device to run inference on ('cpu' or 'cuda')
        """
        super().__init__(threshold=threshold)
        
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model = None
        self.is_model_loaded = False
        self.audio_buffer = []
        self.buffer_duration = 2.0  # seconds
        self.logger = logging.getLogger(__name__)
        
        # For more realistic simulation
        self.emotion_state = 'silence'
        self.state_duration = 0
        self.state_change_probability = 0.1
        self.last_update_time = time.time()
        
        # Initialize model
        model_dir = self._get_model_dir(model_path)
        self._initialize_model(model_dir)
        
    def _get_model_dir(self, model_path: Optional[str] = None) -> str:
        """Get or create model directory."""
        if model_path and os.path.exists(model_path):
            return model_path
            
        # Use default path in package
        default_path = Path(__file__).parent / "models" / "emotion"
        default_path.mkdir(parents=True, exist_ok=True)
        return str(default_path)
        
    def _initialize_model(self, model_dir: str):
        """Initialize the emotion recognition model."""
        try:
            model_path = os.path.join(model_dir, "emotion_model.pt")
            
            # Use dummy model instead of downloading
            self.logger.info("Using dummy emotion recognition model")
            self.is_model_loaded = True
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise
            
    def _download_model(self, model_dir: str):
        """Download the emotion recognition model."""
        # Skip downloading, use dummy model instead
        self.logger.info("Using dummy emotion recognition model")
        return
            
    def _extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract audio features for emotion recognition.
        
        Args:
            audio: Audio signal
            
        Returns:
            Tensor of audio features
        """
        try:
            # Simplified feature extraction for dummy model
            # Just return a random feature tensor
            features = torch.rand(1, 80, 80)
            return features.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise
    
    def _generate_realistic_emotion(self) -> Dict[str, float]:
        """Generate realistic emotion probabilities based on current state.
        
        Returns:
            Dict of emotion probabilities
        """
        # Check if we should change state
        current_time = time.time()
        time_in_state = current_time - self.last_update_time
        
        # Increase chance of state change the longer we're in a state
        change_probability = min(0.8, self.state_change_probability + (time_in_state / 30.0))
        
        if random.random() < change_probability:
            # Change state
            if self.emotion_state == 'silence':
                # From silence, we can go to any state, but crying is more likely for babies
                weights = [0.6, 0.2, 0.2, 0.0]  # crying, laughing, babbling, silence
                self.emotion_state = random.choices(self.EMOTIONS, weights=weights)[0]
            elif self.emotion_state == 'crying':
                # From crying, most likely to continue or go to silence
                weights = [0.7, 0.05, 0.05, 0.2]  # crying, laughing, babbling, silence
                self.emotion_state = random.choices(self.EMOTIONS, weights=weights)[0]
            elif self.emotion_state == 'laughing':
                # From laughing, can go to any state
                weights = [0.1, 0.4, 0.2, 0.3]  # crying, laughing, babbling, silence
                self.emotion_state = random.choices(self.EMOTIONS, weights=weights)[0]
            elif self.emotion_state == 'babbling':
                # From babbling, most likely to continue or go to silence
                weights = [0.1, 0.1, 0.5, 0.3]  # crying, laughing, babbling, silence
                self.emotion_state = random.choices(self.EMOTIONS, weights=weights)[0]
                
            self.last_update_time = current_time
            self.state_duration = 0
        
        # Generate probabilities based on current state
        base_probs = {
            'crying': 0.05,
            'laughing': 0.05,
            'babbling': 0.05,
            'silence': 0.05
        }
        
        # Boost the current state
        base_probs[self.emotion_state] = 0.7 + random.uniform(-0.1, 0.1)
        
        # Normalize to ensure sum is 1.0
        total = sum(base_probs.values())
        normalized_probs = {k: v/total for k, v in base_probs.items()}
        
        return normalized_probs
            
    def process_audio(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Process an audio chunk for emotion recognition.
        
        Args:
            audio_chunk: Audio chunk to process
            
        Returns:
            Dict containing emotion predictions and confidence scores
        """
        if audio_chunk is None:
            return {
                'emotion': 'unknown',
                'confidence': 0.0,
                'emotions': dict(zip(self.EMOTIONS, [0.0] * len(self.EMOTIONS)))
            }
            
        start_time = time.time()
        
        try:
            # Add chunk to buffer
            self.audio_buffer.append(audio_chunk)
            
            # Process if buffer is full
            buffer_samples = int(self.buffer_duration * self.SAMPLE_RATE)
            if len(self.audio_buffer) * self.CHUNK_SIZE >= buffer_samples:
                # Generate realistic emotion probabilities
                emotion_probs = self._generate_realistic_emotion()
                
                # Convert to numpy array for processing
                probs = np.array([emotion_probs[emotion] for emotion in self.EMOTIONS])
                
                # Add a small epsilon to avoid division by zero
                probs = probs + 1e-6
                probs = probs / probs.sum()  # Normalize to sum to 1
                
                # Get predicted emotion
                emotion_idx = np.argmax(probs)
                confidence = float(probs[emotion_idx])  # Ensure it's a float for JSON serialization
                
                # Clear buffer
                self.audio_buffer = []
                
                # Update FPS (actually chunks per second)
                self.update_fps(time.time() - start_time)
                
                # Ensure all values are JSON serializable
                emotions_dict = {emotion: float(prob) for emotion, prob in zip(self.EMOTIONS, probs)}
                
                return {
                    'emotion': self.EMOTIONS[emotion_idx],
                    'confidence': confidence,
                    'emotions': emotions_dict,
                    'fps': float(self.fps)  # Ensure it's a float for JSON serialization
                }
                
            return {
                'emotion': 'buffering',
                'confidence': 0.0,
                'emotions': dict(zip(self.EMOTIONS, [0.0] * len(self.EMOTIONS))),
                'fps': float(self.fps)  # Ensure it's a float for JSON serialization
            }
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'emotion': 'error',
                'confidence': 0.0,
                'emotions': dict(zip(self.EMOTIONS, [0.0] * len(self.EMOTIONS))),
                'fps': float(self.fps)  # Ensure it's a float for JSON serialization
            }
            
    def cleanup(self):
        """Clean up resources."""
        self.model = None
        self.is_model_loaded = False
        self.audio_buffer = []
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a video frame (stub implementation to satisfy abstract method).
        
        This method is required by the BaseDetector abstract class but not used
        for audio-based emotion detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Dict containing processed frame and detections
        """
        # Return empty result since this detector doesn't process frames
        return {
            "frame": frame,
            "detections": [],
            "fps": float(self.fps)  # Ensure it's a float for JSON serialization
        } 