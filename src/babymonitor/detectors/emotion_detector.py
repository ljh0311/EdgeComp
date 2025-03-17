"""
Emotion Detection Module
---------------------
Implements emotion detection using HuBERT model for audio analysis.
This module provides efficient emotion detection from audio data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import logging
import numpy as np
import time
import psutil
from typing import Dict, List, Tuple, Optional
from .base_detector import BaseDetector

# Configure logging
logger = logging.getLogger(__name__)

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

class EmotionDetector(BaseDetector):
    """Emotion detector using HuBERT model for audio analysis."""
    
    def __init__(self, 
                 model_path: str = "models/emotion_model.pth",
                 confidence_threshold: float = 0.6,
                 warning_threshold: float = 0.7,
                 critical_threshold: float = 0.8,
                 sample_rate: int = 16000):
        """Initialize the emotion detector.
        
        Args:
            model_path: Path to the emotion detection model
            confidence_threshold: Minimum confidence for valid detection
            warning_threshold: Threshold for warning level
            critical_threshold: Threshold for critical level
            sample_rate: Audio sample rate (default: 16kHz)
        """
        super().__init__(threshold=confidence_threshold)
        
        # Configuration
        self.config = {
            'model_path': model_path,
            'confidence_threshold': confidence_threshold,
            'warning_threshold': warning_threshold,
            'critical_threshold': critical_threshold,
            'sample_rate': sample_rate
        }
        
        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device} for emotion detection")
        
        # Performance optimization
        self.frame_skip = 5  # Process every 5th audio frame (emotions change slowly)
        self.cache_duration = 2.0  # Cache results for 2 seconds
        self.last_process_time = 0
        self.cached_result = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the emotion detection model."""
        try:
            # Initialize feature extractor
            self.embedding = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')
            
            # Initialize emotion classifier
            self.model = EmotionClassifier().to(self.device)
            
            # Load model weights
            state_dict = torch.load(self.config['model_path'], map_location=self.device)
            
            # Create new state dict with proper structure
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("BERT."):
                    # Map BERT keys to hubert
                    new_key = "hubert." + key[5:]
                    new_state_dict[new_key] = value
                elif key.startswith("classifier."):
                    # Layer mapping
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
            
            # Set emotion labels
            self.emotion_labels = ["Natural", "Anger", "Worried", "Happy", "Fear", "Sadness"]
            logger.info("Emotion detection model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize emotion detection model: {str(e)}")
            raise
    
    def process_frame(self, audio_data: np.ndarray) -> Dict:
        """Process audio data for emotion detection.
        
        Args:
            audio_data: Audio data array (sampling rate should be 16kHz)
            
        Returns:
            Dict containing emotion detection results
        """
        start_time = time.time()
        
        # Skip frames for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0 and self.cached_result is not None:
            return self.cached_result
        
        # Check if we can use cached result
        current_time = time.time()
        if (current_time - self.last_process_time < self.cache_duration and 
            self.cached_result is not None):
            return self.cached_result
        
        try:
            with torch.no_grad():
                # Ensure audio data is the right shape and length
                audio_data = audio_data.flatten()
                
                # We need at least 1 second of audio (16000 samples)
                min_length = self.config['sample_rate']
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
                    sampling_rate=self.config['sample_rate'],
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
                
                # Determine alert level
                alert_level = self.get_emotion_level(emotion, confidence)
                
                # Calculate FPS
                frame_time = time.time() - start_time
                self.frame_times.append(frame_time)
                if len(self.frame_times) > self.max_frame_history:
                    self.frame_times.pop(0)
                self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                
                # Create result
                result = {
                    'emotion': emotion,
                    'confidence': confidence,
                    'alert_level': alert_level,
                    'all_emotions': {
                        self.emotion_labels[i]: probs[0][i].item() 
                        for i in range(len(self.emotion_labels))
                    },
                    'fps': self.fps
                }
                
                # Cache result
                self.cached_result = result
                self.last_process_time = current_time
                
                # Adjust frame skip based on performance
                self._adjust_frame_skip()
                
                return result
                
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return {
                'emotion': None,
                'confidence': 0.0,
                'alert_level': None,
                'all_emotions': {},
                'fps': 0
            }
    
    def detect(self, audio_data: np.ndarray) -> Tuple[Optional[str], float]:
        """Alias for process_frame to maintain backward compatibility.
        
        Args:
            audio_data: Audio data array
            
        Returns:
            tuple: (emotion_label, confidence)
        """
        result = self.process_frame(audio_data)
        return result['emotion'], result['confidence']
    
    def get_emotion_level(self, emotion: str, confidence: float) -> Optional[str]:
        """Determine alert level based on emotion and confidence.
        
        Args:
            emotion: Detected emotion
            confidence: Confidence score
            
        Returns:
            Alert level ('critical', 'warning', or None)
        """
        if confidence < self.config['confidence_threshold']:
            return None
            
        if emotion in ['Anger', 'Fear', 'Sadness'] and confidence > self.config['critical_threshold']:
            return 'critical'
        elif emotion in ['Worried'] or confidence > self.config['warning_threshold']:
            return 'warning'
        
        return None
    
    def _adjust_frame_skip(self):
        """Adjust frame skip based on performance."""
        avg_time = self.get_processing_time()
        if avg_time > 0:
            # Emotion detection is slow, so we aim for lower FPS
            target_fps = 5  # Target FPS for emotion detection
            ideal_skip = max(1, int(avg_time * target_fps))
            
            # Gradually adjust frame skip
            if ideal_skip > self.frame_skip:
                self.frame_skip = min(ideal_skip, self.frame_skip + 1)
            elif ideal_skip < self.frame_skip and self.frame_skip > 1:
                self.frame_skip = max(1, self.frame_skip - 1)
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Remove model references
            self.model = None
            self.embedding = None
        except Exception as e:
            logger.error(f"Error cleaning up emotion detector: {e}") 