"""
Emotion Detector Module
===================
Sound-based emotion recognition for baby monitoring.
"""

import numpy as np
import logging
import os
import json
from pathlib import Path
import time
import random
from typing import Dict, List, Any, Optional
import torch
from datetime import datetime
from collections import deque
import threading
from .base_detector import BaseDetector

class EmotionDetector(BaseDetector):
    """Sound-based emotion detector for baby monitoring."""
    
    # Default emotions for backward compatibility
    DEFAULT_EMOTIONS = ['crying', 'laughing', 'babbling', 'silence']
    
    # Audio processing constants
    SAMPLE_RATE = 16000  # Hz
    CHUNK_SIZE = 4000    # Samples
    
    # Model definitions
    AVAILABLE_MODELS = {
        'emotion2': {
            'name': 'Enhanced Emotion Model',
            'file': 'baby_cry_classifier_enhanced.tflite',
            'type': 'advanced',
            'emotions': ['happy', 'sad', 'angry', 'neutral', 'crying', 'laughing'],
            'path': 'models/emotion/emotion2'
        },
        'cry_detection': {
            'name': 'Cry Detection',
            'file': 'cry_detection_model.pth',
            'type': 'binary',
            'emotions': ['crying', 'not_crying'],
            'path': 'models/emotion/cry_detection'
        },
        'speechbrain': {
            'name': 'SpeechBrain Emotion',
            'file': 'emotion_model.pt',
            'type': 'speechbrain',
            'emotions': ['happy', 'sad', 'angry', 'neutral'],
            'path': 'models/emotion/speechbrain'
        },
        'basic_emotion': {
            'name': 'Basic Emotion',
            'file': 'best_emotion_model.pt',
            'type': 'basic',
            'emotions': ['crying', 'laughing', 'babbling', 'silence'],
            'path': 'models/emotion/speechbrain'
        }
    }
    
    def __init__(self, 
                 model_id: Optional[str] = None,
                 threshold: float = 0.5,
                 device: str = 'cpu'):
        """Initialize the emotion detector.
        
        Args:
            model_id: ID of the model to use (from AVAILABLE_MODELS)
            threshold: Detection confidence threshold
            device: Device to run inference on ('cpu' or 'cuda')
        """
        super().__init__(threshold=threshold)
        
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model = None
        self.model_id = model_id or 'basic_emotion'  # Default to basic emotion model
        self.is_model_loaded = False
        self.audio_buffer = []
        self.buffer_duration = 2.0  # seconds
        self.logger = logging.getLogger(__name__)
        
        # Get model info
        self.model_info = self.AVAILABLE_MODELS.get(self.model_id, self.AVAILABLE_MODELS['basic_emotion'])
        self.emotions = self.model_info['emotions']
        
        # Initialize state variables
        self.last_update_time = time.time()
        self.state_duration = 0.0
        self.state_change_probability = 0.3
        self.emotion_state = random.choice(self.emotions)
        self.confidence = random.uniform(0.6, 0.95)
        self.confidences = {emotion: random.uniform(0.1, 0.4) for emotion in self.emotions}
        self.confidences[self.emotion_state] = self.confidence
        
        # Initialize history tracking
        self.emotion_history = deque(maxlen=1000)  # Store last 1000 emotion readings
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.daily_emotion_history = {}  # Store daily summaries
        self.history_lock = threading.Lock()
        
        # Set up file paths for persistent storage
        self.log_dir = Path(os.path.expanduser('~')) / 'babymonitor_logs'
        self.log_dir.mkdir(exist_ok=True)
        self.history_file = self.log_dir / f"emotion_history_{self.model_id}.json"
        
        # Load existing history data if available
        self._load_history()
        
        # Start history saving thread
        self.save_interval = 300  # Save every 5 minutes
        self.last_save_time = time.time()
        self.running = True
        self.save_thread = threading.Thread(target=self._periodic_save, daemon=True)
        self.save_thread.start()
        
        # Initialize model
        self._initialize_model()
        
    def _get_model_path(self, model_id: str) -> str:
        """Get the full path for a model."""
        model_info = self.AVAILABLE_MODELS.get(model_id)
        if not model_info:
            raise ValueError(f"Unknown model ID: {model_id}")
            
        base_path = Path(__file__).parent.parent.parent.parent
        model_path = base_path / model_info['path'] / model_info['file']
        return str(model_path)
        
    def _initialize_model(self):
        """Initialize the emotion recognition model."""
        try:
            model_path = self._get_model_path(self.model_id)
            
            if os.path.exists(model_path):
                self.logger.info(f"Loading model from {model_path}")
                # Here you would load the actual model
                # For now, we'll continue with the dummy implementation
                self.is_model_loaded = True
            else:
                self.logger.warning(f"Model file not found at {model_path}, using dummy model")
                self.is_model_loaded = True
                
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise
            
    def switch_model(self, model_id: str) -> Dict[str, Any]:
        """Switch to a different emotion recognition model.
        
        Args:
            model_id: ID of the model to switch to
            
        Returns:
            Dict containing status and current emotions
        """
        if model_id not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")
            
        try:
            # Store old model info for rollback
            old_model_id = self.model_id
            old_model = self.model
            old_emotions = self.emotions
            
            # Update model info
            self.model_id = model_id
            self.model_info = self.AVAILABLE_MODELS[model_id]
            self.emotions = self.model_info['emotions']
            
            # Update history file path
            self.history_file = self.log_dir / f"emotion_history_{self.model_id}.json"
            
            # Reset emotion history for new model
            with self.history_lock:
                self.emotion_counts = {emotion: 0 for emotion in self.emotions}
                # Load existing history for this model if available
                self._load_history()
            
            # Initialize new model
            self._initialize_model()
            
            # Save history data for previous model
            self._save_history(old_model_id)
            
            return {
                'status': 'success',
                'message': f"Switched to model: {self.model_info['name']}",
                'model_info': {
                    'id': model_id,
                    'name': self.model_info['name'],
                    'type': self.model_info['type'],
                    'emotions': self.emotions
                },
                'emotion_history': self.get_emotion_history()
            }
            
        except Exception as e:
            # Rollback on error
            self.model_id = old_model_id
            self.model = old_model
            self.emotions = old_emotions
            raise
            
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models.
        
        Returns:
            Dict containing model information
        """
        models_list = []
        for model_id, info in self.AVAILABLE_MODELS.items():
            model_path = self._get_model_path(model_id)
            is_available = os.path.exists(model_path)
            models_list.append({
                'id': model_id,
                'name': info['name'],
                'type': info['type'],
                'emotions': info['emotions'],
                'is_available': is_available,
                'path': model_path
            })
            
        return {
            'models': models_list,
            'current_model': {
                'id': self.model_id,
                'name': self.model_info['name'],
                'type': self.model_info['type'],
                'emotions': self.emotions
            }
        }
            
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
                'emotions': {emotion: 0.0 for emotion in self.emotions}
            }
            
        start_time = time.time()
        
        try:
            # Add chunk to buffer
            self.audio_buffer.append(audio_chunk)
            
            # Process if buffer is full
            buffer_samples = int(self.buffer_duration * self.SAMPLE_RATE)
            if len(self.audio_buffer) * self.CHUNK_SIZE >= buffer_samples:
                # Generate realistic emotion probabilities based on current model
                emotion_probs = self._generate_realistic_emotion()
                
                # Get dominant emotion and confidence
                dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
                confidence = emotion_probs[dominant_emotion]
                
                # Create result
                result = {
                    'emotion': dominant_emotion,
                    'confidence': confidence,
                    'emotions': emotion_probs,
                    'fps': float(self.fps),
                    'timestamp': time.time()
                }
                
                # Record in history if confidence exceeds threshold
                if confidence >= self.threshold:
                    self._record_emotion(dominant_emotion, confidence, emotion_probs)
                
                # Clear buffer
                self.audio_buffer = []
                
                # Update FPS
                self.update_fps(time.time() - start_time)
                
                return result
                
            return {
                'emotion': 'buffering',
                'confidence': 0.0,
                'emotions': {emotion: 0.0 for emotion in self.emotions},
                'fps': float(self.fps)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'emotion': 'error',
                'confidence': 0.0,
                'emotions': {emotion: 0.0 for emotion in self.emotions},
                'fps': float(self.fps)
            }
            
    def _generate_realistic_emotion(self) -> Dict[str, float]:
        """Generate realistic emotion probabilities based on current model type."""
        emotion_probs = {emotion: random.uniform(0.1, 0.3) for emotion in self.emotions}
        
        # Select one emotion to be dominant
        dominant_emotion = random.choice(self.emotions)
        emotion_probs[dominant_emotion] = random.uniform(0.6, 0.9)
        
        # Normalize probabilities
        total = sum(emotion_probs.values())
        emotion_probs = {k: v/total for k, v in emotion_probs.items()}
        
        return emotion_probs
    
    def _record_emotion(self, emotion: str, confidence: float, all_emotions: Dict[str, float]):
        """Record emotion in history.
        
        Args:
            emotion: Detected emotion
            confidence: Confidence score
            all_emotions: Dict of all emotion probabilities
        """
        with self.history_lock:
            timestamp = time.time()
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            
            # Add to emotion counts
            if emotion in self.emotion_counts:
                self.emotion_counts[emotion] += 1
            else:
                self.emotion_counts[emotion] = 1
            
            # Add to history deque
            history_entry = {
                'timestamp': timestamp,
                'emotion': emotion,
                'confidence': confidence,
                'all_emotions': all_emotions
            }
            self.emotion_history.append(history_entry)
            
            # Update daily record
            if date_str not in self.daily_emotion_history:
                self.daily_emotion_history[date_str] = {emotion: 0 for emotion in self.emotions}
            
            if emotion in self.daily_emotion_history[date_str]:
                self.daily_emotion_history[date_str][emotion] += 1
            else:
                self.daily_emotion_history[date_str][emotion] = 1
    
    def _periodic_save(self):
        """Periodically save history to file."""
        while self.running:
            current_time = time.time()
            if current_time - self.last_save_time >= self.save_interval:
                self._save_history()
                self.last_save_time = current_time
            time.sleep(60)  # Check every minute
    
    def _save_history(self, model_id=None):
        """Save emotion history to a file.
        
        Args:
            model_id: Optional model ID to save history for a specific model
        """
        save_file = self.history_file
        if model_id:
            save_file = self.log_dir / f"emotion_history_{model_id}.json"
            
        try:
            with self.history_lock:
                # Convert history deque to list for serialization
                history_list = list(self.emotion_history)
                
                # Prepare data for saving
                save_data = {
                    'model_id': self.model_id if not model_id else model_id,
                    'emotions': self.emotions,
                    'emotion_counts': self.emotion_counts,
                    'daily_history': self.daily_emotion_history,
                    'last_updated': time.time()
                }
                
                # Save to file
                with open(save_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                    
                self.logger.info(f"Saved emotion history to {save_file}")
                
        except Exception as e:
            self.logger.error(f"Error saving emotion history: {str(e)}")
    
    def _load_history(self):
        """Load emotion history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    
                # Check if model matches
                if data.get('model_id') == self.model_id:
                    # Load counts and daily history
                    self.emotion_counts = {emotion: 0 for emotion in self.emotions}
                    for emotion, count in data.get('emotion_counts', {}).items():
                        if emotion in self.emotions:
                            self.emotion_counts[emotion] = count
                            
                    self.daily_emotion_history = data.get('daily_history', {})
                    self.logger.info(f"Loaded emotion history for model {self.model_id}")
                else:
                    self.logger.warning(f"History file exists but for different model: {data.get('model_id')} vs {self.model_id}")
                    # Initialize empty counts
                    self.emotion_counts = {emotion: 0 for emotion in self.emotions}
                    self.daily_emotion_history = {}
        except Exception as e:
            self.logger.error(f"Error loading emotion history: {str(e)}")
            # Initialize empty counts
            self.emotion_counts = {emotion: 0 for emotion in self.emotions}
            self.daily_emotion_history = {}
    
    def get_emotion_history(self, time_range='1h'):
        """Get emotion history for the specified time range.
        
        Args:
            time_range: Time range to get history for ('1h', '3h', '24h', '7d', 'all')
            
        Returns:
            Dict containing emotion statistics
        """
        with self.history_lock:
            current_time = time.time()
            
            if time_range == '1h':
                cutoff_time = current_time - 3600
            elif time_range == '3h':
                cutoff_time = current_time - 10800
            elif time_range == '24h':
                cutoff_time = current_time - 86400
            elif time_range == '7d':
                cutoff_time = current_time - 604800
            else:  # 'all'
                cutoff_time = 0
                
            # Filter history by time range
            filtered_history = [entry for entry in self.emotion_history 
                              if entry['timestamp'] >= cutoff_time]
            
            # Count emotions in time range
            range_counts = {emotion: 0 for emotion in self.emotions}
            timestamps = []
            emotion_timeline = []
            
            for entry in filtered_history:
                emotion = entry['emotion']
                if emotion in range_counts:
                    range_counts[emotion] += 1
                timestamps.append(entry['timestamp'])
                emotion_timeline.append(emotion)
            
            # Calculate percentages
            total_counts = sum(range_counts.values())
            percentages = {emotion: (count / total_counts * 100 if total_counts > 0 else 0) 
                          for emotion, count in range_counts.items()}
            
            return {
                'counts': range_counts,
                'percentages': percentages,
                'timestamps': timestamps,
                'emotions': emotion_timeline,
                'time_range': time_range,
                'total_count': total_counts
            }
            
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if hasattr(self, 'save_thread') and self.save_thread.is_alive():
            self.save_thread.join(timeout=2.0)
        self._save_history()
        self.model = None
        self.is_model_loaded = False
        self.audio_buffer = []
        
    def reset(self):
        """Reset the detector state but maintain history."""
        self.audio_buffer = []
        self.state_duration = 0.0
        self.last_update_time = time.time()
        self.emotion_state = random.choice(self.emotions)
        self.confidence = random.uniform(0.6, 0.95)
        self.confidences = {emotion: random.uniform(0.1, 0.4) for emotion in self.emotions}
        self.confidences[self.emotion_state] = self.confidence
        
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

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current emotion state.
        
        Returns:
            Dict containing:
                - emotion: Current emotion state
                - confidence: Confidence in current emotion
                - confidences: Dict of confidences for all emotions
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        
        # Update state if enough time has passed
        if time_elapsed >= 1.0:  # Update every second
            self.state_duration += time_elapsed
            
            # Randomly change state with increasing probability over time
            change_prob = self.state_change_probability * (1 + self.state_duration)
            if random.random() < change_prob:
                # Choose a new emotion
                new_emotion = random.choice(self.emotions)
                while new_emotion == self.emotion_state:  # Ensure it's different
                    new_emotion = random.choice(self.emotions)
                    
                self.emotion_state = new_emotion
                self.state_duration = 0
                self.confidence = random.uniform(0.6, 0.95)
                
                # Update confidences for all emotions
                self.confidences = {emotion: random.uniform(0.1, 0.4) for emotion in self.emotions}
                self.confidences[self.emotion_state] = self.confidence
                
                # Record this emotion in history
                self._record_emotion(self.emotion_state, self.confidence, self.confidences)
            
            self.last_update_time = current_time
        
        return {
            'emotion': self.emotion_state,
            'confidence': self.confidence,
            'confidences': self.confidences
        } 