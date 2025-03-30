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
import sounddevice as sd
import threading
import queue
import traceback
from datetime import datetime
from collections import deque
from .base_detector import BaseDetector

class EmotionDetector(BaseDetector):
    """Sound-based emotion detector for baby monitoring."""
    
    # Default emotions for backward compatibility
    DEFAULT_EMOTIONS = ['crying', 'laughing', 'babbling', 'silence']
    
    # Audio processing constants
    SAMPLE_RATE = 16000  # Hz
    CHUNK_SIZE = 4000    # Samples
    CHANNELS = 1         # Mono audio
    DTYPE = np.float32   # Audio data type
    
    # Model definitions with detailed information
    AVAILABLE_MODELS = {
        'wav2vec2': {
            'name': 'Wav2Vec2 Emotion Model',
            'description': 'Advanced speech recognition model for emotion detection',
            'file': 'model.safetensors',
            'config_file': 'config.json',
            'type': 'wav2vec2',
            'emotions': ['crying', 'laughing', 'babbling', 'silence', 'noise'],
            'path': 'models/emotion/wav2vec2',
            'min_confidence': 0.6,
            'supported_formats': ['.safetensors'],
            'features': ['noise_resistant', 'high_accuracy']
        },
        'speechbrain': {
            'name': 'SpeechBrain Model',
            'description': 'Neural speech processing model with advanced features',
            'file': 'best_emotion_model.pt',
            'config_file': 'config.json',
            'type': 'speechbrain',
            'emotions': ['crying', 'laughing', 'babbling', 'silence', 'noise', 'speech'],
            'path': 'models/emotion/speechbrain',
            'min_confidence': 0.65,
            'supported_formats': ['.pt'],
            'features': ['speech_recognition', 'noise_filtering'],
            'variants': {
                'base': 'emotion_model.pt',
                'hubert': 'hubert-base-ls960_emotion.pt',
                'best': 'best_emotion_model.pt'
            }
        },
        'emotion2': {
            'name': 'Enhanced Emotion Model',
            'description': 'Advanced emotion detection with TFLite format',
            'file': 'baby_cry_classifier_enhanced.tflite',
            'label_file': 'label_encoder_enhanced.pkl',
            'type': 'emotion2',
            'emotions': ['crying', 'laughing', 'babbling', 'silence', 'coughing', 'sneezing'],
            'path': 'models/emotion/emotion2',
            'min_confidence': 0.6,
            'supported_formats': ['.tflite'],
            'features': ['enhanced_detection', 'mobile_optimized']
        },
        'cry_detection': {
            'name': 'Cry Detection Model',
            'description': 'Specialized model for cry detection',
            'file': 'cry_detection_model.pth',
            'config_file': 'config.json',
            'type': 'cry_detection',
            'emotions': ['crying', 'not_crying'],
            'path': 'models/emotion/cry_detection',
            'min_confidence': 0.7,
            'supported_formats': ['.pth'],
            'features': ['specialized_crying']
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
        # Set default model to 'speechbrain' since it's the most comprehensive
        self.model_id = model_id or 'speechbrain'
        self.is_model_loaded = False
        self.logger = logging.getLogger(__name__)
        
        # Audio processing setup
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        self.buffer_duration = 2.0  # seconds
        self.stream = None
        self.audio_thread = None
        self.processing_thread = None
        self.is_running = False
        
        # Microphone settings
        self.current_microphone_id = None
        self.microphone_initialized = False
        self.is_muted = False  # Track if current microphone is muted
        
        # Get model info with proper fallback
        if self.model_id not in self.AVAILABLE_MODELS:
            self.logger.warning(f"Model {self.model_id} not found, falling back to speechbrain")
            self.model_id = 'speechbrain'
        
        self.model_info = self.AVAILABLE_MODELS[self.model_id]
        self.emotions = self.model_info['emotions']
        
        # Initialize history tracking
        self.emotion_history = deque(maxlen=1000)
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.daily_emotion_history = {}
        self.history_lock = threading.Lock()
        
        # Set up file paths for persistent storage
        self.log_dir = Path(os.path.expanduser('~')) / 'babymonitor_logs'
        self.log_dir.mkdir(exist_ok=True)
        self.history_file = self.log_dir / f"emotion_history_{self.model_id}.json"
        
        # Load existing history data if available
        self._load_history()
        
        # Initialize model
        self._initialize_model()
        
        # Try to initialize default microphone
        try:
            import sounddevice as sd
            default_device = sd.default.device[0]  # Get default input device
            if default_device is not None:
                self.set_microphone(str(default_device))
        except Exception as e:
            self.logger.warning(f"Could not initialize default microphone: {e}")
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream."""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        if self.is_running:
            self.audio_queue.put(indata.copy())
            
    def set_microphone(self, mic_id: str) -> bool:
        """Set the active microphone.
        
        Args:
            mic_id: ID of the microphone to use
            
        Returns:
            bool: True if microphone was set successfully
        """
        try:
            # Stop existing audio processing
            was_running = self.is_running
            self.is_running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            # Stop existing stream if any
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    self.logger.warning(f"Error closing existing stream: {e}")
            
            # Convert mic_id to int for sounddevice
            mic_id_int = int(mic_id)
            
            # Get device info to check if it's muted/inactive
            device_info = sd.query_devices(mic_id_int)
            
            # Check if device is an input device
            if device_info['max_input_channels'] == 0:
                self.logger.error(f"Device {mic_id} is not an input device")
                return False
            
            # Update muted state based on device info
            self.is_muted = device_info.get('is_muted', False) or device_info['max_input_channels'] == 0
            
            # Create new audio stream
            self.stream = sd.InputStream(
                device=mic_id_int,
                channels=self.CHANNELS,
                samplerate=self.SAMPLE_RATE,
                dtype=self.DTYPE,
                blocksize=self.CHUNK_SIZE,
                callback=self.audio_callback
            )
            
            # Store the microphone ID
            self.current_microphone_id = mic_id
            
            # Reset audio buffer and queue
            self.audio_buffer = []
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Mark microphone as initialized
            self.microphone_initialized = True
            
            # Start the stream
            self.stream.start()
            
            # Always start audio processing when setting a new microphone
            self.is_running = True
            self.start_audio_processing()
            
            self.logger.info(f"Successfully set microphone to {mic_id} (Name: {device_info.get('name', 'Unknown')})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting microphone: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Reset state on error
            self.microphone_initialized = False
            self.is_muted = True
            self.is_running = False
            return False

    def get_audio_level(self) -> float:
        if self.is_muted or not self.is_running or not self.microphone_initialized:
            return -60.0  # Return silence level for muted/uninitialized microphone
        
        try:
            # Try to get latest audio data from the queue
            try:
                latest_audio = None
                # Get the most recent data without blocking
                while not self.audio_queue.empty():
                    latest_audio = self.audio_queue.get_nowait()
                
                if latest_audio is None and self.audio_buffer:
                    latest_audio = self.audio_buffer[-1]
                
                if latest_audio is None:
                    return -60.0
                
                # Ensure audio data is flattened
                if len(latest_audio.shape) > 1:
                    latest_audio = latest_audio.flatten()
                
                # Calculate RMS value
                rms = np.sqrt(np.mean(np.square(latest_audio)))
                
                # Convert to dB (avoid log of 0)
                if rms > 0:
                    db = 20 * np.log10(rms)
                    # Clamp between -60 and -10 dB
                    return np.clip(db, -60.0, -10.0)
                
            except queue.Empty:
                pass
            
            return -60.0
            
        except Exception as e:
            self.logger.error(f"Error calculating audio level: {str(e)}")
            return -60.0
        
    def _process_audio_thread(self):
        """Background thread for processing audio data."""
        while self.is_running:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=1.0)
                
                # Process the audio data
                if len(audio_data) > 0:
                    result = self.process_audio(audio_data)
                    
                    # Log the result if confidence exceeds threshold
                    if result['confidence'] >= self.threshold:
                        self.logger.debug(f"Detected emotion: {result['emotion']} "
                                        f"(confidence: {result['confidence']:.2f})")
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in audio processing thread: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                time.sleep(1)  # Prevent tight loop on error
                
    def process_audio(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Process audio chunk and detect emotion.
        
        Args:
            audio_chunk: Numpy array containing audio samples
            
        Returns:
            Dict containing:
                - emotion: Detected emotion label
                - confidence: Confidence score
                - emotions: Dict of all emotion probabilities
                - fps: Current processing FPS
                - timestamp: Processing timestamp
        """
        if audio_chunk is None or len(audio_chunk) == 0:
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
            if sum(len(chunk) for chunk in self.audio_buffer) >= buffer_samples:
                # Concatenate audio chunks
                audio_data = np.concatenate(self.audio_buffer)
                
                # Ensure we have the right amount of data
                if len(audio_data) > buffer_samples:
                    audio_data = audio_data[:buffer_samples]
                
                # Normalize audio
                audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Extract features using the model's feature extractor
                inputs = self.feature_extractor(
                    audio_data, 
                    sampling_rate=self.SAMPLE_RATE,
                    return_tensors="pt"
                )
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Convert to emotion probabilities dict
                emotion_probs = {
                    emotion: float(prob)
                    for emotion, prob in zip(self.emotions, probabilities[0])
                }
                
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
    
    def _initialize_model(self):
        """Initialize the emotion recognition model."""
        try:
            model_path = self._get_model_path(self.model_id)
            config_path = None
            
            # Get config file path if specified
            if 'config_file' in self.model_info:
                config_dir = os.path.dirname(model_path)
                config_path = os.path.join(config_dir, self.model_info['config_file'])
            
            # Check if model file exists
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found at {model_path}")
                self.is_model_loaded = False
                return
                
            # Initialize model based on type
            model_type = self.model_info['type']
            if model_type == 'wav2vec2':
                from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(os.path.dirname(model_path))
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(os.path.dirname(model_path))
            elif model_type == 'speechbrain':
                from speechbrain.inference import EncoderClassifier
                import yaml
                
                # Load config file
                if not config_path or not os.path.exists(config_path):
                    self.logger.error(f"Config file not found at {config_path}")
                    self.is_model_loaded = False
                    return
                    
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Create a basic hparams structure if modules is missing
                if 'modules' not in config:
                    config['modules'] = {
                        'encoder': {'select': 'CNN'},
                        'classifier': {'select': 'MLP'}
                    }
                
                try:
                    self.model = EncoderClassifier.from_hparams(
                        source=os.path.dirname(model_path),
                        hparams_file=config_path,
                        savedir=os.path.dirname(model_path)
                    )
                    self.feature_extractor = getattr(self.model, 'mods', {}).get('preprocessor', None)
                except Exception as e:
                    self.logger.error(f"Error loading SpeechBrain model: {str(e)}")
                    self.logger.error(f"Config contents: {config}")
                    raise
                    
            elif model_type == 'emotion2':
                import tensorflow as tf
                import pickle
                self.model = tf.lite.Interpreter(model_path=model_path)
                self.model.allocate_tensors()
                # Load label encoder if available
                label_file = os.path.join(os.path.dirname(model_path), self.model_info['label_file'])
                if os.path.exists(label_file):
                    with open(label_file, 'rb') as f:
                        self.label_encoder = pickle.load(f)
            elif model_type == 'cry_detection':
                self.model = torch.jit.load(model_path)
                self.model.eval()
                
            if hasattr(self, 'model') and self.model is not None:
                self.model = self.model.to(self.device)
                self.is_model_loaded = True
                self.logger.info(f"Successfully loaded {model_type} model from {model_path}")
            else:
                self.logger.error(f"Failed to load {model_type} model")
                self.is_model_loaded = False
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            self.is_model_loaded = False
            
    def switch_model(self, model_id: str) -> Dict[str, Any]:
        """Switch to a different emotion recognition model.
        
        Args:
            model_id: ID of the model to switch to
            
        Returns:
            Dict containing status and current emotions
        """
        if model_id not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")
            
        # Store old model info for rollback
        old_model_id = self.model_id
        old_model = self.model
        old_emotions = self.emotions
        
        try:
            # Update model and history
            self.model_id = model_id
            self.model_info = self.AVAILABLE_MODELS[model_id]
            self.emotions = self.model_info['emotions']
            self.history_file = self.log_dir / f"emotion_history_{self.model_id}.json"
            
            # Reset and reload history
            with self.history_lock:
                self.emotion_counts = {emotion: 0 for emotion in self.emotions}
                self._load_history()
            
            self._initialize_model()
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
            
        except Exception:
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
            
    def _get_model_path(self, model_id: str) -> str:
        """Get the full path for a model."""
        model_info = self.AVAILABLE_MODELS.get(model_id)
        if not model_info:
            raise ValueError(f"Unknown model ID: {model_id}")
            
        # Get the absolute path to the workspace root
        workspace_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
        model_path = workspace_root / model_info['path'] / model_info['file']
        return str(model_path.resolve())
        
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
        self.is_running = False
        
        # Stop audio stream
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                self.logger.error(f"Error closing audio stream: {str(e)}")
        
        # Wait for processing thread to finish
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2.0)
        
        if hasattr(self, 'save_thread') and self.save_thread.is_alive():
            self.save_thread.join(timeout=2.0)
            
        self._save_history()
        self.model = None
        self.is_model_loaded = False
        self.audio_buffer = []
        
    def reset(self):
        """Reset the detector state but maintain history."""
        self.audio_buffer = []
        self.audio_queue = queue.Queue()
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
        # Return empty state since we're not generating fake data anymore
        return {
            'emotion': 'unknown',
            'confidence': 0.0,
            'confidences': {emotion: 0.0 for emotion in self.emotions}
        }

    def start_audio_processing(self):
        """Start the audio processing thread."""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_audio_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.logger.info("Started audio processing thread") 