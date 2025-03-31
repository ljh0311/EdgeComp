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
import yaml
from datetime import datetime
from collections import deque
from .base_detector import BaseDetector
from ..config import config

class EmotionDetector(BaseDetector):
    """Sound-based emotion detector for baby monitoring."""
    
    # Default emotions for backward compatibility
    DEFAULT_EMOTIONS = ['crying', 'laughing', 'babbling', 'silence']
    
    # Audio processing constants - now from config
    SAMPLE_RATE = config.audio.get('sample_rate', 16000)  # Hz
    CHUNK_SIZE = config.audio.get('chunk_size', 4000)    # Samples
    CHANNELS = config.audio.get('channels', 1)         # Mono audio
    DTYPE = np.float32   # Audio data type
    
    # Model definitions with detailed information
    AVAILABLE_MODELS = {
        'basic_emotion': {
            'name': 'Basic Emotion',
            'file': 'model.pt',
            'type': 'basic',
            'emotions': ['crying', 'laughing', 'babbling', 'silence'],
            'path': 'models/emotion/basic_emotion'
        },
        'emotion2': {
            'name': 'Enhanced Emotion Model',
            'file': 'model.pt',
            'type': 'emotion2',
            'emotions': ['happy', 'sad', 'angry', 'neutral', 'crying', 'laughing'],
            'path': 'models/emotion/emotion2'
        },
        'cry_detection': {
            'name': 'Cry Detection',
            'file': 'model.pth',
            'type': 'cry_detection',
            'emotions': ['crying', 'not_crying'],
            'path': 'models/emotion/cry_detection'
        },
        'speechbrain': {
            'name': 'SpeechBrain Emotion',
            'file': 'best_emotion_model.pt',
            'type': 'speechbrain',
            'emotions': ['happy', 'sad', 'angry', 'neutral'],
            'path': 'models/emotion/speechbrain'
        }
    }
    
    def __init__(self, model_id=None, threshold=None, device=None):
        """Initialize the emotion detector.
        
        Args:
            model_id: ID of the model to use (from AVAILABLE_MODELS)
            threshold: Detection confidence threshold
            device: Device to run inference on ('cpu' or 'cuda')
        """
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config', 'emotion_detector.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Error loading config from {config_path}: {e}")
            config = {}
        
        # Get configuration values with defaults
        emotion_config = config.get('model', {})
        threshold = threshold or emotion_config.get('threshold', 0.5)
        device = device or emotion_config.get('device', 'cpu')
        
        super().__init__(threshold=threshold)
        
        # Set audio configuration from config file
        audio_config = config.get('audio', {})
        self.SAMPLE_RATE = audio_config.get('sample_rate', 16000)
        self.CHUNK_SIZE = audio_config.get('chunk_size', 1024)
        self.CHANNELS = audio_config.get('channels', 1)
        self.DTYPE = np.float32
        
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model = None
        # Set default model to basic_emotion
        self.model_id = model_id or emotion_config.get('default_model', 'basic_emotion')
        self.is_model_loaded = False
        self.logger = logging.getLogger(__name__)
        
        # Audio processing setup
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        self.buffer_duration = audio_config.get('buffer_duration', 2.0)  # seconds
        self.stream = None
        self.audio_thread = None
        self.processing_thread = None
        self.is_running = False
        
        # Microphone settings
        self.current_microphone_id = None
        self.microphone_initialized = False
        self.is_muted = True  # Start muted until we confirm mic works
        
        # Get model info with proper fallback
        if self.model_id not in self.AVAILABLE_MODELS:
            self.logger.warning(f"Model {self.model_id} not found, falling back to basic_emotion")
            self.model_id = 'basic_emotion'
        
        self.model_info = self.AVAILABLE_MODELS[self.model_id]
        self.emotions = self.model_info['emotions']
        
        # Initialize history tracking with config values
        system_config = config.get('system', {})
        history_length = system_config.get('history_length', 1000)
        self.emotion_history = deque(maxlen=history_length)
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.daily_emotion_history = {}
        self.history_lock = threading.Lock()
        self.latest_result = None  # Store latest result
        self.latest_result_lock = threading.Lock()  # Lock for thread-safe access
        
        # Set up file paths for persistent storage using config
        logging_config = config.get('logging', {})
        self.log_dir = Path(logging_config.get('file', 'logs/babymonitor.log')).parent
        self.log_dir.mkdir(exist_ok=True)
        self.history_file = self.log_dir / f"emotion_history_{self.model_id}.json"
        
        # Load existing history data if available
        self._load_history()
        
        # Initialize model
        self._initialize_model()
        
        # Initialize audio - now with better error handling
        self._initialize_audio()

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream."""
        if status and status.output_underflow:
            self.logger.warning("Audio buffer underflow")
        if self.is_running and indata is not None and len(indata) > 0:
            try:
                # Only log audio levels at debug level and very rarely
                if self.logger.isEnabledFor(logging.DEBUG) and random.random() < 0.001:  # 0.1% chance
                    rms = np.sqrt(np.mean(np.square(indata)))
                    db = 20 * np.log10(rms) if rms > 0 else -60
                    self.logger.debug(f"Audio input level: {db:.1f} dB")
                
                self.audio_queue.put(indata.copy())
            except Exception as e:
                self.logger.error(f"Error in audio callback: {e}")

    def set_microphone(self, device_id: str) -> bool:
        """Set the microphone device to use."""
        try:
            # Stop any existing audio stream
            if self.is_running:
                self.stop()
            
            # Convert device_id to int if possible (sounddevice prefers int indices)
            try:
                device_index = int(device_id)
            except ValueError:
                device_index = device_id
                
            # Query device info
            try:
                devices = sd.query_devices()
                matching_devices = [d for d in devices if d['index'] == device_index and d['max_input_channels'] > 0]
                
                if not matching_devices:
                    self.logger.error(f"No valid input device found with index {device_id}")
                    return False
                    
                device_info = matching_devices[0]  # Use the first matching input device
                
            except Exception as e:
                self.logger.error(f"Error querying device {device_id}: {e}")
                return False
                
            # Test device by opening a stream briefly
            try:
                test_stream = sd.InputStream(
                    device=device_info['index'],
                    channels=self.CHANNELS,
                    samplerate=self.SAMPLE_RATE,
                    dtype=self.DTYPE
                )
                test_stream.start()
                time.sleep(0.1)  # Wait a bit to ensure device works
                test_stream.stop()
                test_stream.close()
            except Exception as e:
                self.logger.error(f"Error testing device {device_id}: {e}")
                return False
                
            # If we got here, device is working
            self.current_microphone_id = device_info['index']
            self.microphone_initialized = True
            self.is_muted = False
            
            # Start the audio processing if detector is running
            if self.is_running:
                self.start()
                
            self.logger.info(f"Successfully initialized microphone: {device_info['name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting microphone {device_id}: {e}")
            self.microphone_initialized = False
            self.is_muted = True
            return False

    def get_microphone_status(self) -> dict:
        """Get the current microphone status."""
        try:
            if self.current_microphone_id is not None:
                try:
                    device_info = sd.query_devices(self.current_microphone_id)
                    name = device_info.get('name', 'Unknown Device')
                    
                    # Test if the microphone is actually working
                    if self.is_running and not self.audio_queue.empty():
                        status = 'ok'
                    elif self.microphone_initialized and not self.is_muted:
                        status = 'initialized'
                    else:
                        status = 'error'
                        
                except Exception as e:
                    self.logger.error(f"Error querying device {self.current_microphone_id}: {e}")
                    name = 'Device Error'
                    status = 'error'
            else:
                name = 'No Device Selected'
                status = 'error'
            
            return {
                'initialized': self.microphone_initialized,
                'is_muted': self.is_muted,
                'device_id': self.current_microphone_id,
                'name': name,
                'status': status
            }
        except Exception as e:
            self.logger.error(f"Error getting microphone status: {e}")
            return {
                'initialized': False,
                'is_muted': True,
                'device_id': None,
                'name': 'Error Getting Status',
                'status': 'error',
                'error': str(e)
            }

    def get_audio_level(self) -> float:
        """Get current audio level in dB."""
        mic_status = self.get_microphone_status()
        if not mic_status['initialized'] or mic_status['is_muted']:
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
        """Process audio chunk and detect emotion."""
        if not self.is_model_loaded:
            return {
                'emotion': 'error',
                'confidence': 0.0,
                'emotions': {emotion: 0.0 for emotion in self.emotions},
                'status': 'error',
                'error_message': 'Model not loaded'
            }

        if audio_chunk is None or len(audio_chunk) == 0:
            return {
                'emotion': 'no_data',
                'confidence': 0.0,
                'emotions': {emotion: 0.0 for emotion in self.emotions},
                'status': 'no_data'
            }
            
        # Log audio chunk statistics only at debug level and rarely
        if self.logger.isEnabledFor(logging.DEBUG) and random.random() < 0.001:  # 0.1% chance
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            db = 20 * np.log10(rms) if rms > 0 else -60
            self.logger.debug(f"Processing audio chunk - Level: {db:.1f} dB, Shape: {audio_chunk.shape}")
        
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
                audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                try:
                    # Process based on model type
                    model_type = self.model_info['type']
                    
                    if model_type == 'basic':
                        # Convert to tensor
                        audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
                        audio_tensor = audio_tensor.to(self.device)
                        
                        # Run inference
                        with torch.no_grad():
                            outputs = self.model(audio_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            
                        # Convert to emotion probabilities dict
                        emotion_probs = {
                            emotion: float(prob)
                            for emotion, prob in zip(self.emotions, probabilities[0])
                        }
                        
                    elif model_type == 'speechbrain':
                        try:
                            from speechbrain.inference import EncoderClassifier
                            
                            # Load the model directly without hyperparameters
                            model_dir = os.path.dirname(model_path)
                            self.model = EncoderClassifier.from_hparams(
                                source=model_dir,
                                run_opts={"device": str(self.device)}
                            )
                            
                            self.model.eval()
                            self.model.to(self.device)
                            self.emotions = self.model_info['emotions']
                            self.is_model_loaded = True
                            self.logger.info(f"Successfully loaded speechbrain model from {model_path}")
                            
                        except Exception as e:
                            self.logger.error(f"Error loading speechbrain model: {str(e)}")
                            self.logger.error(traceback.format_exc())
                            # Try fallback to cry detection model
                            if self.model_id != 'cry_detection':
                                self.logger.info("Attempting to fall back to cry detection model...")
                                self.model_id = 'cry_detection'
                                self.model_info = self.AVAILABLE_MODELS['cry_detection']
                                return self._initialize_model()
                            else:
                                self.is_model_loaded = False
                                return
                    
                    else:  # Default processing for other model types
                        audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
                        audio_tensor = audio_tensor.to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(audio_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            
                        emotion_probs = {
                            emotion: float(prob)
                            for emotion, prob in zip(self.emotions, probabilities[0])
                        }
                    
                    # Get dominant emotion and confidence
                    dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
                    confidence = emotion_probs[dominant_emotion]
                    
                    # Record emotion if confidence exceeds threshold
                    if confidence >= self.threshold:
                        self._record_emotion(dominant_emotion, confidence, emotion_probs)
                    
                    # Create result
                    result = {
                        'emotion': dominant_emotion,
                        'confidence': confidence,
                        'emotions': emotion_probs,
                        'fps': float(self.fps),
                        'timestamp': time.time(),
                        'status': 'success'
                    }
                    
                    # Store the latest result
                    with self.latest_result_lock:
                        self.latest_result = result
                    
                    # Clear buffer
                    self.audio_buffer = []
                    
                    # Update FPS
                    self.update_fps(time.time() - start_time)
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Error in model inference: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    return {
                        'emotion': 'error',
                        'confidence': 0.0,
                        'emotions': {emotion: 0.0 for emotion in self.emotions},
                        'status': 'error',
                        'error_message': str(e)
                    }
            
            return {
                'emotion': 'buffering',
                'confidence': 0.0,
                'emotions': {emotion: 0.0 for emotion in self.emotions},
                'fps': float(self.fps),
                'status': 'buffering'
            }
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'emotion': 'error',
                'confidence': 0.0,
                'emotions': {emotion: 0.0 for emotion in self.emotions},
                'status': 'error',
                'error_message': str(e)
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
            
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found at {model_path}")
                self.is_model_loaded = False
                return
            
            # Initialize cry detection model
            try:
                from .models.cry_detection import CryDetectionModel
                self.model = CryDetectionModel()
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model = self.model.to(self.device)
                self.is_model_loaded = True
                self.logger.info(f"Successfully loaded model from {model_path}")
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.is_model_loaded = False
                
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.is_model_loaded = False
            
    def _initialize_audio(self):
        """Initialize audio settings and try to set up default microphone."""
        try:
            # Get list of available input devices
            devices = sd.query_devices()
            input_devices = [dev for dev in devices if dev['max_input_channels'] > 0]
            
            if not input_devices:
                self.logger.error("No input devices found")
                self.microphone_initialized = False
                self.is_muted = True
                return
            
            # Try to find the best microphone in this order:
            # 1. System default input device
            # 2. Any device with "microphone" in the name
            # 3. First available input device
            
            default_device = sd.default.device[0]
            mic_devices = [dev for dev in input_devices if 'mic' in dev['name'].lower()]
            
            # Try system default first
            if default_device is not None:
                success = self.set_microphone(str(default_device))
                if success:
                    return
                    
            # Try microphone-named devices next
            if mic_devices:
                for mic in mic_devices:
                    success = self.set_microphone(str(mic['index']))
                    if success:
                        return
                    
            # Finally, try any input device
            for device in input_devices:
                success = self.set_microphone(str(device['index']))
                if success:
                    return
                
            self.logger.error("Could not initialize any microphone")
            self.microphone_initialized = False
            self.is_muted = True
            
        except Exception as e:
            self.logger.error(f"Could not initialize microphone: {e}")
            self.logger.error(traceback.format_exc())
            self.microphone_initialized = False
            self.is_muted = True

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
        
        # Clear audio buffer and queue
        self.audio_buffer = []
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
            
        # Reset microphone state
        self.microphone_initialized = False
        self.is_muted = True
        self.current_microphone_id = None
        
        # Save any pending data
        self._save_history()
        
        # Clear model
        self.model = None
        self.is_model_loaded = False
        
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

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Get the latest emotion detection result.
        
        Returns:
            Dict containing the latest emotion detection result or None if no result available
        """
        with self.latest_result_lock:
            return self.latest_result 

    def start(self):
        """Start the detector."""
        if not self.is_running:
            try:
                if not self.microphone_initialized:
                    self.logger.error("Cannot start: Microphone not initialized")
                    return False
                    
                self.logger.info(f"Starting audio stream with device {self.current_microphone_id}")
                self.stream = sd.InputStream(
                    device=self.current_microphone_id,
                    channels=self.CHANNELS,
                    samplerate=self.SAMPLE_RATE,
                    dtype=self.DTYPE,
                    callback=self.audio_callback,
                    blocksize=self.CHUNK_SIZE
                )
                self.stream.start()
                self.is_running = True
                self.start_audio_processing()
                self.logger.info("Audio stream started successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Error starting audio stream: {e}")
                self.logger.error(traceback.format_exc())
                self.is_running = False
                return False
        return True 

    def stop(self):
        """Stop the detector and cleanup resources."""
        self.is_running = False
        
        # Stop audio stream
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            except Exception as e:
                self.logger.error(f"Error stopping audio stream: {e}")
        
        # Wait for processing thread to finish
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        
        # Clear audio buffer and queue
        self.audio_buffer = []
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break