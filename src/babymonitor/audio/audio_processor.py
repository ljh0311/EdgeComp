"""
Audio Processor Module
==================
Handles real-time audio processing and sound classification for baby monitoring.
Uses PyAudio for audio capture and Wav2Vec2 for sound classification.
"""

import pyaudio
import numpy as np
import torch
import torch.nn.functional as F
import threading
import queue
import time
import logging
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

class AudioProcessor:
    def __init__(self, config, alert_callback=None):
        """Initialize the audio processor with configuration and callback."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.alert_callback = alert_callback
        self.is_running = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.processing_thread = None
        self.audio_queue = queue.Queue(maxsize=100)
        self.visualization_callback = None
        self.stream_lock = threading.Lock()
        
        # Initialize state
        self.last_alert_time = 0
        
        # Audio parameters
        self.chunk_size = self.config.get('chunk_size', 1024)
        self.sample_rate = self.config.get('sample_rate', 16000)  # Wav2Vec2 requires 16kHz
        self.channels = self.config.get('channels', 1)
        self.format = self.config.get('format', 'paFloat32')
        self.use_callback = self.config.get('use_callback', False)  # Default to blocking mode
        
        # Initialize audio processing components
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Sound classification labels and thresholds
        self.emotion_labels = ['cry', 'laugh', 'babble', 'background']
        self.thresholds = {
            'cry': 0.6,
            'laugh': 0.5,
            'babble': 0.5
        }
        
        self.setup_model()
        
        self.logger.info("Audio processor initialized")

    def setup_model(self):
        """Setup the Wav2Vec2 model for audio classification."""
        try:
            # Initialize Wav2Vec2 model
            model_name = "facebook/wav2vec2-base"
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.emotion_labels),
                ignore_mismatched_sizes=True
            )
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("Wav2Vec2 model initialized successfully")
            
        except Exception as e:
            self.logger.error("Error setting up Wav2Vec2 model: %s", str(e))
            raise

    def _initialize_stream(self):
        """Initialize the audio stream."""
        try:
            with self.stream_lock:
                if self.stream is not None:
                    self.stream.stop_stream()
                    self.stream.close()
            
            format_map = {
                'paFloat32': pyaudio.paFloat32,
                'paInt16': pyaudio.paInt16,
                'paInt32': pyaudio.paInt32
            }
            
            audio_format = format_map.get(self.format, pyaudio.paFloat32)
            
            # Find the SteelSeries Sonar microphone
            device_index = None
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if "SteelSeries Sonar - Microphone" in device_info["name"] and device_info["maxInputChannels"] > 0:
                    device_index = i
                    self.logger.info(f"Found SteelSeries Sonar microphone at index {i}")
                    break
            
            if device_index is None:
                self.logger.warning("SteelSeries Sonar microphone not found, using default input device")
            
            stream_kwargs = {
                'format': audio_format,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'input_device_index': device_index,
                'frames_per_buffer': self.chunk_size
            }
            
            # Add callback only if using callback mode
            if self.use_callback:
                stream_kwargs['stream_callback'] = self._audio_callback
            
            with self.stream_lock:
                self.stream = self.audio.open(**stream_kwargs)
                if self.use_callback:
                    self.stream.start_stream()
            
            self.logger.info(f"Audio stream initialized successfully in {'callback' if self.use_callback else 'blocking'} mode")
            
        except Exception as e:
            self.logger.error(f"Error initializing audio stream: {str(e)}")
            raise

    def set_visualization_callback(self, callback):
        """Set callback for visualization updates."""
        self.visualization_callback = callback

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream to process incoming audio data."""
        try:
            if self.is_running:
                # Convert to float32 and normalize
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                
                # Calculate RMS volume
                rms = np.sqrt(np.mean(np.square(audio_data)))
                
                # Apply gain with soft clipping for better dynamics
                gain = self.config.get('gain', 2.0)
                audio_data = np.tanh(audio_data * gain)
                
                if not self.audio_queue.full():
                    self.audio_queue.put_nowait((audio_data, rms))
                
                # Send data for visualization if callback is set
                if self.visualization_callback:
                    self.visualization_callback(audio_data)
                    
            return (in_data, pyaudio.paContinue)
        except Exception as e:
            self.logger.error("Error in audio callback: %s", str(e))
            return (None, pyaudio.paAbort)

    def process_audio(self, audio_data):
        """Process audio data for visualization and sound detection."""
        try:
            if audio_data is None or len(audio_data) == 0:
                return None, None

            # Calculate RMS value for decibel calculation
            rms = np.sqrt(np.mean(np.square(audio_data)))
            
            # Convert to decibels with proper reference
            if rms > 0:
                db = 20 * np.log10(rms) - 20
            else:
                db = -80  # Minimum dB level
            
            # Apply calibration offset
            calibration_offset = self.config.get('calibration_offset', 0)
            db = db + calibration_offset

            # Update visualization if callback is set
            if self.visualization_callback:
                # Normalize audio data for visualization
                normalized_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
                self.visualization_callback(normalized_data)

            # Process audio with Wav2Vec2 model if enough samples
            if len(audio_data) >= self.sample_rate:  # At least 1 second of audio
                try:
                    with torch.no_grad():
                        # Prepare input for Wav2Vec2
                        inputs = self.processor(
                            audio_data, 
                            sampling_rate=self.sample_rate,
                            return_tensors="pt",
                            padding=True
                        ).input_values.to(self.device)
                        
                        # Get model predictions
                        outputs = self.model(inputs)
                        probs = F.softmax(outputs.logits, dim=-1)
                        predicted_id = torch.argmax(probs, dim=-1).item()
                        confidence = probs[0][predicted_id].item()
                        
                        # Check if confidence exceeds threshold
                        emotion = self.emotion_labels[predicted_id]
                        if emotion in self.thresholds and confidence > self.thresholds[emotion]:
                            if self.alert_callback:
                                self.alert_callback('warning', f'Detected {emotion} sound ({confidence:.2f} confidence)')
                
                except Exception as e:
                    self.logger.error(f"Error in Wav2Vec2 processing: {str(e)}")

            # Check for loud sounds
            threshold = self.config.get('LOUD_THRESHOLD', -20)  # dB
            if db > threshold:
                if self.alert_callback:
                    self.alert_callback('warning', f'Loud sound detected: {db:.1f} dB')

            return db, audio_data

        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            return None, None

    def start(self):
        """Start audio processing."""
        if self.is_running:
            return

        try:
            self.is_running = True
            self._initialize_stream()  # Initialize stream when starting
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio_stream)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.logger.info("Audio processing started")
            
        except Exception as e:
            self.logger.error(f"Error starting audio processing: {str(e)}")
            self.is_running = False
            raise

    def stop(self):
        """Stop audio processing."""
        if not self.is_running:
            return

        try:
            self.is_running = False
            
            # Stop and close the stream with proper locking
            with self.stream_lock:
                if self.stream is not None:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
            
            # Wait for processing thread to finish
            if self.processing_thread is not None:
                self.processing_thread.join(timeout=2.0)
            
            # Clean up PyAudio
            if self.audio is not None:
                self.audio.terminate()
                self.audio = None
                
            self.logger.info("Audio processing stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping audio processing: {str(e)}")
            raise

    def _process_audio_stream(self):
        """Process audio stream in a separate thread."""
        try:
            while self.is_running:
                try:
                    if self.use_callback:
                        # In callback mode, just sleep and let the callback handle the data
                        time.sleep(0.001)
                        continue
                    
                    # In blocking mode, read the data directly
                    with self.stream_lock:
                        if self.stream is None or not self.is_running:
                            break
                        try:
                            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                        except IOError as e:
                            if e.errno == -9981:  # Input overflowed
                                self.logger.warning("Audio input overflow")
                                continue
                            else:
                                raise
                    
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    
                    # Process the audio data
                    db, processed_data = self.process_audio(audio_data)
                    
                    # If processing was successful and we have web app integration
                    if db is not None and hasattr(self, 'web_app'):
                        self.web_app.emit_audio_data({
                            'waveform': processed_data[0].tolist() if processed_data is not None else [],
                            'decibel': db
                        })
                        
                except Exception as e:
                    if self.is_running:  # Only log if we're still supposed to be running
                        self.logger.error(f"Error in audio processing loop: {str(e)}")
                    time.sleep(0.1)  # Prevent tight loop on error
                    
        except Exception as e:
            if self.is_running:  # Only log if we're still supposed to be running
                self.logger.error(f"Fatal error in audio processing thread: {str(e)}")
            self.is_running = False

    def _process_audio(self):
        """Process audio data from the queue."""
        while self.is_running:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                    
                    # Update visualization if callback is set
                    if self.visualization_callback:
                        self.visualization_callback(audio_data)
                    
                    # Process audio for alerts (implement your logic here)
                    if self.alert_callback and self._should_alert(audio_data):
                        self.alert_callback("High audio level detected", "warning")
                
                time.sleep(0.001)  # Small sleep to prevent CPU overload
                
            except Exception as e:
                self.logger.error(f"Error processing audio data: {str(e)}")
                time.sleep(0.1)  # Longer sleep on error

    def _should_alert(self, audio_data):
        """Check if audio level should trigger an alert."""
        try:
            # Simple threshold-based detection
            audio_level = np.abs(audio_data).mean()
            return audio_level > self.config.get('alert_threshold', 0.1)
        except Exception as e:
            self.logger.error(f"Error checking audio level: {str(e)}")
            return False

    def is_active(self):
        """Check if audio processing is active."""
        return self.is_running and self.stream and self.stream.is_active() 

    def __del__(self):
        """Cleanup when object is deleted."""
        try:
            self.stop()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 