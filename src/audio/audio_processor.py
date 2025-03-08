"""
Audio Processor Module
==================
Handles real-time audio processing for baby monitoring.
Uses PyAudio for audio capture and interfaces with the HuBERT emotion detector.
"""

import pyaudio
import numpy as np
import torch
import threading
import queue
import time
import logging
from pathlib import Path

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
        
        # Initialize audio processing components
        device_str = config.get('device', 'cpu')
        self.device = torch.device(device_str if isinstance(device_str, str) else 'cpu')
        self._initialize_stream()
        
        self.logger.info("Audio processor initialized")

    def _initialize_stream(self):
        """Initialize the audio stream."""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            format_map = {
                'paFloat32': pyaudio.paFloat32,
                'paInt16': pyaudio.paInt16,
                'paInt32': pyaudio.paInt32
            }
            
            audio_format = format_map.get(self.config.get('format', 'paFloat32'), pyaudio.paFloat32)
            
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
            
            self.stream = self.audio.open(
                format=audio_format,
                channels=self.config.get('channels', 1),
                rate=self.config.get('sample_rate', 16000),
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.config.get('chunk_size', 1024),
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            self.logger.info("Audio stream initialized successfully")
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
                
                # Calculate RMS volume for better sensitivity
                rms = np.sqrt(np.mean(np.square(audio_data)))
                gain = self.config.get('gain', 5.0)  # Adjustable gain factor
                
                # Apply dynamic range compression for better visualization of loud sounds
                audio_data = np.sign(audio_data) * np.log1p(np.abs(audio_data) * gain)
                
                # Normalize to [-1, 1]
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                if not self.audio_queue.full():
                    self.audio_queue.put_nowait((audio_data, rms))
                
                # Send data for visualization if callback is set
                if self.visualization_callback:
                    self.visualization_callback(audio_data)
                    
            return (in_data, pyaudio.paContinue)
        except Exception as e:
            self.logger.error("Error in audio callback: %s", str(e))
            return (None, pyaudio.paAbort)

    def start(self):
        """Start audio processing."""
        try:
            with self.stream_lock:
                if not self.is_running:
                    self.is_running = True
                    if not self.stream or not self.stream.is_active():
                        self._initialize_stream()
                    self.processing_thread = threading.Thread(target=self.process_audio)
                    self.processing_thread.daemon = True
                    self.processing_thread.start()
                    self.logger.info("Audio processing started")
        except Exception as e:
            self.logger.error(f"Error starting audio processing: {str(e)}")
            self.is_running = False
            raise

    def stop(self):
        """Stop audio processing."""
        try:
            with self.stream_lock:
                if self.is_running:
                    self.is_running = False
                    if self.stream:
                        self.stream.stop_stream()
                        self.stream.close()
                        self.stream = None
                    if self.processing_thread:
                        self.processing_thread.join(timeout=2)
                    self.logger.info("Audio processing stopped")
        except Exception as e:
            self.logger.error(f"Error stopping audio processing: {str(e)}")

    def process_audio(self):
        """Process audio data from the queue."""
        while self.is_running:
            try:
                if not self.audio_queue.empty():
                    audio_data, rms = self.audio_queue.get_nowait()
                    
                    # Update visualization if callback is set
                    if self.visualization_callback:
                        self.visualization_callback(audio_data)
                    
                    # Audio processing is now handled by the HuBERT emotion detector
                    # This class only handles audio capture and preprocessing
                
                time.sleep(0.001)  # Small sleep to prevent CPU overload
                
            except Exception as e:
                self.logger.error(f"Error processing audio data: {str(e)}")
                time.sleep(0.1)

    def is_active(self):
        """Check if audio processing is active."""
        return self.is_running and self.stream and self.stream.is_active()

    def __del__(self):
        """Cleanup when object is deleted."""
        try:
            self.stop()
            self.audio.terminate()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 