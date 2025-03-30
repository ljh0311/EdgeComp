import pyaudio
import numpy as np
import logging
from .devices import get_available_microphones, test_microphone

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio processing for the Baby Monitor system."""
    
    def __init__(self, device=None, sample_rate=44100, chunk_size=1024, channels=1):
        """Initialize the audio processor.
        
        Args:
            device: ID of the audio device to use. If None, uses default device.
            sample_rate (int): Sample rate in Hz
            chunk_size (int): Number of frames per buffer
            channels (int): Number of audio channels
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.current_device = None
        self.audio_buffer = []
        self.max_buffer_size = 10  # Keep last 10 chunks
        
        # Start the stream if a device is provided
        if device is not None:
            self.start(device)
    
    def start(self, device_id=None):
        """Start audio processing with the specified device.
        
        Args:
            device_id: ID of the audio device to use. If None, uses default device.
        """
        try:
            # Stop any existing stream
            if hasattr(self, 'stream') and self.stream:
                self.stop()
            
            # Get device info
            if device_id is None:
                device_info = self.audio.get_default_input_device_info()
            else:
                device_info = self.audio.get_device_info_by_index(int(device_id))
            
            # Open stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_info['index'],
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.current_device = device_info
            logger.info(f"Started audio processing with device: {device_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting audio processing: {str(e)}")
            return False
    
    def stop(self):
        """Stop audio processing."""
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream: {str(e)}")
            finally:
                self.stream = None
                self.current_device = None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Process audio data from the stream."""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Add to buffer
            self.audio_buffer.append(audio_data)
            
            # Keep buffer size limited
            while len(self.audio_buffer) > self.max_buffer_size:
                self.audio_buffer.pop(0)
            
            return (in_data, pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"Error in audio callback: {str(e)}")
            return (in_data, pyaudio.paAbort)
    
    def get_audio_level(self):
        """Get the current audio level in dB."""
        try:
            if not self.audio_buffer:
                return -60  # Silent
            
            # Calculate RMS of the most recent audio data
            audio_data = np.concatenate(self.audio_buffer)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            
            # Convert to dB
            db_level = 20 * np.log10(rms + 1e-10)  # Add small value to prevent log(0)
            
            # Clamp between -60 and 0 dB
            return max(-60, min(0, float(db_level)))
            
        except Exception as e:
            logger.error(f"Error calculating audio level: {str(e)}")
            return -60
    
    def get_available_devices(self):
        """Get list of available audio devices."""
        return get_available_microphones()
    
    def test_device(self, device_id):
        """Test a specific audio device."""
        return test_microphone(device_id)
    
    def __del__(self):
        """Clean up resources."""
        self.stop()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate() 