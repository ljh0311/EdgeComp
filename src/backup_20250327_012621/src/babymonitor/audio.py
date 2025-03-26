"""
Audio Processor module for Baby Monitor System
"""

import threading
import logging
import numpy as np
import sounddevice as sd
import queue

logger = logging.getLogger('AudioProcessor')

class AudioProcessor:
    """
    Audio Processor class for capturing and processing audio
    """
    def __init__(self, channels=1, sample_rate=16000, chunk_size=1024, device=None):
        """
        Initialize the audio processor
        
        Args:
            channels: Number of audio channels
            sample_rate: Sample rate in Hz
            chunk_size: Chunk size in samples
            device: Audio device ID
        """
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        
        # Audio buffer
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Threading
        self.stream = None
        self.stop_event = threading.Event()
        self.thread = None
        
        # Start audio capture
        self._start_capture()
        
        logger.info(f"Audio processor initialized (Channels: {channels}, Sample Rate: {sample_rate} Hz, Chunk Size: {chunk_size})")
    
    def _start_capture(self):
        """
        Start audio capture
        """
        try:
            # Initialize audio stream
            self.stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device,
                callback=self._audio_callback
            )
            
            # Start stream
            self.stream.start()
            
            logger.info("Audio capture started")
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            logger.info("Available audio devices:")
            print(sd.query_devices())
    
    def _audio_callback(self, indata, frames, time, status):
        """
        Audio callback function
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time: Time info
            status: Status info
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if self.stop_event.is_set():
            return
        
        # Put audio data in queue, don't block if queue is full
        try:
            self.audio_queue.put(indata.copy(), block=False)
        except queue.Full:
            # If queue is full, discard oldest chunk
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put(indata.copy(), block=False)
            except:
                pass
    
    def get_audio_chunk(self):
        """
        Get an audio chunk from the queue
        
        Returns:
            numpy.ndarray: Audio chunk or None if no data is available
        """
        try:
            return self.audio_queue.get(block=False)
        except queue.Empty:
            return None
    
    def stop(self):
        """
        Stop audio capture
        """
        self.stop_event.set()
        
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                pass
        
        logger.info("Audio capture stopped")
    
    def get_sample_rate(self):
        """
        Get the sample rate
        
        Returns:
            int: Sample rate in Hz
        """
        return self.sample_rate
    
    def get_channels(self):
        """
        Get the number of channels
        
        Returns:
            int: Number of channels
        """
        return self.channels 