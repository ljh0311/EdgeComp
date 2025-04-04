"""
Audio Processor module for Baby Monitor System
"""

import logging
import numpy as np
import sounddevice as sd
import eventlet
from eventlet.queue import Queue
from typing import Optional, Callable
import time

# Create a global message queue for callbacks
# This allows the callbacks to be processed in the main eventlet loop
global_message_queue = Queue()

logger = logging.getLogger('AudioProcessor')

class AudioProcessor:
    """
    Audio Processor class for capturing and processing audio
    """
    def __init__(self, channels=1, sample_rate=16000, chunk_size=1024, device=None, emotion_detector=None):
        """
        Initialize the audio processor
        
        Args:
            channels: Number of audio channels
            sample_rate: Sample rate in Hz
            chunk_size: Chunk size in samples
            device: Audio device ID
            emotion_detector: EmotionDetector instance for processing audio
        """
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        self.emotion_detector = emotion_detector
        
        # Audio buffer using eventlet's Queue
        self.audio_queue = Queue(maxsize=10)
        
        # Threading control
        self.stream = None
        self.running = False
        
        # Callback for emotion updates
        self.emotion_callback: Optional[Callable] = None
        
        # Initialize but don't start yet
        self._initialize()
        
        logger.info(f"Audio processor initialized (Channels: {channels}, Sample Rate: {sample_rate} Hz, Chunk Size: {chunk_size})")
    
    def _initialize(self):
        """Initialize audio processing but don't start streams yet"""
        try:
            # Initialize audio stream
            self.stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device,
                callback=self._audio_callback
            )
            logger.info("Audio stream initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio stream: {e}")
            logger.info("Available audio devices:")
            print(sd.query_devices())
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Audio callback function
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time info from sounddevice
            status: Status info
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if not self.running:
            return
        
        # Log audio details occasionally
        current_time = time.time()
        if not hasattr(self, '_last_audio_log_time') or current_time - self._last_audio_log_time > 5.0:
            self._last_audio_log_time = current_time
            logger.debug(f"Audio data: shape={indata.shape}, dtype={indata.dtype}, min={np.min(indata):.4f}, max={np.max(indata):.4f}")
        
        # Put audio data in queue, don't block if queue is full
        try:
            # Use eventlet's put_nowait to avoid blocking the audio callback
            self.audio_queue.put_nowait(indata.copy())
        except eventlet.queue.Full:
            # If queue is full, discard oldest chunk
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(indata.copy())
            except:
                pass
    
    def _process_audio(self):
        """Process audio chunks from the queue"""
        chunk_count = 0
        last_log_time = time.time()
        audio_stats_time = time.time()
        total_audio_chunks = 0
        
        # Log start of audio processing
        logger.info("Audio processing thread started")
        
        while self.running:
            try:
                # Get audio chunk from queue with timeout
                try:
                    chunk = self.audio_queue.get(timeout=0.5)
                except eventlet.queue.Empty:
                    # No audio data available, just yield and continue
                    eventlet.sleep(0.1)
                    continue
                
                # Count chunks for statistics
                chunk_count += 1
                total_audio_chunks += 1
                current_time = time.time()
                
                # Log processing statistics every 10 seconds
                if current_time - last_log_time > 10.0:
                    avg_chunks_per_second = chunk_count / 10.0
                    logger.info(f"Audio processing stats: processed {chunk_count} chunks in the last 10 seconds ({avg_chunks_per_second:.2f}/sec)")
                    
                    # Calculate total audio duration processed
                    total_audio_duration = (total_audio_chunks * self.chunk_size) / self.sample_rate
                    logger.info(f"Total audio processed: {total_audio_chunks} chunks ({total_audio_duration:.2f} seconds)")
                    
                    chunk_count = 0
                    last_log_time = current_time
                
                # Log audio stats every minute
                if current_time - audio_stats_time > 60.0:
                    # Check if we're getting good audio
                    avg_amplitude = np.mean(np.abs(chunk))
                    max_amplitude = np.max(np.abs(chunk))
                    logger.info(f"Audio quality check: avg_amplitude={avg_amplitude:.6f}, max_amplitude={max_amplitude:.6f}")
                    
                    if max_amplitude < 0.01:
                        logger.warning("Audio levels very low - check if microphone is active")
                    elif max_amplitude > 0.9:
                        logger.warning("Audio levels very high - possible clipping")
                    else:
                        logger.info("Audio levels normal")
                    
                    audio_stats_time = current_time
                
                if self.emotion_detector:
                    # Calculate audio energy to check if there's actual sound
                    energy = np.mean(np.square(chunk))
                    peak = np.max(np.abs(chunk))
                    
                    # Log that we're sending audio to the emotion detector
                    logger.debug(f"Sending audio chunk to emotion detector: shape={chunk.shape}, energy={energy:.6f}, peak={peak:.6f}")
                    
                    # Process the chunk in the emotion detector
                    processing_start = time.time()
                    result = self.emotion_detector.process_audio_chunk(chunk)
                    processing_time = time.time() - processing_start
                    
                    # Log the result with timing information
                    if result:
                        emotion = result.get('emotion', 'unknown')
                        confidence = result.get('confidence', 0.0)
                        logger.debug(f"Emotion detection result: {emotion} ({confidence:.4f}) - processing time: {processing_time*1000:.2f}ms")
                        
                        
                        # Instead of using the queue directly, use a thread-safe approach
                        # Add batch_id to ensure proper tracking
                        if 'batch_id' not in result:
                            result['batch_id'] = f"audio_{int(time.time() * 1000) % 10000}"
                            
                        try:
                            # First try the direct callback if it exists
                            if self.emotion_callback:
                                try:
                                    # Call in this thread context instead of sending to queue
                                    self.emotion_callback(result)
                                    logger.debug(f"Called emotion callback directly with result: {emotion}")
                                except Exception as e:
                                    logger.error(f"Error in direct emotion callback: {str(e)}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                            
                            # Then send to the global queue as a backup
                            # Wrap this in a try-except to ensure failures here don't crash the audio processor
                            try:
                                # Use try_put to avoid blocking
                                if not global_message_queue.full():
                                    global_message_queue.put_nowait(('emotion', None, result))
                                    logger.debug(f"Sent emotion result to global queue: {emotion}")
                                else:
                                    logger.warning("Global message queue is full, skipping emotion update")
                            except Exception as e:
                                logger.error(f"Error sending to global queue: {str(e)}")
                        except Exception as e:
                            logger.error(f"Error in emotion result handling: {str(e)}")
                    else:
                        logger.debug(f"No emotion result returned - processing time: {processing_time*1000:.2f}ms")
                
                # Always yield to other greenlets after processing a chunk
                eventlet.sleep(0)
                    
            except Exception as e:
                logger.error(f"Error processing audio chunk: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                # Sleep a bit to avoid tight error loops
                eventlet.sleep(0.5)
                continue
    
    def start(self):
        """Start audio processing"""
        if self.running:
            return
            
        self.running = True
        
        # Start the audio stream
        if self.stream:
            self.stream.start()
            logger.info("Audio stream started")
        
        # Start the processing greenlet
        eventlet.spawn(self._process_audio)
        logger.info("Audio processing started")
    
    def set_emotion_callback(self, callback: Callable):
        """Set callback for emotion updates
        
        Args:
            callback: Function to call with emotion updates
        """
        self.emotion_callback = callback
    
    def get_audio_chunk(self):
        """
        Get an audio chunk from the queue
        
        Returns:
            numpy.ndarray: Audio chunk or None if no data is available
        """
        try:
            return self.audio_queue.get_nowait()
        except eventlet.queue.Empty:
            return None
    
    def stop(self):
        """Stop audio capture and processing"""
        if not self.running:
            return
            
        self.running = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        # Clear the queue
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
        except:
            pass
            
        logger.info("Audio processor stopped")
    
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