"""
Baby Monitor System
"""

import os
import cv2
import time
import threading
import logging
import numpy as np
import psutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BabyMonitorSystem')

class BabyMonitorSystem:
    """
    Main system class for the Baby Monitor
    """
    def __init__(self, camera=None, audio_processor=None, person_detector=None, emotion_detector=None, web_interface=None):
        """
        Initialize the Baby Monitor System
        
        Args:
            camera: Camera instance
            audio_processor: AudioProcessor instance
            person_detector: PersonDetector instance
            emotion_detector: EmotionDetector instance
            web_interface: BabyMonitorWeb instance
        """
        self.camera = camera
        self.audio_processor = audio_processor
        self.person_detector = person_detector
        self.emotion_detector = emotion_detector
        self.web_interface = web_interface
        
        # Threading
        self.video_thread = None
        self.audio_thread = None
        self.metrics_thread = None
        self.stop_event = threading.Event()
        
        # Metrics
        self.metrics = {
            'fps': 0,
            'detections': 0,
            'cpu_usage': 0,
            'memory_usage': 0,
            'processing_time': 0
        }
        
        # Detection results
        self.detection_results = {
            'count': 0,
            'detections': []
        }
        
        # Emotion results
        self.emotion_results = {
            'emotion': 'unknown',
            'confidence': 0,
            'emotions': {
                'crying': 0,
                'laughing': 0,
                'babbling': 0,
                'silence': 0
            }
        }
        
        logger.info("Baby Monitor System initialized")
    
    def start(self):
        """
        Start the Baby Monitor System
        """
        logger.info("Starting Baby Monitor System")
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start video processing thread
        if self.camera and self.person_detector:
            self.video_thread = threading.Thread(target=self._video_processing_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
            logger.info("Video processing thread started")
        
        # Start audio processing thread
        if self.audio_processor and self.emotion_detector:
            self.audio_thread = threading.Thread(target=self._audio_processing_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            logger.info("Audio processing thread started")
        
        # Start metrics collection thread
        self.metrics_thread = threading.Thread(target=self._metrics_collection_loop)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
        logger.info("Metrics collection thread started")
        
        # Start web interface
        if self.web_interface:
            logger.info("Starting web interface")
            web_thread = threading.Thread(target=self.web_interface.run)
            web_thread.daemon = True
            web_thread.start()
    
    def stop(self):
        """
        Stop the Baby Monitor System
        """
        logger.info("Stopping Baby Monitor System")
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2.0)
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=2.0)
        
        # Clean up resources
        if self.camera:
            self.camera.release()
        
        if self.audio_processor:
            self.audio_processor.stop()
        
        if self.person_detector:
            self.person_detector.cleanup()
        
        if self.emotion_detector:
            self.emotion_detector.cleanup()
        
        logger.info("Baby Monitor System stopped")
    
    def _video_processing_loop(self):
        """
        Main video processing loop
        """
        last_fps_update = time.time()
        frame_count = 0
        
        while not self.stop_event.is_set():
            # Read frame from camera
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Process frame with person detector
            start_time = time.time()
            processed_frame, detections = self.person_detector.process_frame(frame)
            processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics['processing_time'] = processing_time
            
            # Update detection results
            self.detection_results = {
                'count': len(detections),
                'detections': detections
            }
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - last_fps_update
            if elapsed >= 1.0:
                self.metrics['fps'] = frame_count / elapsed
                frame_count = 0
                last_fps_update = time.time()
            
            # Send frame to web interface
            if self.web_interface:
                # Add timestamp to frame
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(processed_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add FPS to frame
                fps_text = f"FPS: {self.metrics['fps']:.1f}"
                cv2.putText(processed_frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add detection count to frame
                detection_text = f"Detections: {self.detection_results['count']}"
                cv2.putText(processed_frame, detection_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add emotion to frame if available
                if self.emotion_results['emotion'] != 'unknown':
                    emotion_text = f"Emotion: {self.emotion_results['emotion']} ({int(self.emotion_results['confidence']*100)}%)"
                    cv2.putText(processed_frame, emotion_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame)
                self.web_interface.update_frame(buffer.tobytes())
                
                # Update detection information
                self.web_interface.update_detection(self.detection_results)
            
            # Sleep to control frame rate
            time.sleep(0.01)
    
    def _audio_processing_loop(self):
        """
        Main audio processing loop
        """
        while not self.stop_event.is_set():
            # Get audio chunk from audio processor
            audio_chunk = self.audio_processor.get_audio_chunk()
            
            if audio_chunk is None:
                time.sleep(0.1)
                continue
            
            # Process audio with emotion detector
            start_time = time.time()
            emotion, confidence, emotions = self.emotion_detector.process_audio(audio_chunk)
            processing_time = time.time() - start_time
            
            # Update emotion results
            self.emotion_results = {
                'emotion': emotion,
                'confidence': confidence,
                'emotions': emotions
            }
            
            # Send emotion to web interface
            if self.web_interface:
                self.web_interface.update_emotion(self.emotion_results)
            
            # Sleep to control processing rate
            time.sleep(0.1)
    
    def _metrics_collection_loop(self):
        """
        Collect system metrics
        """
        while not self.stop_event.is_set():
            # Get CPU and memory usage
            self.metrics['cpu_usage'] = psutil.cpu_percent()
            self.metrics['memory_usage'] = psutil.virtual_memory().percent
            
            # Update metrics in web interface
            if self.web_interface:
                self.web_interface.update_metrics(self.metrics)
            
            # Sleep for a while
            time.sleep(1.0) 