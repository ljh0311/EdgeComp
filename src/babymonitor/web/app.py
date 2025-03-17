"""
Web Interface Module
-------------------
Handles the web interface for the Baby Monitor System.
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import logging
import time
import threading
import queue
from typing import Dict, Any
import os
import base64
import json

# Import detectors
from ..detectors.person_detector import PersonDetector
from ..detectors.emotion_detector import EmotionDetector
from ..config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BabyMonitorWeb:
    """Web interface for the Baby Monitor System."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        """Initialize the web interface.
        
        Args:
            host: Host address to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        
        # Set up Flask with correct template directory
        template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
        static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
        self.app = Flask(__name__, 
                        template_folder=template_dir,
                        static_folder=static_dir)
        
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", json=json)
        
        # Initialize detectors
        self.person_detector = PersonDetector(threshold=Config.PERSON_DETECTION.get("threshold", 0.7))
        self.emotion_detector = EmotionDetector(threshold=Config.EMOTION_DETECTION.get("threshold", 0.7))
        
        # Thread safety
        self.frame_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        
        # Queues for data
        self.frame_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Thread management
        self.is_running = False
        self.video_thread = None
        self.audio_thread = None
        self.stop_event = threading.Event()
        
        # Video capture
        self.video_capture = None
        self.camera_id = 0
        
        # System metrics
        self.start_time = time.time()
        self.detection_history = []
        self.emotion_history = []
        
        # Set up routes
        self.setup_routes()
        
        # Set up socket events
        self.setup_socketio()
        
        # Start processing threads
        self.start_processing()
        
    def setup_routes(self):
        """Set up Flask routes."""
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
        @self.app.route('/metrics')
        def metrics():
            return render_template('metrics.html')
            
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route."""
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
                          
        @self.app.route('/api/status')
        def status():
            """Return system status."""
            return jsonify({
                'camera_enabled': self.video_capture is not None and self.video_capture.isOpened(),
                'person_detector_fps': float(self.person_detector.fps),
                'emotion_detector_fps': float(self.emotion_detector.fps),
                'uptime': time.time() - self.start_time
            })
            
        @self.app.route('/api/toggle_camera', methods=['POST'])
        def toggle_camera():
            """Toggle camera on/off."""
            if self.video_capture is not None and self.video_capture.isOpened():
                self.stop_camera()
                return jsonify({'status': 'Camera stopped'})
            else:
                self.start_camera()
                return jsonify({'status': 'Camera started'})
                
    def setup_socketio(self):
        """Set up Socket.IO event handlers."""
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
            
        @self.socketio.on('toggle_camera')
        def handle_toggle_camera():
            if self.video_capture is not None and self.video_capture.isOpened():
                self.stop_camera()
                self.socketio.emit('camera_status', {'enabled': False})
            else:
                self.start_camera()
                self.socketio.emit('camera_status', {'enabled': True})
                
    def start_camera(self):
        """Start the camera."""
        if self.video_capture is None or not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(self.camera_id)
            if not self.video_capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
                
            # Set camera properties
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            logger.info("Camera started")
            return True
        return False
        
    def stop_camera(self):
        """Stop the camera."""
        if self.video_capture is not None and self.video_capture.isOpened():
            self.video_capture.release()
            self.video_capture = None
            logger.info("Camera stopped")
            return True
        return False
        
    def start_processing(self):
        """Start processing threads."""
        self.is_running = True
        self.stop_event.clear()
        
        # Start video processing thread
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.audio_thread.start()
        
        logger.info("Processing threads started")
        
    def stop_processing(self):
        """Stop processing threads."""
        self.is_running = False
        self.stop_event.set()
        
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
            
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
            
        self.stop_camera()
        logger.info("Processing threads stopped")
        
    def process_video(self):
        """Process video frames."""
        logger.info("Video processing thread started")
        
        # Start camera if not already started
        if not self.start_camera():
            logger.error("Failed to start camera")
            return
            
        while self.is_running and not self.stop_event.is_set():
            try:
                if self.video_capture is not None and self.video_capture.isOpened():
                    ret, frame = self.video_capture.read()
                    if not ret:
                        logger.error("Failed to read frame from camera")
                        time.sleep(0.1)
                        continue
                        
                    # Process frame with person detector
                    result = self.person_detector.process_frame(frame)
                    processed_frame = result['frame']
                    
                    # Put frame in queue for streaming
                    if not self.frame_queue.full():
                        self.frame_queue.put(processed_frame)
                        
                    # Prepare detection data for Socket.IO
                    detections = result.get('detections', [])
                    
                    # Ensure all values are JSON serializable
                    detection_data = {
                        'count': len(detections),
                        'fps': float(result.get('fps', 0)),
                        'detections': [
                            {
                                'bbox': [float(x) for x in det.get('bbox', [0, 0, 0, 0])],
                                'confidence': float(det.get('confidence', 0)),
                                'class': str(det.get('class', 'unknown'))
                            }
                            for det in detections
                        ]
                    }
                    
                    # Add to detection history
                    self.detection_history.append({
                        'timestamp': time.time(),
                        'count': len(detections),
                        'types': {
                            det_type: sum(1 for d in detections if d.get('class') == det_type)
                            for det_type in set(d.get('class', 'unknown') for d in detections)
                        }
                    })
                    
                    # Limit history size
                    if len(self.detection_history) > 100:
                        self.detection_history.pop(0)
                    
                    # Emit detection data via Socket.IO
                    self.socketio.emit('detection_update', detection_data)
                    
                    # Emit metrics
                    metrics = {
                        'fps': float(self.person_detector.fps),
                        'detection_count': len(detections),
                        'detection_types': {
                            str(det_type): sum(1 for d in detections if d.get('class') == det_type)
                            for det_type in set(d.get('class', 'unknown') for d in detections)
                        }
                    }
                    self.socketio.emit('metrics_update', {'current': metrics})
                    
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in video processing: {e}")
                time.sleep(0.1)
                
    def process_audio(self):
        """Process audio chunks."""
        logger.info("Audio processing thread started")
        
        # Set up audio capture
        try:
            import sounddevice as sd
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                if not self.audio_queue.full():
                    self.audio_queue.put(indata.copy())
                    
            # Start audio stream
            with sd.InputStream(
                channels=1,
                samplerate=self.emotion_detector.SAMPLE_RATE,
                blocksize=self.emotion_detector.CHUNK_SIZE,
                callback=audio_callback
            ):
                while self.is_running and not self.stop_event.is_set():
                    try:
                        if not self.audio_queue.empty():
                            audio_data = self.audio_queue.get(timeout=0.1)
                            
                            # Process audio with emotion detector
                            result = self.emotion_detector.process_audio(audio_data)
                            
                            # Emit emotion data via Socket.IO
                            if result['emotion'] not in ['buffering', 'unknown', 'error']:
                                # Ensure all values are JSON serializable
                                emotion_data = {
                                    'emotion': str(result['emotion']),
                                    'confidence': float(result['confidence']),
                                    'emotions': {
                                        str(k): float(v) for k, v in result['emotions'].items()
                                    }
                                }
                                
                                # Add to emotion history
                                self.emotion_history.append({
                                    'timestamp': time.time(),
                                    'emotion': result['emotion'],
                                    'confidence': float(result['confidence'])
                                })
                                
                                # Limit history size
                                if len(self.emotion_history) > 100:
                                    self.emotion_history.pop(0)
                                
                                self.socketio.emit('emotion_update', emotion_data)
                                
                                # Emit alert for crying
                                if result['emotion'] == 'crying' and result['confidence'] > self.emotion_detector.threshold:
                                    self.socketio.emit('alert', {
                                        'type': 'crying',
                                        'message': 'Baby is crying!',
                                        'timestamp': time.time()
                                    })
                        else:
                            time.sleep(0.01)
                            
                    except queue.Empty:
                        time.sleep(0.01)
                    except Exception as e:
                        logger.error(f"Error in audio processing: {e}")
                        time.sleep(0.1)
                        
        except ImportError:
            logger.error("sounddevice not installed, audio processing disabled")
        except Exception as e:
            logger.error(f"Error setting up audio: {e}")
            
    def generate_frames(self):
        """Generate frames for video streaming."""
        while True:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                # Return an empty frame if queue is empty
                empty_frame = np.zeros((Config.CAMERA_HEIGHT, Config.CAMERA_WIDTH, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', empty_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"Error generating frames: {e}")
                time.sleep(0.1)
                
    def start(self):
        """Start the web interface."""
        try:
            logger.info(f"Baby Monitor web system started on http://{self.host}:{self.port}")
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        except Exception as e:
            logger.error(f"Error starting web interface: {e}")
            self.stop()
            raise
            
    def stop(self):
        """Stop the web interface."""
        try:
            self.stop_processing()
            
            # Clean up detectors
            if hasattr(self, 'person_detector'):
                self.person_detector.cleanup()
                
            if hasattr(self, 'emotion_detector'):
                self.emotion_detector.cleanup()
                
            logger.info("Web interface stopped")
        except Exception as e:
            logger.error(f"Error stopping web interface: {e}")
            raise

if __name__ == '__main__':
    web_app = BabyMonitorWeb()
    web_app.start() 