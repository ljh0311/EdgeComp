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

# Import detectors
from ..detectors.person_detector import PersonDetector
from ..detectors.detector_factory import DetectorFactory, DetectorType
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
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.detector = None
        self.frame_lock = threading.Lock()
        self.metrics_queue = queue.Queue()
        self.video_stream = None
        self.processing_thread = None
        self.is_running = False
        
        # Set up routes
        self.app.route('/')(self.index)
        self.app.route('/metrics')(self.metrics)
        
        # Set up socket events
        self.socketio.on('connect')(self.handle_connect)
        self.socketio.on('disconnect')(self.handle_disconnect)
        self.socketio.on('switch_device')(self.handle_switch_device)
        self.socketio.on('switch_detector')(self.handle_switch_detector)
        
        # Initialize detector
        self._initialize_detector()
        
        # Start metrics thread
        self.metrics_thread = threading.Thread(target=self.emit_metrics, daemon=True)
        self.metrics_thread.start()
        
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()
        
    def _initialize_detector(self):
        """Initialize the detector based on configuration."""
        try:
            if Config.DETECTOR_TYPE.lower() == "lightweight":
                logger.info("Web interface: Using lightweight detector")
                self.detector = DetectorFactory.create_detector(
                    DetectorType.LIGHTWEIGHT.value,
                    config=Config.LIGHTWEIGHT_DETECTION
                )
                # Initialize video stream
                self.video_stream = DetectorFactory.create_video_stream(
                    DetectorType.LIGHTWEIGHT.value,
                    config=Config.LIGHTWEIGHT_DETECTION
                )
                self.video_stream.start()
            else:
                logger.info("Web interface: Using YOLOv8 detector")
                self.detector = PersonDetector(
                    model_path=Config.PERSON_DETECTION["model_path"],
                    device=Config.PERSON_DETECTION["device"]
                )
                # Initialize video stream
                self.video_stream = DetectorFactory.create_video_stream(
                    DetectorType.YOLOV8.value,
                    config={
                        "resolution": (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT),
                        "camera_index": 0
                    }
                )
                
            logger.info("Detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise
        
    def index(self):
        """Render the main page."""
        return render_template('index.html')
        
    def handle_connect(self):
        """Handle client connection."""
        client_id = request.sid
        logger.info(f"Client connected: {client_id}")
        
        # Send initial system state
        self.emit_system_state(client_id)
        
    def handle_disconnect(self):
        """Handle client disconnection."""
        client_id = request.sid
        logger.info(f"Client disconnected: {client_id}")
        
    def emit_system_state(self, client_id: str):
        """Emit current system state to client.
        
        Args:
            client_id: ID of the client to emit to
        """
        try:
            # Emit system status
            self.socketio.emit('status', {
                'status': 'ready',
                'message': 'System ready',
                'camera_enabled': True,
                'audio_enabled': False,
                'detector_type': Config.DETECTOR_TYPE.lower()
            }, room=client_id)
            
            # Emit initial monitoring data
            self.socketio.emit('monitoring', {
                'people_count': 0,
                'movement_detected': False,
                'movement_type': 'none',
                'sound_level': 0,
                'sound_status': 'quiet'
            }, room=client_id)
            
        except Exception as e:
            logger.error(f"Error emitting system state: {e}")
            self.socketio.emit('init_status', {
                'status': 'error',
                'message': f'System error: {str(e)}'
            }, room=client_id)
            
    def handle_switch_device(self, data):
        """Handle device switching request."""
        try:
            if self.detector is None:
                raise RuntimeError("Detector not initialized")
            
            force_cpu = data.get('force_cpu', False)
            logger.info(f"Attempting to switch device (force_cpu={force_cpu})")
            
            with self.frame_lock:  # Ensure no frame is being processed during switch
                # Try to switch device
                success = self.detector.switch_device(force_cpu)
                
                if success:
                    # Get updated device info after successful switch
                    device_info = self.detector.get_device_info()
                    self.socketio.emit('switch_result', {
                        'success': True,
                        'device_info': device_info
                    })
                    logger.info(f"Device switched successfully to {device_info['current_device']}")
                    
                    # Broadcast updated metrics immediately after switch
                    current_metrics = self.get_current_metrics()
                    if current_metrics:
                        self.socketio.emit('metrics_update', {
                            'current': current_metrics['current'],
                            'device_info': device_info
                        })
                else:
                    raise RuntimeError(f"Failed to switch to {'CPU' if force_cpu else 'GPU'}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error switching device: {error_msg}")
            self.socketio.emit('switch_result', {
                'success': False,
                'error': error_msg
            })
            
            # Try to emit current device info after error
            try:
                if self.detector is not None:
                    self.socketio.emit('device_info', self.detector.get_device_info())
            except:
                pass

    def get_current_metrics(self):
        """Get current performance metrics."""
        try:
            if self.detector is None:
                return None
            
            # Initialize metrics with default values
            metrics = {
                'current': {
                    'fps': 0,
                    'frame_time': 0,
                    'cpu_usage': 0,
                    'memory_usage': 0
                },
                'device_info': {
                    'current_device': 'CPU',
                    'gpu_name': 'N/A',
                    'gpu_memory': 0
                }
            }
            
            # Try to get device info
            try:
                metrics['device_info'] = self.detector.get_device_info()
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not get device info: {e}")
                
            # Try to get FPS
            try:
                metrics['current']['fps'] = self.detector.get_fps()
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not get FPS: {e}")
                
            # Try to get processing time
            try:
                metrics['current']['frame_time'] = self.detector.get_processing_time()
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not get processing time: {e}")
                
            # Try to get CPU usage
            try:
                metrics['current']['cpu_usage'] = self.detector.get_cpu_usage()
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not get CPU usage: {e}")
                # Fallback to psutil
                try:
                    import psutil
                    metrics['current']['cpu_usage'] = psutil.cpu_percent()
                except:
                    pass
                
            # Try to get memory usage
            try:
                metrics['current']['memory_usage'] = self.detector.get_memory_usage()
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not get memory usage: {e}")
                # Fallback to psutil
                try:
                    import psutil
                    metrics['current']['memory_usage'] = psutil.virtual_memory().percent
                except:
                    pass
                
            return metrics
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return None

    def emit_metrics(self):
        """Emit metrics periodically."""
        metrics_history = {
            'fps': [],
            'frame_time': [],
            'cpu_usage': [],
            'memory_usage': [],
            'timestamps': []
        }
        max_history = 60  # Keep 60 seconds of history
        start_time = time.time()
        
        while True:
            try:
                current_metrics = self.get_current_metrics()
                if current_metrics:
                    current_time = time.time()
                    # Update history
                    for key in metrics_history:
                        if key != 'timestamps':  # Skip timestamps in this loop
                            metrics_history[key].append(current_metrics['current'][key])
                            if len(metrics_history[key]) > max_history:
                                metrics_history[key] = metrics_history[key][-max_history:]
                    
                    # Add timestamp
                    timestamp = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(current_time))
                    metrics_history['timestamps'].append(timestamp)
                    if len(metrics_history['timestamps']) > max_history:
                        metrics_history['timestamps'] = metrics_history['timestamps'][-max_history:]
                    
                    # Emit both current values and history
                    self.socketio.emit('metrics_update', {
                        'current': current_metrics['current'],
                        'history': metrics_history,
                        'start_time': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(start_time))
                    })
                    
                    # Emit device info separately
                    self.socketio.emit('device_info', current_metrics['device_info'])
                    
            except Exception as e:
                logger.error(f"Error in metrics emission: {e}")
            
            time.sleep(1)  # Update every second

    def metrics(self):
        """Render the metrics page."""
        return render_template('metrics.html')

    def start(self):
        """Start the web interface."""
        try:
            logger.info(f"Baby Monitor web system started on http://{self.host}:{self.port}")
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        except Exception as e:
            logger.error(f"Error starting web interface: {e}")
            raise
            
    def stop(self):
        """Stop the web interface."""
        try:
            self.is_running = False
            
            # Stop video stream
            if self.video_stream:
                if hasattr(self.video_stream, 'stop'):
                    self.video_stream.stop()
                elif hasattr(self.video_stream, 'release'):
                    self.video_stream.release()
            
            # Clean up detector
            if self.detector and hasattr(self.detector, 'cleanup'):
                self.detector.cleanup()
                
            logger.info("Web interface stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping web interface: {e}")
            raise

    def handle_switch_detector(self, data):
        """Handle switch detector event."""
        detector_type = data.get('detector_type', 'yolov8')
        
        try:
            # Clean up existing resources
            with self.frame_lock:
                # Stop video stream
                if hasattr(self.video_stream, 'stop'):
                    self.video_stream.stop()
                elif hasattr(self.video_stream, 'release'):
                    self.video_stream.release()
                
                # Clean up detector
                if self.detector and hasattr(self.detector, 'cleanup'):
                    self.detector.cleanup()
            
                # Initialize new detector
                if detector_type.lower() == "lightweight":
                    logger.info("Switching to lightweight detector")
                    self.detector = DetectorFactory.create_detector(
                        DetectorType.LIGHTWEIGHT.value,
                        config=Config.LIGHTWEIGHT_DETECTION
                    )
                    # Initialize video stream
                    self.video_stream = DetectorFactory.create_video_stream(
                        DetectorType.LIGHTWEIGHT.value,
                        config=Config.LIGHTWEIGHT_DETECTION
                    )
                    if hasattr(self.video_stream, 'start'):
                        self.video_stream.start()
                    # Update config
                    Config.DETECTOR_TYPE = "lightweight"
                else:
                    logger.info("Switching to YOLOv8 detector")
                    self.detector = PersonDetector(
                        model_path=Config.PERSON_DETECTION["model_path"],
                        device=Config.PERSON_DETECTION["device"]
                    )
                    # Initialize video stream
                    self.video_stream = DetectorFactory.create_video_stream(
                        DetectorType.YOLOV8.value,
                        config={
                            "resolution": (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT),
                            "camera_index": 0
                        }
                    )
                    # Update config
                    Config.DETECTOR_TYPE = "yolov8"
            
            # Emit success message
            self.socketio.emit('detector_switched', {
                'status': 'success',
                'detector_type': detector_type
            })
            
            # Emit updated status with new detector type
            self.socketio.emit('status', {
                'detector_type': detector_type
            })
            
            # Emit alert
            self.emit_alert('info', f'Switched to {detector_type} detector')
            
        except Exception as e:
            logger.error(f"Error switching detector: {e}")
            self.socketio.emit('detector_switched', {
                'status': 'error',
                'message': str(e)
            })
            
            # Emit alert
            self.emit_alert('warning', f'Failed to switch detector: {str(e)}')
            
    def emit_frame(self, frame):
        """Emit a frame to connected clients.
        
        Args:
            frame: The frame to emit
        """
        try:
            if frame is None:
                return
                
            # Resize frame for web streaming (reduce bandwidth)
            max_width = 640
            if frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (max_width, new_height))
                
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            # Convert to base64 for sending over socket.io
            import base64
            frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
            
            # Emit frame to all clients
            self.socketio.emit('frame', {'data': frame_b64})
            
        except Exception as e:
            logger.error(f"Error emitting frame: {e}")
            
    def emit_alert(self, level, message):
        """Emit an alert to connected clients.
        
        Args:
            level: Alert level (info, warning, critical)
            message: Alert message
        """
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            self.socketio.emit('alert', {
                'level': level,
                'message': message,
                'timestamp': timestamp,
                'should_beep': level in ['warning', 'critical']
            })
            
        except Exception as e:
            logger.error(f"Error emitting alert: {e}")
            
    def emit_status(self, status_data):
        """Emit status update to connected clients.
        
        Args:
            status_data: Dictionary containing status information
        """
        try:
            self.socketio.emit('status', status_data)
        except Exception as e:
            logger.error(f"Error emitting status: {e}")
            
    def emit_monitoring(self, monitoring_data):
        """Emit monitoring data to connected clients.
        
        Args:
            monitoring_data: Dictionary containing monitoring information
        """
        try:
            self.socketio.emit('monitoring', monitoring_data)
        except Exception as e:
            logger.error(f"Error emitting monitoring data: {e}")

    def process_frames(self):
        """Process frames from the video stream and emit them to clients."""
        last_detection_time = time.time()
        detection_interval = 0.1  # Process detections every 100ms
        
        while self.is_running:
            try:
                if self.video_stream is None or self.detector is None:
                    time.sleep(0.1)
                    continue
                    
                # Read frame from video stream
                if hasattr(self.video_stream, 'read'):
                    frame = self.video_stream.read()
                else:
                    # OpenCV VideoCapture
                    ret, frame = self.video_stream.read()
                    if not ret:
                        time.sleep(0.1)
                        continue
                
                # Skip if frame is empty
                if frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue
                
                current_time = time.time()
                
                # Process frame with detector at regular intervals
                if current_time - last_detection_time >= detection_interval:
                    with self.frame_lock:
                        results = self.detector.process_frame(frame)
                    
                    # Emit processed frame
                    self.emit_frame(results['frame'])
                    
                    # Emit detection results
                    people_count = len(results.get('detections', []))
                    
                    # Determine movement status
                    movement_detected = False
                    movement_type = "none"
                    for detection in results.get('detections', []):
                        if 'status' in detection:
                            if detection['status'] in ['moving', 'falling']:
                                movement_detected = True
                                movement_type = detection['status']
                                break
                    
                    # Emit monitoring data
                    self.emit_monitoring({
                        'people_count': people_count,
                        'movement_detected': movement_detected,
                        'movement_type': movement_type
                    })
                    
                    last_detection_time = current_time
                
                # Sleep to reduce CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in frame processing: {e}")
                time.sleep(0.1)
                

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 