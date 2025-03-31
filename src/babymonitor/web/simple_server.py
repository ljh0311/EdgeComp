"""
Simple Baby Monitor Web Server
============================
A simplified web server for the Baby Monitor System that works reliably on Windows.
"""

import os
import json
import time
import signal
import socket
import threading
import logging
import eventlet
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, redirect, url_for, request
from flask_socketio import SocketIO
import numpy as np
import cv2
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SimpleBabyMonitorWeb')

def find_free_port(start_port=5000, max_port=5100):
    """Find a free port to use"""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError("No free ports available")

class SimpleBabyMonitorWeb:
    """
    Simplified web interface for the Baby Monitor System
    """
    def __init__(self, camera=None, person_detector=None, emotion_detector=None, host='0.0.0.0', port=5000, mode='normal', debug=False):
        """Initialize the web interface"""
        self.camera = camera
        self.person_detector = person_detector
        self.emotion_detector = emotion_detector
        self.host = host
        self.port = port
        self.mode = mode
        self.debug = debug
        self.app = Flask(__name__, 
                         static_folder=os.path.join(os.path.dirname(__file__), 'static'),
                         template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
        
        # Add throttling for emotion updates
        self.last_emotion_update = {
            'emotion': None,
            'confidence': 0.0,
            'timestamp': 0,
            'emitted': False
        }
        self.emotion_update_interval = 0.5  # Minimum seconds between emotion updates
        
        # Initialize metrics
        self._setup_metrics()
        
        # Configure logging
        self._configure_logging()
        
        # Initialize Socket.IO
        self._init_socketio()
        
        # Initialize routes
        self._setup_routes()
        
        # Running flag
        self.running = True
        
        # Frame buffer
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        
        # Create a blank frame for when no camera feed is available
        self.blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.blank_frame, "No camera feed available", (120, 240), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, self.blank_buffer = cv2.imencode('.jpg', self.blank_frame)
        self.blank_bytes = self.blank_buffer.tobytes()
        
        # Start time
        self.start_time = time.time()
        
    def _setup_metrics(self):
        """Initialize metrics"""
        self.metrics = {
            'current': {
                'emotion': 'unknown',
                'emotion_confidence': 0.0,
                'person_detected': False,
                'person_confidence': 0.0,
                'fps': 0.0,
                'audio_level': 0.0,
                'detections': 0
            },
            'history': {
                'emotions': {
                    'crying': 0,
                    'laughing': 0,
                    'babbling': 0,
                    'silence': 0,
                    'unknown': 0
                },
                'person_detections': 0,
                'alerts': 0
            },
            'settings': {
                'emotion_threshold': 0.5,
                'person_threshold': 0.5,
                'alert_timeout': 60,
                'alert_on_crying': True,
                'alert_on_person': True
            },
            'detection_types': {
                'face': 0,
                'upper_body': 0,
                'full_body': 0
            },
            'total_detections': 0
        }
        
        # Initialize emotion log
        self.emotion_log = []
        self.max_log_entries = 100
        
        # Initialize alerts
        self.alerts = []
        
        # Initialize emotion distribution with supported emotions
        if self.emotion_detector and hasattr(self.emotion_detector, 'emotions'):
            self.metrics['history']['emotions'] = {emotion: 0 for emotion in self.emotion_detector.emotions}
        
        # System status
        self.system_status = {
            'uptime': '00:00:00',
            'cpu_usage': 0,
            'memory_usage': 0,
            'camera_status': 'connected' if self.camera and self.camera.is_opened() else 'disconnected',
            'person_detector_status': 'running' if self.person_detector else 'stopped',
            'emotion_detector_status': 'running' if self.emotion_detector else 'stopped',
        }
        
        # Activity log for emotion events
        self.emotion_log = []
        self.max_log_entries = 50  # Keep last 50 entries
        
        # Initialize alerts list
        self.alerts = []
        
    def _configure_logging(self):
        """Configure logging"""
        # This method is now empty as the existing logging configuration is used
        pass

    def _init_socketio(self):
        """Initialize Socket.IO"""
        # Configure Socket.IO with more reliable settings
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*", 
            async_mode='eventlet',
            logger=True,
            engineio_logger=True if self.debug else False,
            ping_timeout=60,
            ping_interval=25
        )
        
        # Set up Socket.IO connection event handlers
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
    def _setup_routes(self):
        """Initialize routes"""
        @self.app.route('/')
        def index():
            """Main page"""
            if self.mode == "dev":
                return redirect(url_for('metrics'))
            return render_template('index.html', mode=self.mode, dev_mode=self.mode == "dev")
        
        @self.app.route('/metrics')
        def metrics():
            """Metrics page"""
            return render_template('metrics.html', mode=self.mode, dev_mode=self.mode == "dev")
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video feed endpoint"""
            return Response(self._generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for metrics"""
            # Get time range from query parameter
            time_range = request.args.get('time_range', '1h')
            
            # Get emotion history from detector if available
            emotion_history = {}
            emotion_percentages = {}
            emotion_timeline = []
            supported_emotions = []
            
            if self.emotion_detector and hasattr(self.emotion_detector, 'get_emotion_history'):
                try:
                    history_data = self.emotion_detector.get_emotion_history(time_range)
                    emotion_percentages = history_data.get('percentages', {})
                    emotion_timeline = history_data.get('emotions', [])
                    supported_emotions = self.emotion_detector.emotions
                except Exception as e:
                    logger.error(f"Error getting emotion history: {str(e)}")
            else:
                # Use current simple metrics as fallback
                if hasattr(self, 'metrics') and 'history' in self.metrics:
                    emotion_counts = self.metrics['history'].get('emotions', {})
                    total = sum(emotion_counts.values()) or 1  # Avoid division by zero
                    emotion_percentages = {k: (v / total * 100) for k, v in emotion_counts.items()}
            
            # Enhanced metrics structure to match what the metrics.js expects
            metrics_data = {
                'current': {
                    'fps': self.metrics['current']['fps'],
                    'detections': self.metrics['current'].get('detections', 0),
                    'emotion': self.metrics['current'].get('emotion', 'unknown'),
                    'emotion_confidence': self.metrics['current'].get('emotion_confidence', 0.0),
                },
                'history': {
                    'emotions': emotion_percentages
                },
                'emotion_timeline': emotion_timeline,
                'supported_emotions': supported_emotions,
                'detection_types': self.metrics['detection_types'],
                'total_detections': self.metrics.get('total_detections', 0),
                # Add YOLOv8 specific metrics
                'detection_confidence_avg': 0.85,  # Example value, adjust based on actual data
                'peak_detections': max(1, self.metrics['current'].get('detections', 0)),  # Default to at least 1
                'frame_skip': 2,  # Frame skip rate from person detector
                'process_resolution': '640x480',  # Processing resolution
                'confidence_threshold': 0.5,  # Detection confidence threshold
                'detection_history_size': 5,  # History size for detection smoothing
                'detector_model': 'YOLOv8n'  # Detector model name
            }
            
            # If person detector is available, get real values
            if self.person_detector:
                metrics_data['frame_skip'] = getattr(self.person_detector, 'frame_skip', 2)
                metrics_data['confidence_threshold'] = getattr(self.person_detector, 'threshold', 0.5)
                metrics_data['detection_history_size'] = getattr(self.person_detector, 'max_history_size', 5)
                
                # Get process resolution if available
                process_res = getattr(self.person_detector, 'process_resolution', None)
                if process_res and isinstance(process_res, tuple) and len(process_res) == 2:
                    metrics_data['process_resolution'] = f"{process_res[0]}x{process_res[1]}"
                    
                # Calculate confidence average from detection history if available
                detection_history = getattr(self.person_detector, 'detection_history', [])
                if detection_history and hasattr(self.person_detector, 'last_result') and self.person_detector.last_result:
                    detections = self.person_detector.last_result.get('detections', [])
                    if detections:
                        confidence_sum = sum(d.get('confidence', 0) for d in detections)
                        if len(detections) > 0:
                            metrics_data['detection_confidence_avg'] = confidence_sum / len(detections)
                            
            # Add model info if emotion detector is available
            if self.emotion_detector:
                metrics_data['emotion_model'] = {
                    'id': getattr(self.emotion_detector, 'model_id', 'unknown'),
                    'name': getattr(self.emotion_detector, 'model_info', {}).get('name', 'Unknown Model'),
                    'emotions': getattr(self.emotion_detector, 'emotions', ['unknown']),
                }
                
            # Add recent emotion log entries
            metrics_data['emotion_log'] = self.emotion_log[-10:]  # Last 10 entries
            
            return jsonify(metrics_data)
        
        @self.app.route('/api/emotion_history')
        def api_emotion_history():
            """API endpoint for detailed emotion history"""
            time_range = request.args.get('time_range', '1h')
            
            if self.emotion_detector and hasattr(self.emotion_detector, 'get_emotion_history'):
                try:
                    history_data = self.emotion_detector.get_emotion_history(time_range)
                    return jsonify({
                        'status': 'success',
                        'history': history_data
                    })
                except Exception as e:
                    logger.error(f"Error getting emotion history: {str(e)}")
                    return jsonify({
                        'status': 'error',
                        'message': str(e)
                    }), 500
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Emotion detector not available or does not support history'
                }), 404
        
        @self.app.route('/api/emotion_log')
        def api_emotion_log():
            """API endpoint for emotion event log"""
            return jsonify({
                'status': 'success',
                'log': self.emotion_log
            })
        
        @self.app.route('/api/system_info')
        def api_system_info():
            """API endpoint for system info"""
            info = self.system_status.copy()
            
            # Add emotion model info if available
            if self.emotion_detector:
                info['emotion_model'] = {
                    'id': getattr(self.emotion_detector, 'model_id', 'unknown'),
                    'name': getattr(self.emotion_detector, 'model_info', {}).get('name', 'Unknown Model'),
                    'emotions': getattr(self.emotion_detector, 'emotions', ['unknown']),
                }
                
                # Add available models if the method exists
                if hasattr(self.emotion_detector, 'get_available_models'):
                    try:
                        info['available_emotion_models'] = self.emotion_detector.get_available_models()
                    except Exception as e:
                        logger.error(f"Error getting available emotion models: {str(e)}")
            
            return jsonify(info)
        
        @self.app.route('/repair')
        def repair_tools():
            """Repair tools page"""
            return render_template('repair_tools.html', mode=self.mode, dev_mode=self.mode == "dev")
        
        @self.app.route('/repair/run', methods=['POST'])
        def repair_run():
            """API endpoint for repair tools"""
            try:
                tool = request.form.get('tool')
                if tool == 'restart_camera':
                    if self.camera:
                        self.camera.release()
                        time.sleep(1)
                        self.camera.open(0)
                        return jsonify({'status': 'success', 'message': 'Camera restarted successfully'})
                    return jsonify({'status': 'error', 'message': 'Camera not initialized'})
                
                elif tool == 'restart_audio':
                    if self.emotion_detector:
                        self.emotion_detector.reset()
                        return jsonify({'status': 'success', 'message': 'Audio system restarted successfully'})
                    return jsonify({'status': 'error', 'message': 'Audio system not initialized'})
                
                elif tool == 'restart_system':
                    if self.camera:
                        self.camera.release()
                        time.sleep(1)
                        self.camera.open(0)
                    if self.emotion_detector:
                        self.emotion_detector.reset()
                    if self.person_detector:
                        self.person_detector.reset()
                    return jsonify({'status': 'success', 'message': 'System restarted successfully'})
                
                elif tool == 'switch_emotion_model':
                    model_id = request.form.get('model_id')
                    if not model_id:
                        return jsonify({'status': 'error', 'message': 'No model ID provided'})
                        
                    if self.emotion_detector and hasattr(self.emotion_detector, 'switch_model'):
                        try:
                            result = self.emotion_detector.switch_model(model_id)
                            # Update metrics to reflect new emotion set
                            if 'model_info' in result and 'emotions' in result['model_info']:
                                self.metrics['history']['emotions'] = {emotion: 0 for emotion in result['model_info']['emotions']}
                                
                            # Emit model change event
                            self.socketio.emit('emotion_model_changed', result['model_info'])
                            
                            return jsonify({
                                'status': 'success', 
                                'message': f"Switched to model: {result['model_info']['name']}",
                                'model_info': result['model_info']
                            })
                        except Exception as e:
                            logger.error(f"Error switching emotion model: {str(e)}")
                            return jsonify({'status': 'error', 'message': str(e)})
                    return jsonify({'status': 'error', 'message': 'Emotion detector not initialized or does not support model switching'})
                
                return jsonify({'status': 'error', 'message': 'Invalid repair tool'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
        # Endpoint for emotion model actions
        @self.app.route('/api/emotion/models', methods=['GET'])
        def get_emotion_models():
            """Get available emotion models"""
            if self.emotion_detector and hasattr(self.emotion_detector, 'get_available_models'):
                try:
                    models_data = self.emotion_detector.get_available_models()
                    return jsonify(models_data)
                except Exception as e:
                    logger.error(f"Error getting emotion models: {str(e)}")
                    return jsonify({'error': str(e)}), 500
            return jsonify({'error': 'Emotion detector not initialized or does not support model listing'}), 404
        
        @self.app.route('/api/emotion/switch_model', methods=['POST'])
        def switch_emotion_model():
            """Switch to a different emotion model"""
            if not self.emotion_detector or not hasattr(self.emotion_detector, 'switch_model'):
                return jsonify({'error': 'Emotion detector not initialized or does not support model switching'}), 404
                
            try:
                data = request.get_json()
                if not data or 'model_id' not in data:
                    return jsonify({'error': 'No model ID provided'}), 400
                    
                model_id = data.get('model_id')
                result = self.emotion_detector.switch_model(model_id)
                
                # Update metrics to reflect new emotion set
                if 'model_info' in result and 'emotions' in result['model_info']:
                    self.metrics['history']['emotions'] = {emotion: 0 for emotion in result['model_info']['emotions']}
                    
                # Emit model change event
                self.socketio.emit('emotion_model_changed', result['model_info'])
                
                # Log the model change
                self._add_to_emotion_log({
                    'type': 'model_changed',
                    'model': result['model_info']['name'],
                    'timestamp': time.time()
                })
                
                return jsonify({
                    'status': 'success', 
                    'message': f"Switched to model: {result['model_info']['name']}",
                    'model_info': result['model_info']
                })
            except Exception as e:
                logger.error(f"Error switching emotion model: {str(e)}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # Socket.io events
        @self.socketio.on('request_metrics')
        def handle_request_metrics(data):
            """Handle request for metrics data"""
            time_range = data.get('timeRange', '1h')
            
            # Get emotion history from detector
            if self.emotion_detector and hasattr(self.emotion_detector, 'get_emotion_history'):
                try:
                    history_data = self.emotion_detector.get_emotion_history(time_range)
                    # Emit history data
                    self.socketio.emit('emotion_history', history_data)
                except Exception as e:
                    logger.error(f"Error getting emotion history: {str(e)}")
        
        if self.mode == "dev":
            @self.app.route('/dev/tools')
            def dev_tools():
                """Developer tools page"""
                return render_template('dev_tools.html', mode=self.mode, dev_mode=True)
            
            @self.app.route('/dev/logs')
            def dev_logs():
                """Logs page"""
                return render_template('logs.html', mode=self.mode, dev_mode=True)
    
    def _generate_frames(self):
        """Generate video frames for streaming"""
        while True:
            try:
                with self.frame_lock:
                    if self.frame_buffer is None:
                        # If no frame is available, yield a blank frame
                        yield (b'--frame\r\n'
                             b'Content-Type: image/jpeg\r\n\r\n' + self.blank_bytes + b'\r\n')
                    else:
                        # Yield the frame
                        yield (b'--frame\r\n'
                             b'Content-Type: image/jpeg\r\n\r\n' + self.frame_buffer + b'\r\n')
                
                # Sleep to control frame rate
                eventlet.sleep(0.03)  # ~30 FPS
            except Exception as e:
                logger.error(f"Error generating frames: {str(e)}")
                eventlet.sleep(0.1)
    
    def _add_to_emotion_log(self, entry):
        """Add an entry to the emotion log"""
        self.emotion_log.append(entry)
        # Trim log if it gets too big
        if len(self.emotion_log) > self.max_log_entries:
            self.emotion_log = self.emotion_log[-self.max_log_entries:]
    
    def _process_frames(self):
        """Process frames from camera"""
        frame_count = 0
        last_fps_time = time.time()
        last_emotion_log_time = time.time()
        
        while self.running:
            try:
                if not self.camera or not self.camera.is_opened():
                    # Update frame buffer with blank frame
                    with self.frame_lock:
                        self.frame_buffer = self.blank_bytes
                    eventlet.sleep(0.1)
                    continue
                
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    eventlet.sleep(0.1)
                    continue
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.metrics['current']['fps'] = frame_count
                    frame_count = 0
                    last_fps_time = current_time
                
                # Process frame with person detector if available
                if self.person_detector:
                    results = self.person_detector.process_frame(frame)
                    processed_frame = results.get('frame', frame)
                else:
                    processed_frame = frame
                
                # Encode and store frame
                with self.frame_lock:
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    self.frame_buffer = buffer.tobytes()
                
                # Sleep to control frame rate
                eventlet.sleep(0.02)  # ~50 FPS maximum
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                eventlet.sleep(0.1)
    
    def _update_system_status(self):
        """Update system status information"""
        while self.running:
            try:
                # Calculate uptime
                uptime = time.time() - self.start_time
                hours, remainder = divmod(uptime, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                # Get emotion state
                current_emotion = "Unknown"
                if self.emotion_detector:
                    current_emotion = self.metrics['current'].get('emotion', 'unknown')
                
                # Track peak detections
                current_detections = self.metrics['current'].get('detections', 0)
                if not hasattr(self, '_peak_detections') or current_detections > self._peak_detections:
                    self._peak_detections = current_detections
                
                # Track total detections (increment by current detection count)
                if not hasattr(self, '_total_detections'):
                    self._total_detections = 0
                self._total_detections += current_detections
                
                # Update metrics with detection data
                self.metrics['peak_detections'] = getattr(self, '_peak_detections', 0)
                self.metrics['total_detections'] = getattr(self, '_total_detections', 0)
                
                # Update system status
                self.system_status.update({
                    'uptime': f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'camera_status': 'connected' if self.camera and self.camera.is_opened() else 'disconnected',
                    'person_detector_status': 'running' if self.person_detector else 'stopped',
                    'emotion_detector_status': 'running' if self.emotion_detector else 'stopped',
                    'fps': self.metrics['current']['fps'],
                    'current_emotion': current_emotion
                })
                
                # Emit system status via Socket.IO
                self.socketio.emit('system_info', self.system_status)
                
                # Emit metrics update
                self.socketio.emit('metrics_update', {
                    'current': {
                        'cpu_usage': self.system_status['cpu_usage'],
                        'memory_usage': self.system_status['memory_usage'],
                        'fps': self.system_status['fps'],
                        'detection_count': current_detections,
                        'emotion': current_emotion,
                        'emotion_confidence': self.metrics['current'].get('emotion_confidence', 0.0)
                    },
                    'history': {
                        'emotions': self.metrics['history']['emotions']
                    }
                })
                
                # Sleep for 1 second
                eventlet.sleep(1)
                
            except Exception as e:
                logger.error(f"Error updating system status: {str(e)}")
                eventlet.sleep(1)
    
    def run(self):
        """Run the web server"""
        if not self.camera:
            logger.error("No camera available")
            return
            
        self.running = True
        logger.info(f"Running Baby Monitor Web Server in {self.mode.upper()} mode on http://{self.host}:{self.port}")
        
        # Start frame processing in a separate greenlet
        eventlet.spawn(self._process_frames)
        
        # Start system status updates in a separate greenlet
        eventlet.spawn(self._update_system_status)
        
        try:
            # Run the Flask app with Socket.IO
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False
            )
        except Exception as e:
            logger.error(f"Error running web server: {str(e)}")
            self.stop()
    
    def stop(self):
        """Stop the web server"""
        if not self.running:
            return
            
        self.running = False
        logger.info("Stopping Baby Monitor Web Server")
        
        try:
            if hasattr(self, 'socketio'):
                self.socketio.stop()
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
        finally:
            logger.info("Web server stopped")

    def emit_emotion_update(self, emotion_data):
        """
        Emit emotion update to connected clients
        
        Args:
            emotion_data (dict): Dictionary containing emotion data with keys:
                - emotion (str): Detected emotion
                - confidence (float): Confidence score
        """
        try:
            if not isinstance(emotion_data, dict):
                logger.warning(f"Invalid emotion_data format: {emotion_data}")
                return
                
            emotion = emotion_data.get('emotion', 'unknown')
            confidence = emotion_data.get('confidence', 0.0)
            confidences = emotion_data.get('confidences', {})
            
            # Get current time
            current_time = time.time()
            
            # Check if we should throttle this update
            should_emit = False
            
            # Get the message queue ID for deduplication
            batch_id = emotion_data.get('batch_id', None)
            
            # Only log and emit if:
            # 1. The emotion has changed from the last emitted one
            # 2. OR significant confidence change (>10%)
            # 3. OR enough time has passed since last emission (at least 1 second)
            # 4. OR we have a new batch ID (to avoid duplicate messages from same audio batch)
            
            # Check if emotion changed
            if emotion != self.last_emotion_update['emotion']:
                should_emit = True
                logger.info(f"Emotion changed from {self.last_emotion_update['emotion']} to {emotion}")
            # Check if confidence changed significantly
            elif abs(confidence - self.last_emotion_update['confidence']) > 0.1:
                should_emit = True
                logger.info(f"Confidence changed significantly: {self.last_emotion_update['confidence']:.2f} -> {confidence:.2f}")
            # Check time interval - increased to 1 second minimum between general updates
            elif current_time - self.last_emotion_update['timestamp'] >= 1.0:
                should_emit = True
                # Only log occasional updates (every 5 seconds) to reduce noise
                if current_time - self.last_emotion_update['timestamp'] >= 5.0:
                    logger.info(f"Periodic emotion update: {emotion} ({confidence:.4f})")
            # Check for duplicate batch IDs
            elif batch_id and batch_id != getattr(self, '_last_batch_id', None):
                setattr(self, '_last_batch_id', batch_id)
                should_emit = True
                logger.debug(f"New batch ID: {batch_id}")
            
            # Update the last emotion state regardless of whether we emit
            self.last_emotion_update['emotion'] = emotion
            self.last_emotion_update['confidence'] = confidence
            self.last_emotion_update['timestamp'] = current_time
            
            # Update metrics (always do this regardless of emission)
            self.metrics['current']['emotion'] = emotion
            self.metrics['current']['emotion_confidence'] = confidence
            
            if emotion in self.metrics['history']['emotions']:
                self.metrics['history']['emotions'][emotion] += 1
            
            # Skip emission if we shouldn't emit or if no clients connected
            num_clients = len(self.socketio.server.manager.rooms.get('/', {}))
            if not should_emit or num_clients == 0:
                return
                
            # Mark as emitted
            self.last_emotion_update['emitted'] = True
            
            # Create message for the log
            message = self._get_emotion_message(emotion, confidence)
            
            # Add to emotion log
            self._add_to_emotion_log({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'emotion': emotion,
                'confidence': confidence,
                'message': message
            })
            
            # Ensure confidences is included in the emit data
            emit_data = {
                'emotion': emotion,
                'confidence': confidence,
                'message': message,
                'metrics': self.metrics,
                'confidences': confidences  # Make sure this is included
            }
            
            # Only log socket emission once per 10 seconds to reduce noise
            static_log_interval = 10.0  # seconds
            static_log_key = 'last_socket_log_time'
            
            if not hasattr(self, static_log_key) or current_time - getattr(self, static_log_key) >= static_log_interval:
                logger.info(f"Socket.IO emit: emotion_update with {num_clients} clients connected")
                setattr(self, static_log_key, current_time)
            
            # Emit to all connected clients
            self.socketio.emit('emotion_update', emit_data, namespace='/')
            
        except Exception as e:
            logger.error(f"Error emitting emotion update: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _get_emotion_message(self, emotion, confidence):
        """Generate a human-readable message for the emotion update"""
        if confidence < 0.3:
            return f"Possible {emotion} detected (low confidence)"
        elif confidence < 0.7:
            return f"{emotion.capitalize()} detected"
        else:
            return f"Strong {emotion} detected" 