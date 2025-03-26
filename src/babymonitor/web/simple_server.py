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
eventlet.monkey_patch()  # Patch standard library for eventlet compatibility
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, redirect, url_for, request
from flask_socketio import SocketIO
import numpy as np
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
    def __init__(self, camera=None, person_detector=None, emotion_detector=None, host='0.0.0.0', port=5000, debug=False, mode="normal"):
        """
        Initialize the web server
        
        Args:
            camera: Camera instance
            person_detector: Person detector instance
            emotion_detector: Emotion detector instance
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
            mode: Web interface mode ('normal' or 'dev')
        """
        self.host = host
        self.port = find_free_port(port)  # Find an available port
        self.debug = debug
        self.mode = mode
        self.camera = camera
        self.person_detector = person_detector
        self.emotion_detector = emotion_detector
        
        # Create Flask app and Socket.IO
        self.app = Flask(__name__, 
                        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                        static_folder=os.path.join(os.path.dirname(__file__), 'static'))
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='eventlet')
        
        # Frame buffer
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            'current': {
                'fps': 0,
                'detections': 0,
                'emotion': 'unknown',
                'emotion_confidence': 0.0,
            },
            'history': {
                'emotions': {
                    'crying': 0,
                    'laughing': 0,
                    'babbling': 0,
                    'silence': 0,
                    'unknown': 0
                }
            },
            'detection_types': {
                'face': 0,
                'upper_body': 0,
                'full_body': 0
            },
            'total_detections': 0,
        }
        
        # Initialize emotion distribution with supported emotions
        if emotion_detector and hasattr(emotion_detector, 'emotions'):
            self.metrics['history']['emotions'] = {emotion: 0 for emotion in emotion_detector.emotions}
        
        # System status
        self.system_status = {
            'uptime': '00:00:00',
            'cpu_usage': 0,
            'memory_usage': 0,
            'camera_status': 'connected' if self.camera else 'disconnected',
            'person_detector_status': 'running' if self.person_detector else 'stopped',
            'emotion_detector_status': 'running' if self.emotion_detector else 'stopped',
        }
        
        # Activity log for emotion events
        self.emotion_log = []
        self.max_log_entries = 50  # Keep last 50 entries
        
        # Running flag
        self.running = False
        
        # Start time
        self.start_time = time.time()
        
        # Setup routes
        self._setup_routes()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        
        # Start system status update thread
        self.status_thread = threading.Thread(target=self._update_system_status)
        self.status_thread.daemon = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize alerts list
        self.alerts = []
        
    def _setup_routes(self):
        """Setup Flask routes"""
        
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
        import cv2
        
        # Create a blank frame for when no camera feed is available
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "No camera feed available", (120, 240), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, blank_buffer = cv2.imencode('.jpg', blank_frame)
        blank_bytes = blank_buffer.tobytes()
        
        while True:
            try:
                with self.frame_lock:
                    if self.frame_buffer is None:
                        # If no frame is available, yield a blank frame
                        yield (b'--frame\r\n'
                             b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
                    else:
                        # Yield the frame
                        yield (b'--frame\r\n'
                             b'Content-Type: image/jpeg\r\n\r\n' + self.frame_buffer + b'\r\n')
                
                # Sleep to control frame rate
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                logger.error(f"Error generating frames: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(0.1)
    
    def _add_to_emotion_log(self, entry):
        """Add an entry to the emotion log"""
        self.emotion_log.append(entry)
        # Trim log if it gets too big
        if len(self.emotion_log) > self.max_log_entries:
            self.emotion_log = self.emotion_log[-self.max_log_entries:]
    
    def _process_frames(self):
        """Process frames from camera"""
        import cv2
        
        # Initialize detection history for smoothing
        detection_history = []
        max_history = 5
        last_fps_time = time.time()
        frame_count = 0
        
        # Set initial emotion timestamp
        last_emotion_log_time = time.time()
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.metrics['current']['fps'] = frame_count
                    frame_count = 0
                    last_fps_time = current_time
                
                # Process frame with person detector
                results = self.person_detector.process_frame(frame)
                
                # Get the processed frame with bounding boxes and use it for display
                processed_frame = results.get('frame', frame)
                
                # Get emotion detection results
                if self.emotion_detector:
                    try:
                        emotion_results = self.emotion_detector.get_current_state()
                        if emotion_results:
                            current_emotion = emotion_results['emotion']
                            confidence = emotion_results['confidence']
                            
                            # Only log emotion changes or high confidence emissions
                            should_log = False
                            
                            # Log emotion changes
                            if current_emotion != self.metrics['current'].get('emotion', 'unknown'):
                                should_log = True
                                
                            # Log high confidence emotions periodically
                            elif confidence > 0.7 and (current_time - last_emotion_log_time) > 60:  # Log once per minute
                                should_log = True
                                last_emotion_log_time = current_time
                            
                            # Update current emotion metrics
                            self.metrics['current']['emotion'] = current_emotion
                            self.metrics['current']['emotion_confidence'] = confidence
                            
                            # Update emotion history
                            if current_emotion in self.metrics['history']['emotions']:
                                self.metrics['history']['emotions'][current_emotion] += 1
                            
                            # Add to emotion log if needed
                            if should_log:
                                log_entry = {
                                    'timestamp': current_time,
                                    'emotion': current_emotion,
                                    'confidence': confidence,
                                    'message': self._get_emotion_message(current_emotion, confidence)
                                }
                                self._add_to_emotion_log(log_entry)
                            
                            # Emit emotion update via Socket.IO
                            self.socketio.emit('emotion_update', {
                                'emotion': current_emotion,
                                'confidence': confidence,
                                'confidences': emotion_results['confidences']
                            })
                    except Exception as e:
                        logger.error(f"Error getting emotion state: {e}")
                
                # Update detection metrics and emit update
                if 'detections' in results:
                    detection_count = len(results['detections'])
                    # Update metrics
                    self.metrics['current']['detections'] = detection_count
                    
                    # Update detection types
                    if detection_count > 0:
                        # Increment full-body count for YOLOv8 detections
                        self.metrics['detection_types']['full_body'] = self.metrics['detection_types'].get('full_body', 0) + detection_count
                        
                    # Emit detection update via Socket.IO
                    self.socketio.emit('detection_update', {
                        'count': detection_count,
                        'fps': self.metrics['current']['fps'],
                        'detections': results['detections']  # Include actual detection data
                    })
                
                # Clean up old alerts
                current_time = time.time()
                self.alerts = [a for a in self.alerts if current_time - a['timestamp'] < 60]
                
                # Update alert and notification for crying
                if self.metrics['current']['emotion'] == 'crying' and self.metrics['current']['emotion_confidence'] > 0.7:
                    # Check if we need to add a new alert
                    if not self.alerts or (current_time - self.alerts[-1]['timestamp'] > 10 and self.alerts[-1]['type'] != 'crying'):
                        alert = {
                            'message': 'Baby is crying with high confidence!',
                            'type': 'crying',
                            'timestamp': current_time
                        }
                        self.alerts.append(alert)
                        self.socketio.emit('alert', alert)
                
                # Encode and store frame
                with self.frame_lock:
                    # Encode the processed frame with bounding boxes
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    self.frame_buffer = buffer.tobytes()
                
                # Sleep to control CPU usage
                time.sleep(0.02)  # 50 FPS maximum
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(0.1)
    
    def _get_emotion_message(self, emotion, confidence):
        """Get a human-readable message for an emotion detection"""
        if emotion == 'crying':
            if confidence > 0.85:
                return "Baby is crying loudly! Needs immediate attention."
            elif confidence > 0.7:
                return "Baby is crying. May need attention."
            else:
                return "Baby might be starting to cry."
        elif emotion == 'laughing':
            if confidence > 0.8:
                return "Baby is happily laughing!"
            else:
                return "Baby is making happy sounds."
        elif emotion == 'babbling':
            return "Baby is babbling or talking."
        elif emotion == 'silence':
            return "Baby is quiet."
        else:
            return f"Detected {emotion}."
    
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
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error updating system status: {e}")
                time.sleep(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        
    def run(self):
        """Run the web server in the current thread"""
        self.running = True
        logger.info(f"Running Baby Monitor Web Server in {self.mode.upper()} mode on http://{self.host}:{self.port}")
        
        # Start processing thread if not already running
        if not self.processing_thread.is_alive():
            self.processing_thread.start()
        
        # Start system status thread if not already running
        if not self.status_thread.is_alive():
            self.status_thread.start()
        
        try:
            # Run the Flask app with Socket.IO
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False  # Disable reloader to prevent duplicate processes
            )
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            self.stop()
        except Exception as e:
            logger.error(f"Error running web server: {e}")
            self.stop()
    
    def stop(self):
        """Stop the web server"""
        if not self.running:
            return
            
        self.running = False
        logger.info("Stopping Baby Monitor Web Server")
        
        try:
            # Stop Socket.IO
            if hasattr(self, 'socketio'):
                self.socketio.stop()
            
            # Stop camera if it's running
            if self.camera and hasattr(self.camera, 'release'):
                self.camera.release()
            
            # Stop emotion detector if it's running
            if self.emotion_detector and hasattr(self.emotion_detector, 'stop'):
                self.emotion_detector.stop()
            
            # Stop person detector if it's running
            if self.person_detector and hasattr(self.person_detector, 'stop'):
                self.person_detector.stop()
            
            # Wait for threads to finish
            if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            if hasattr(self, 'status_thread') and self.status_thread.is_alive():
                self.status_thread.join(timeout=1.0)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
        finally:
            logger.info("Web server stopped")
            os._exit(0)  # Force exit if normal shutdown fails 