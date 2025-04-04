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
from typing import Tuple, Dict, Any

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
    def __init__(self, camera=None, person_detector=None, emotion_detector=None, host='0.0.0.0', port=5000, mode='normal', debug=False, stop_event=None):
        """Initialize the web interface"""
        self.camera = camera
        self.person_detector = person_detector
        self.emotion_detector = emotion_detector
        self.host = host
        self.port = port
        self.mode = mode
        self.debug = debug
        self.stop_event = stop_event  # Store the stop event for graceful shutdown
        
        # Set up Flask app
        self.app = Flask(__name__, 
                         static_folder=os.path.join(os.path.dirname(__file__), 'static'),
                         template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
        
        # Configure logging first
        self._configure_logging()
        
        self.logger.info("Initializing SimpleBabyMonitorWeb")
        
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
        
        # Initialize Socket.IO
        self._init_socketio()
        
        # Initialize routes
        self._setup_routes()
        
        # Running flag
        self.running = True
        
        # Frame resize parameters
        self.resize_frame = True
        self.max_width = 640  # Maximum width for display frames
        
        # Frame buffer
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        
        # Create a blank frame for when no camera feed is available
        self.blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.blank_frame, "No camera feed available", (120, 240), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, self.blank_buffer = cv2.imencode('.jpg', self.blank_frame)
        self.blank_bytes = self.blank_buffer.tobytes()
        
        # Emotion log
        self.emotion_log = []
        self.max_log_entries = 100
        
        # System status
        self.system_status = {
            'uptime': '00:00:00',
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'camera_status': 'unknown',
            'person_detector_status': 'unknown',
            'emotion_detector_status': 'unknown'
        }
        
        # Start time
        self.start_time = time.time()
        
        # Store the greenlets for cleanup
        self.greenlets = []
        
        # Make sure attributes exist for controlling behavior in derived classes
        if not hasattr(self, 'keep_history'):
            self.keep_history = True
        if not hasattr(self, 'max_history'):
            self.max_history = 100
        
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
        """Configure logging for the web server"""
        # Set up logging
        self.logger = logging.getLogger('SimpleBabyMonitorWeb')
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        # Create handlers
        if not self.logger.handlers:
            # Only add handlers if they don't exist yet
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG if self.debug else logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

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
            """Route for video feed"""
            self.logger.debug("Video feed requested")
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for getting current metrics."""
            # Get current metrics
            metrics = self.metrics.copy()
            
            # Calculate uptime
            uptime_seconds = time.time() - self.start_time
            metrics['uptime_seconds'] = uptime_seconds
            metrics['uptime_formatted'] = self._format_uptime(uptime_seconds)
            
            # Add system resources
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            metrics['cpu_usage'] = cpu_percent
            metrics['memory_usage'] = memory.percent
            metrics['memory_available'] = memory.available / (1024 * 1024)  # MB
            metrics['memory_total'] = memory.total / (1024 * 1024)  # MB
            
            # Add emotion history summary if available
            if self.emotion_history:
                # Get the last 10 entries (or fewer if not available)
                recent_history = self.emotion_history[-10:]
                emotion_counts = {}
                
                for entry in recent_history:
                    emotion = entry.get('emotion', 'unknown')
                    if emotion in emotion_counts:
                        emotion_counts[emotion] += 1
                    else:
                        emotion_counts[emotion] = 1
                
                metrics['recent_emotions'] = emotion_counts
                
                # Calculate average confidence of recent emotions
                confidences = [entry.get('confidence', 0) for entry in recent_history if 'confidence' in entry]
                if confidences:
                    metrics['avg_emotion_confidence'] = sum(confidences) / len(confidences)
                else:
                    metrics['avg_emotion_confidence'] = 0
            
            # Add person detection summaries
            if 'person_states' in metrics and metrics['person_states']:
                # Add count of each state
                state_counts = {}
                for state in metrics['person_states']:
                    if state in state_counts:
                        state_counts[state] += 1
                    else:
                        state_counts[state] = 1
                
                metrics['state_counts'] = state_counts
                
                # Determine most common state
                if state_counts:
                    most_common_state = max(state_counts.items(), key=lambda x: x[1])[0]
                    metrics['most_common_state'] = most_common_state
            
            # Return metrics as JSON
            return jsonify(self._make_json_serializable(metrics))
        
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
        
        @self.app.route('/api/person/status')
        def api_person_status():
            """API endpoint to check the status of the person detector model."""
            if self.person_detector:
                status = self.person_detector.check_model()
                return jsonify(status)
            else:
                return jsonify({
                    "status": "not_available",
                    "message": "Person detector not initialized"
                })
                
        @self.app.route('/api/person/reload', methods=['POST'])
        def api_reload_person_model():
            """API endpoint to reload the person detector model."""
            if self.person_detector:
                result = self.person_detector.reload_model()
                return jsonify(result)
            else:
                return jsonify({
                    "status": "error",
                    "message": "Person detector not initialized"
                })
        
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
        """Generate frames for the video feed."""
        self.logger.info("Starting frame generation")
        
        while True:  # Always run, regardless of self.running
            try:
                # Check if camera is available
                if not self.camera or not self.camera.is_opened():
                    # Yield a placeholder frame
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder, 
                        "Camera not available", 
                        (120, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255, 255, 255), 
                        2
                    )
                    _, jpeg = cv2.imencode('.jpg', placeholder)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    eventlet.sleep(1)
                    continue
                
                # Try to get frame from the frame buffer first (if available)
                with self.frame_lock:
                    if self.frame_buffer is not None:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + self.frame_buffer + b'\r\n')
                        eventlet.sleep(0.03)  # ~30 FPS
                        continue
                
                # If no frame buffer, read directly from camera
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    # Yield a placeholder frame
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder, 
                        "No frame available", 
                        (120, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255, 255, 255), 
                        2
                    )
                    _, jpeg = cv2.imencode('.jpg', placeholder)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    eventlet.sleep(0.1)
                    continue
                
                # Process the frame only if we're not using the frame buffer
                try:
                    processed_frame, _ = self._process_single_frame(frame)
                except Exception as e:
                    self.logger.error(f"Error processing frame for stream: {str(e)}")
                    processed_frame = frame  # Use original frame if processing fails
                
                # Encode the frame as JPEG
                _, jpeg = cv2.imencode('.jpg', processed_frame)
                
                # Yield the frame in multipart HTTP response format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                # Sleep to control frame rate
                eventlet.sleep(0.03)  # ~30 FPS
                
                # Check if we should stop
                if self.stop_event and self.stop_event.is_set():
                    self.logger.info("Stop event detected in frame generation")
                    break
            
            except Exception as e:
                self.logger.error(f"Error generating frames: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                
                # Create an error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                error_message = f"Error: {str(e)}"
                cv2.putText(
                    error_frame, 
                    error_message if len(error_message) < 40 else error_message[:37] + "...", 
                    (20, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
                _, jpeg = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                eventlet.sleep(0.5)  # Sleep longer if there's an error
    
    def _add_to_emotion_log(self, entry):
        """Add an entry to the emotion log"""
        self.emotion_log.append(entry)
        # Trim log if it gets too big
        if len(self.emotion_log) > self.max_log_entries:
            self.emotion_log = self.emotion_log[-self.max_log_entries:]
    
    def _process_frames(self):
        """Background thread for processing frames from camera."""
        try:
            # Camera frame processing loop
            self.logger.info("Starting frame processing")
                
            frame_counter = 0
            last_fps_time = time.time()
            
            # Initialize person variables
            person_detected = False
            person_confidence = 0.0
            bounding_boxes = []
            
            while not self.stop_event.is_set():
                try:
                    # Acquire frame from camera
                    if self.camera:
                        frame = self.camera.get_frame()
                        if frame is None or frame.size == 0:
                            self.logger.warning("Received empty frame, waiting for next frame")
                        eventlet.sleep(0.1)
                        continue
                    else:
                        # Generate a placeholder frame if no camera available
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "No Camera Available", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Create a copy of the frame for display
                    display_frame = frame.copy()
                    
                    # Process frame with detectors
                    if self.person_detector:
                        try:
                            # Reset the bounding boxes array for new frame
                            bounding_boxes = []
                            
                            # Process frame with person detector
                            processing_result = self.person_detector.process_frame(frame)
                            
                            # Get detection results
                            person_detected = processing_result.get('person_detected', False)
                            person_confidence = processing_result.get('confidence', 0.0)
                            detections = processing_result.get('detections', [])
                            
                            # Extract bounding boxes and get state information
                            if detections:
                                # Get person state information
                                state_results = self.person_detector.detect_person_state(frame, detections)
                                person_states = state_results.get('states', [])
                                
                                # Process each detection
                                for idx, detection in enumerate(detections):
                                    if 'bbox' in detection:
                                        # Format is [x1, y1, x2, y2]
                                        bbox = detection['bbox']
                                        conf = detection.get('confidence', 0.0)
                                            
                                        # Get state for this person (if available)
                                        person_state = 'unknown'
                                        if idx < len(person_states):
                                            person_state = person_states[idx]
                                        
                                        # Store person number with the bounding box
                                        bounding_boxes.append((bbox, conf, idx + 1, person_state))
                        
                            # Define colors for different states
                            state_colors = {
                                'seated': (0, 255, 0),    # Green
                                'lying': (255, 165, 0),   # Orange
                                'moving': (0, 0, 255),    # Red
                                'standing': (255, 0, 0),  # Blue
                                'unknown': (128, 128, 128)  # Gray
                            }
                            
                            # Draw bounding boxes on display frame
                            if bounding_boxes:
                                for (bbox, conf, person_num, state) in bounding_boxes:
                                    x1, y1, x2, y2 = bbox
                                    # Get color based on state
                                    color = state_colors.get(state, (0, 255, 0))
                                    
                                    # Draw rectangle
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                    
                                    # Draw person number and state
                                    label = f"Person {person_num}: {state.capitalize()}"
                                    cv2.putText(display_frame, label, (x1, y1-10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                            
                                    # Draw confidence below
                                    cv2.putText(display_frame, f"Conf: {conf:.2f}", (x1, y2+15), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            elif person_detected:
                                # If no bounding boxes but person detected, draw general indicator
                                cv2.putText(display_frame, "Person Detected", (10, 30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        except Exception as e:
                            self.logger.error(f"Error processing frame with person detector: {str(e)}")
                    
                    # Update metrics with person detection data
                    self.metrics['person_detected'] = person_detected
                    self.metrics['person_confidence'] = person_confidence
                    self.metrics['bounding_boxes'] = len(bounding_boxes)
                    
                    # Update metrics with state information if available
                    if bounding_boxes:
                        # Extract states from all detections
                        states = [state for _, _, _, state in bounding_boxes]
                        
                        # Count instances of each state
                        from collections import Counter
                        state_count = Counter(states)
                        
                        # Store state counts in metrics
                        self.metrics['person_states'] = states
                        self.metrics['state_counts'] = {state: count for state, count in state_count.items()}
                        
                        # Get the most common state
                        if state_count:
                            most_common_state, _ = state_count.most_common(1)[0]
                            self.metrics['primary_state'] = most_common_state
                    
                    # Add timestamp
                    current_time = time.strftime("%H:%M:%S")
                    cv2.putText(display_frame, current_time, (10, display_frame.shape[0] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Encode image to JPEG
                    _, jpeg = cv2.imencode('.jpg', display_frame)
                    self.frame_buffer = jpeg.tobytes()
                    
                    # Update FPS calculation
                    frame_counter += 1
                    current_time = time.time()
                    time_diff = current_time - last_fps_time
                    
                    if time_diff >= 1.0:  # Update FPS every second
                        fps = frame_counter / time_diff
                        self.metrics['fps'] = round(fps, 1)
                        self.logger.debug(f"Processing frames at {fps:.1f} FPS")
                        frame_counter = 0
                        last_fps_time = current_time
                    
                    # Don't hog the CPU
                    eventlet.sleep(0.01)
                except Exception as e:
                    self.logger.error(f"Error in frame processing loop: {str(e)}")
                    eventlet.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Fatal error in frame processing thread: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info("Frame processing stopped")
    
    def _process_single_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single frame from the camera feed.
        
        Args:
            frame (np.ndarray): Raw frame from the camera
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Processed frame and metrics
        """
        metrics_update = {}
        
        # Skip processing if frame is None or empty
        if frame is None or frame.size == 0:
            self.logger.warning("Received empty frame")
            return frame, metrics_update
            
        # Initialize variables
        person_detected = False
        person_confidence = 0.0
        bounding_boxes = []
        person_states = []
        
        try:
            # Use the person detector
            if self.person_detector:
                detection_start = time.time()
                
                # Process frame with person detector
                detection_results = self.person_detector.process_frame(frame)
                
                    # Update metrics
                detection_time = time.time() - detection_start
                metrics_update['detection_time'] = detection_time
                
                # Extract results
                if detection_results:
                    person_detected = detection_results.get('person_detected', False)
                    person_confidence = detection_results.get('confidence', 0.0)
                    
                    # Get bounding boxes if available
                    if 'detections' in detection_results and detection_results['detections']:
                        # Get person state information
                        state_results = self.person_detector.detect_person_state(frame, detection_results['detections'])
                        metrics_update['person_state'] = state_results.get('state', 'unknown')
                        metrics_update['state_confidence'] = state_results.get('confidence', 0.0)
                        metrics_update['movement_level'] = state_results.get('movement_level', 0.0)
                        metrics_update['person_states'] = state_results.get('states', [])
                        
                        # Store states for each detection
                        person_states = state_results.get('states', [])
                        
                        # Process each detection
                        for idx, detection in enumerate(detection_results['detections']):
                            if 'bbox' in detection:
                                # Store bounding box and confidence
                                box = detection['bbox']  # [x1, y1, x2, y2]
                                conf = detection.get('confidence', 0.0)
                                
                                # Get state for this person (if available)
                                person_state = 'unknown'
                                if idx < len(person_states):
                                    person_state = person_states[idx]
                                    
                                # Store person number with the bounding box
                                bounding_boxes.append((box, conf, idx + 1, person_state))
                
                # Update metrics
                metrics_update['person_detected'] = person_detected
                metrics_update['person_confidence'] = person_confidence
                self.logger.debug(f"Person detected: {person_detected}, confidence: {person_confidence:.2f}")
                
                # Log bounding boxes for debugging
                if bounding_boxes:
                    self.logger.debug(f"Found {len(bounding_boxes)} bounding boxes")
                elif person_detected:
                    self.logger.debug("Person detected but no bounding boxes returned")
        except Exception as e:
            self.logger.error(f"Error in person detection: {str(e)}")
            
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Resize if needed
        if hasattr(self, 'resize_frame') and self.resize_frame and display_frame.shape[1] > self.max_width:
            scale = self.max_width / display_frame.shape[1]
            new_width = int(display_frame.shape[1] * scale)
            new_height = int(display_frame.shape[0] * scale)
            display_frame = cv2.resize(display_frame, (new_width, new_height))
        elif not hasattr(self, 'resize_frame') and display_frame.shape[1] > 640:
            # Default resize if attribute not set
            display_frame = cv2.resize(display_frame, (640, 480))
            
        # Draw bounding boxes if person detected
        if person_detected:
            # Draw specific bounding boxes if available
            if bounding_boxes:
                # Define colors for different states
                state_colors = {
                    'seated': (0, 255, 0),    # Green
                    'lying': (255, 165, 0),   # Orange
                    'moving': (0, 0, 255),    # Red
                    'standing': (255, 0, 0),  # Blue
                    'unknown': (128, 128, 128)  # Gray
                }
                
                for idx, (box, conf, person_num, state) in enumerate(bounding_boxes):
                    # Scale box coordinates if frame was resized
                    if hasattr(self, 'resize_frame') and self.resize_frame and frame.shape[1] > self.max_width:
                        scale = self.max_width / frame.shape[1]
                        x1, y1, x2, y2 = box
                        x1 = int(x1 * scale)
                        y1 = int(y1 * scale)
                        x2 = int(x2 * scale)
                        y2 = int(y2 * scale)
                        scaled_box = [x1, y1, x2, y2]
                    elif not hasattr(self, 'resize_frame') and display_frame.shape[1] == 640:
                        # Default scale if attribute not set
                        scale = 640 / frame.shape[1]
                        x1, y1, x2, y2 = box
                        x1 = int(x1 * scale)
                        y1 = int(y1 * scale)
                        x2 = int(x2 * scale)
                        y2 = int(y2 * scale)
                        scaled_box = [x1, y1, x2, y2]
                    else:
                        scaled_box = box
                        
                    # Get color based on state
                    color = state_colors.get(state, (0, 255, 0))
                    
                    # Draw rectangle with color based on state
                    cv2.rectangle(
                        display_frame,
                        (int(scaled_box[0]), int(scaled_box[1])),
                        (int(scaled_box[2]), int(scaled_box[3])),
                        color,
                        2
                    )
                    
                    # Draw person number and state
                    label = f"Person {person_num}: {state.capitalize()}"
                    cv2.putText(
                        display_frame,
                        label,
                        (int(scaled_box[0]), int(scaled_box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
                    
                    # Draw confidence below
                    cv2.putText(
                        display_frame,
                        f"Conf: {conf:.2f}",
                        (int(scaled_box[0]), int(scaled_box[3]) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1
                    )
            else:
                # If no specific boxes but person detected, draw a general detection indicator
                cv2.rectangle(
                    display_frame,
                    (10, 10),
                    (display_frame.shape[1] - 10, display_frame.shape[0] - 10),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    display_frame,
                    f"Person: {person_confidence:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
        # Process with emotion detector if enabled
        if self.emotion_detector:
            try:
                # Convert BGR to RGB for audio processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with emotion detector
                emotion_start = time.time()
                emotion_results = self.emotion_detector.process_frame(rgb_frame)
                emotion_time = time.time() - emotion_start
                
                # Update metrics
                metrics_update['emotion_time'] = emotion_time
                if emotion_results:
                    for key, value in emotion_results.items():
                        metrics_update[key] = value
                        
                    # Add emotion text if detected
                    if 'emotion' in emotion_results and emotion_results['emotion']:
                        emotion = emotion_results['emotion']
                        confidence = emotion_results.get('emotion_confidence', 0)
                        
                        # Display emotion on frame
                        cv2.putText(
                            display_frame,
                            f"{emotion}: {confidence:.2f}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2
                        )
            except Exception as e:
                self.logger.error(f"Error in emotion detection: {str(e)}")
                
        return display_frame, metrics_update
    
    def _update_system_status(self):
        """Update system status periodically"""
        self.logger.info("Starting system status updates")
        try:
            while self.running and (not self.stop_event or not self.stop_event.is_set()):
                # Calculate uptime
                uptime_seconds = time.time() - self.start_time
                hours, remainder = divmod(uptime_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                self.system_status['uptime'] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                
                # Get CPU and memory usage
                self.system_status['cpu_usage'] = psutil.cpu_percent()
                self.system_status['memory_usage'] = psutil.virtual_memory().percent
                
                # Check camera status
                self.system_status['camera_status'] = 'connected' if self.camera and self.camera.is_opened() else 'disconnected'
                
                # Check detector status
                self.system_status['person_detector_status'] = 'running' if self.person_detector else 'stopped'
                self.system_status['emotion_detector_status'] = 'running' if self.emotion_detector else 'stopped'
                
                # Emit to clients
                try:
                    self.socketio.emit('system_status', self.system_status)
                except Exception as e:
                    self.logger.error(f"Error emitting system status: {str(e)}")
                
                # Sleep for 5 seconds before updating again
                eventlet.sleep(5)
        except Exception as e:
            self.logger.error(f"Error updating system status: {str(e)}")
        finally:
            self.logger.info("System status updates stopped")
    
    def run(self, stop_event=None):
        """Run the web server"""
        self.logger.info(f"Running Baby Monitor Web Server in {self.mode.upper()} mode on http://{self.host}:{self.port}")
        
        if stop_event:
            self.stop_event = stop_event
        
        # Start system status update thread
        self.logger.info("Starting system status updates")
        status_updater = eventlet.spawn(self._update_system_status)
        self.greenlets.append(status_updater)
        
        # Start frame processing thread
        self.logger.info("Starting frame processing")
        frame_processor = eventlet.spawn(self._process_frames)
        self.greenlets.append(frame_processor)
        
        if not self.stop_event or not self.stop_event.is_set():
            # Set up signal handler to allow graceful shutdown
            def handle_signal(sig, frame):
                self.logger.info("Signal received in web server, shutting down...")
                self.stop()
                
                # If we have a stop_event, set it to notify the main application
                if self.stop_event:
                    self.stop_event.set()
            
            # Set up signal handlers for SIGINT and SIGTERM
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
            
            # Run the server
            try:
                # Run the SocketIO server
                self.socketio.run(
                    self.app,
                    host=self.host,
                    port=self.port,
                    debug=self.debug
                )
            except Exception as e:
                self.logger.error(f"Error running web server: {str(e)}")
            finally:
                self.stop()
        else:
            self.logger.info("Stop event already set, not starting web server")
    
    def stop(self):
        """Stop the web server"""
        if not self.running:
            return
            
        self.running = False
        logger.info("Stopping Baby Monitor Web Server")
        
        # Kill all greenlets
        for greenlet in self.greenlets:
            try:
                if greenlet and not greenlet.dead:
                    greenlet.kill()
            except Exception as e:
                logger.error(f"Error killing greenlet: {str(e)}")
        
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
                self.logger.warning(f"Invalid emotion_data format: {type(emotion_data)}")
                return
                
            # Convert emotion_data to ensure all values are JSON serializable
            emotion_data_serializable = self._make_json_serializable(emotion_data)
                
            emotion = emotion_data_serializable.get('emotion', 'unknown')
            confidence = emotion_data_serializable.get('confidence', 0.0)
            confidences = emotion_data_serializable.get('confidences', {})
            
            # Get current time
            current_time = time.time()
            
            # Check if we should throttle this update
            should_emit = False
            
            # Get the message queue ID for deduplication
            batch_id = emotion_data_serializable.get('batch_id', None)
            
            # Only log and emit if:
            # 1. The emotion has changed from the last emitted one
            # 2. OR significant confidence change (>10%)
            # 3. OR enough time has passed since last emission (at least 1 second)
            # 4. OR we have a new batch ID (to avoid duplicate messages from same audio batch)
            
            # Check if emotion changed
            if emotion != self.last_emotion_update['emotion']:
                should_emit = True
                self.logger.info(f"Emotion changed from {self.last_emotion_update['emotion']} to {emotion}")
            # Check if confidence changed significantly
            elif abs(confidence - self.last_emotion_update['confidence']) > 0.1:
                should_emit = True
                self.logger.info(f"Confidence changed significantly: {self.last_emotion_update['confidence']:.2f} -> {confidence:.2f}")
            # Check time interval - increased to 1 second minimum between general updates
            elif current_time - self.last_emotion_update['timestamp'] >= 1.0:
                should_emit = True
                # Only log occasional updates (every 5 seconds) to reduce noise
                if current_time - self.last_emotion_update['timestamp'] >= 5.0:
                    self.logger.info(f"Periodic emotion update: {emotion} ({confidence:.4f})")
            # Check for duplicate batch IDs
            elif batch_id and batch_id != getattr(self, '_last_batch_id', None):
                setattr(self, '_last_batch_id', batch_id)
                should_emit = True
                self.logger.debug(f"New batch ID: {batch_id}")
            
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
            
            # Create JSON-serializable emit data
            emit_data = {
                'emotion': emotion,
                'confidence': confidence,
                'message': message,
                'metrics': self._make_json_serializable(self.metrics),
                'confidences': confidences
            }
            
            # Only log socket emission once per 10 seconds to reduce noise
            static_log_interval = 10.0  # seconds
            static_log_key = 'last_socket_log_time'
            
            if not hasattr(self, static_log_key) or current_time - getattr(self, static_log_key) >= static_log_interval:
                self.logger.info(f"Socket.IO emit: emotion_update with {num_clients} clients connected")
                setattr(self, static_log_key, current_time)
            
            # Emit to all connected clients
            self.socketio.emit('emotion_update', emit_data, namespace='/')
            
        except Exception as e:
            self.logger.error(f"Error emitting emotion update: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _get_emotion_message(self, emotion, confidence):
        """Generate a human-readable message for the emotion update"""
        if confidence < 0.3:
            return f"Possible {emotion} detected (low confidence)"
        elif confidence < 0.7:
            return f"{emotion.capitalize()} detected"
        else:
            return f"Strong {emotion} detected"
    
    def _make_json_serializable(self, obj):
        """Convert any object to a JSON serializable format recursively"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # For any other types, convert to string
            return str(obj) 