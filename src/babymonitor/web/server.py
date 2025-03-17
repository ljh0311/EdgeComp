"""
Baby Monitor Web Server
"""

import os
import json
import time
import threading
import logging
import psutil
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, redirect, url_for, request
from flask_socketio import SocketIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BabyMonitorWeb')

class BabyMonitorWeb:
    """
    Web interface for the Baby Monitor System
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
        self.port = port
        self.debug = debug
        self.mode = mode
        self.camera = camera
        self.person_detector = person_detector
        self.emotion_detector = emotion_detector
        
        # Create Flask app
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Frame buffer
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            'current': {
                'fps': 0,
                'detections': 0,
                'cpu_usage': 0,
                'memory_usage': 0,
                'emotion': 'unknown',
                'emotion_confidence': 0.0,
            },
            'history': {
                'fps': [],
                'detections': [],
                'cpu_usage': [],
                'memory_usage': [],
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
            'detection_log': []
        }
        
        # Running flag
        self.running = False
        
        # Start time
        self.start_time = time.time()
        
        # Setup routes and socket.io events
        self._setup_routes()
        self._setup_socketio()
        
        # Start metrics emission thread
        self.metrics_thread = threading.Thread(target=self._emit_metrics)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
        
        # Start system info emission thread
        self.system_info_thread = threading.Thread(target=self._emit_system_info)
        self.system_info_thread.daemon = True
        self.system_info_thread.start()
        
        logger.info(f"Baby Monitor Web Server initialized in {self.mode.upper()} mode")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            # In dev mode, redirect to metrics page
            if self.mode == "dev":
                return redirect(url_for('metrics'))
            # In normal mode, show the main dashboard
            return render_template('index.html', mode=self.mode)
        
        @self.app.route('/metrics')
        def metrics():
            # In both modes, show metrics but with different access levels
            return render_template('metrics.html', mode=self.mode)
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/metrics')
        def api_metrics():
            return jsonify(self.metrics)
        
        @self.app.route('/api/system_info')
        def api_system_info():
            uptime = time.time() - self.start_time
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            return jsonify({
                'uptime': f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
                'mode': self.mode
            })
        
        # Developer tools routes - only accessible in dev mode
        @self.app.route('/dev/tools')
        def dev_tools():
            if self.mode != "dev":
                return redirect(url_for('index'))
            return render_template('dev_tools.html', mode=self.mode)
        
        @self.app.route('/dev/logs')
        def dev_logs():
            if self.mode != "dev":
                return redirect(url_for('index'))
            
            # Get the last 100 lines from the log file
            log_file = os.path.join('logs', 'baby_monitor.log')
            log_lines = []
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_lines = f.readlines()[-100:]
            
            return render_template('logs.html', mode=self.mode, logs=log_lines)
        
        @self.app.route('/dev/settings')
        def dev_settings():
            if self.mode != "dev":
                return redirect(url_for('index'))
            return render_template('settings.html', mode=self.mode)
        
        # Repair tools - accessible in both modes
        @self.app.route('/repair')
        def repair_tools():
            return render_template('repair_tools.html', mode=self.mode)
        
        @self.app.route('/repair/run', methods=['POST'])
        def run_repair():
            repair_type = request.json.get('type')
            
            if repair_type == 'camera':
                # Attempt to restart the camera
                if self.camera:
                    try:
                        self.camera.release()
                        time.sleep(1)
                        self.camera.start()
                        return jsonify({'success': True, 'message': 'Camera restarted successfully'})
                    except Exception as e:
                        return jsonify({'success': False, 'message': f'Failed to restart camera: {str(e)}'})
            
            elif repair_type == 'audio':
                # Attempt to restart the audio processor
                if self.emotion_detector and hasattr(self.emotion_detector, 'audio_processor'):
                    try:
                        self.emotion_detector.audio_processor.stop()
                        time.sleep(1)
                        self.emotion_detector.audio_processor.start()
                        return jsonify({'success': True, 'message': 'Audio processor restarted successfully'})
                    except Exception as e:
                        return jsonify({'success': False, 'message': f'Failed to restart audio processor: {str(e)}'})
            
            elif repair_type == 'system':
                # Attempt to restart all components
                try:
                    if self.camera:
                        self.camera.release()
                    if self.emotion_detector and hasattr(self.emotion_detector, 'audio_processor'):
                        self.emotion_detector.audio_processor.stop()
                    
                    time.sleep(2)
                    
                    if self.camera:
                        self.camera.start()
                    if self.emotion_detector and hasattr(self.emotion_detector, 'audio_processor'):
                        self.emotion_detector.audio_processor.start()
                    
                    return jsonify({'success': True, 'message': 'System components restarted successfully'})
                except Exception as e:
                    return jsonify({'success': False, 'message': f'Failed to restart system components: {str(e)}'})
            
            return jsonify({'success': False, 'message': 'Invalid repair type'})
    
    def _setup_socketio(self):
        """Setup Socket.IO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            # Send initial metrics and system info
            self.socketio.emit('metrics_update', self.metrics)
            
            uptime = time.time() - self.start_time
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            self.socketio.emit('system_info', {
                'uptime': f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
                'mode': self.mode
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        # Developer commands - only processed in dev mode
        @self.socketio.on('dev_command')
        def handle_dev_command(data):
            if self.mode != "dev":
                self.socketio.emit('command_response', {
                    'success': False,
                    'message': 'Developer commands are only available in developer mode'
                })
                return
            
            command = data.get('command')
            params = data.get('params', {})
            
            if command == 'clear_metrics':
                # Reset metrics history
                self.metrics['history'] = {
                    'fps': [],
                    'detections': [],
                    'cpu_usage': [],
                    'memory_usage': [],
                    'emotions': {
                        'crying': 0,
                        'laughing': 0,
                        'babbling': 0,
                        'silence': 0,
                        'unknown': 0
                    }
                }
                self.metrics['detection_types'] = {
                    'face': 0,
                    'upper_body': 0,
                    'full_body': 0
                }
                self.metrics['total_detections'] = 0
                self.metrics['detection_log'] = []
                
                self.socketio.emit('command_response', {
                    'success': True,
                    'message': 'Metrics history cleared'
                })
            
            elif command == 'simulate_detection':
                # Simulate a detection event
                count = params.get('count', 1)
                detection_type = params.get('type', 'face')
                
                self.update_detection({
                    'count': count,
                    'type': detection_type,
                    'confidence': 0.95,
                    'fps': self.metrics['current']['fps']
                })
                
                self.socketio.emit('command_response', {
                    'success': True,
                    'message': f'Simulated {count} {detection_type} detection(s)'
                })
            
            elif command == 'simulate_emotion':
                # Simulate an emotion event
                emotion = params.get('emotion', 'crying')
                confidence = params.get('confidence', 0.8)
                
                self.update_emotion({
                    'emotion': emotion,
                    'confidence': confidence
                })
                
                self.socketio.emit('command_response', {
                    'success': True,
                    'message': f'Simulated {emotion} emotion with {confidence:.2f} confidence'
                })
            
            else:
                self.socketio.emit('command_response', {
                    'success': False,
                    'message': f'Unknown command: {command}'
                })
    
    def _generate_frames(self):
        """Generate frames for video streaming"""
        while True:
            with self.frame_lock:
                if self.frame_buffer is None:
                    # If no frame is available, yield a blank frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n'
                           b'\r\n')
                    time.sleep(0.1)
                    continue
                
                # Yield the frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.frame_buffer + b'\r\n')
    
    def update_frame(self, frame_bytes):
        """Update the frame buffer with a new frame"""
        with self.frame_lock:
            self.frame_buffer = frame_bytes
    
    def update_metrics(self, metrics_data):
        """Update metrics with new data"""
        # Update current metrics
        if 'fps' in metrics_data:
            self.metrics['current']['fps'] = metrics_data['fps']
            self.metrics['history']['fps'].append(metrics_data['fps'])
            # Keep only the last 100 values
            if len(self.metrics['history']['fps']) > 100:
                self.metrics['history']['fps'].pop(0)
        
        if 'cpu_usage' in metrics_data:
            self.metrics['current']['cpu_usage'] = metrics_data['cpu_usage']
            self.metrics['history']['cpu_usage'].append(metrics_data['cpu_usage'])
            # Keep only the last 100 values
            if len(self.metrics['history']['cpu_usage']) > 100:
                self.metrics['history']['cpu_usage'].pop(0)
        
        if 'memory_usage' in metrics_data:
            self.metrics['current']['memory_usage'] = metrics_data['memory_usage']
            self.metrics['history']['memory_usage'].append(metrics_data['memory_usage'])
            # Keep only the last 100 values
            if len(self.metrics['history']['memory_usage']) > 100:
                self.metrics['history']['memory_usage'].pop(0)
        
        # Emit updated metrics
        self.socketio.emit('metrics_update', self.metrics)
    
    def update_detection(self, detection_data):
        """Update detection metrics with new data"""
        # Update current detection count
        if 'count' in detection_data:
            self.metrics['current']['detections'] = detection_data['count']
            self.metrics['history']['detections'].append(detection_data['count'])
            # Keep only the last 100 values
            if len(self.metrics['history']['detections']) > 100:
                self.metrics['history']['detections'].pop(0)
        
        # Update detection types
        if 'type' in detection_data:
            detection_type = detection_data['type']
            if detection_type in self.metrics['detection_types']:
                self.metrics['detection_types'][detection_type] += 1
        
        # Update total detections
        if 'count' in detection_data and detection_data['count'] > 0:
            self.metrics['total_detections'] += 1
        
        # Add to detection log
        if 'count' in detection_data and 'type' in detection_data and 'confidence' in detection_data:
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.metrics['detection_log'].insert(0, {
                'time': timestamp,
                'type': detection_data['type'],
                'count': detection_data['count'],
                'confidence': detection_data['confidence']
            })
            # Keep only the last 20 log entries
            if len(self.metrics['detection_log']) > 20:
                self.metrics['detection_log'].pop()
        
        # Update FPS if provided
        if 'fps' in detection_data:
            self.metrics['current']['fps'] = detection_data['fps']
        
        # Emit detection update
        self.socketio.emit('detection_update', {
            'count': detection_data.get('count', 0),
            'type': detection_data.get('type', 'unknown'),
            'confidence': detection_data.get('confidence', 0.0),
            'fps': detection_data.get('fps', 0.0)
        })
    
    def update_emotion(self, emotion_data):
        """Update emotion metrics with new data"""
        # Update current emotion
        if 'emotion' in emotion_data and 'confidence' in emotion_data:
            self.metrics['current']['emotion'] = emotion_data['emotion']
            self.metrics['current']['emotion_confidence'] = emotion_data['confidence']
            
            # Update emotion history
            if emotion_data['emotion'] in self.metrics['history']['emotions']:
                self.metrics['history']['emotions'][emotion_data['emotion']] += 1
        
        # Emit emotion update
        self.socketio.emit('emotion_update', {
            'emotion': emotion_data.get('emotion', 'unknown'),
            'confidence': emotion_data.get('confidence', 0.0)
        })
        
        # Add alert for crying with high confidence
        if emotion_data.get('emotion') == 'crying' and emotion_data.get('confidence', 0) > 0.7:
            self.add_alert('Baby is crying!', 'danger')
    
    def add_alert(self, message, alert_type='info'):
        """Add an alert message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Emit alert
        self.socketio.emit('alert', {
            'time': timestamp,
            'message': message,
            'type': alert_type
        })
    
    def _emit_metrics(self):
        """Periodically emit metrics updates"""
        while True:
            if self.running:
                # Update CPU and memory usage
                self.metrics['current']['cpu_usage'] = psutil.cpu_percent()
                self.metrics['current']['memory_usage'] = psutil.virtual_memory().percent
                
                # Emit metrics update
                self.socketio.emit('metrics_update', self.metrics)
            
            # Sleep for 2 seconds
            time.sleep(2)
    
    def _emit_system_info(self):
        """Periodically emit system info updates"""
        while True:
            if self.running:
                # Calculate uptime
                uptime = time.time() - self.start_time
                hours, remainder = divmod(uptime, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                # Emit system info
                self.socketio.emit('system_info', {
                    'uptime': f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
                    'mode': self.mode
                })
            
            # Sleep for 1 second
            time.sleep(1)
    
    def start(self):
        """Start the web server in a background thread"""
        self.running = True
        logger.info(f"Starting Baby Monitor Web Server in {self.mode.upper()} mode on http://{self.host}:{self.port}")
        # Start in a background thread
        thread = threading.Thread(target=self.run)
        thread.daemon = True
        thread.start()
    
    def run(self):
        """Run the web server in the current thread"""
        self.running = True
        logger.info(f"Running Baby Monitor Web Server in {self.mode.upper()} mode on http://{self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug, use_reloader=False)
    
    def stop(self):
        """Stop the web server"""
        self.running = False
        logger.info("Stopping Baby Monitor Web Server") 