"""
Baby Monitor Web Application
=========================
Web interface for the Baby Monitor System using Flask and Flask-SocketIO.
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import cv2
import numpy as np
import threading
import logging
import time
import base64
import os
import signal
from datetime import datetime


class WebApp:
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins='*')
        self.host = host
        self.port = port
        self.monitor_system = None
        self.logger = logging.getLogger(__name__)
        
        # Setup routes
        self.setup_routes()
        self.setup_socketio_handlers()
    
    def set_monitor_system(self, monitor_system):
        """Set reference to the monitor system"""
        self.monitor_system = monitor_system
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/status')
        def status():
            if not self.monitor_system:
                return jsonify({'error': 'Monitor system not initialized'})
            
            return jsonify({
                'camera_enabled': self.monitor_system.camera_enabled,
                'audio_enabled': self.monitor_system.audio_enabled
            })
        
        @self.app.route('/control/<action>', methods=['POST'])
        def control(action):
            if not self.monitor_system:
                return jsonify({'error': 'Monitor system not initialized'})
            
            try:
                if action == 'toggle_camera':
                    self.monitor_system.toggle_camera()
                    return jsonify({
                        'status': 'success',
                        'camera_enabled': self.monitor_system.camera_enabled
                    })
                elif action == 'toggle_audio':
                    self.monitor_system.toggle_audio()
                    return jsonify({
                        'status': 'success',
                        'audio_enabled': self.monitor_system.audio_enabled
                    })
                else:
                    return jsonify({'error': f'Unknown action: {action}'})
            except Exception as e:
                self.logger.error(f"Error in control action {action}: {str(e)}")
                return jsonify({'error': str(e)})
    
    def setup_socketio_handlers(self):
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("Client connected")
            if self.monitor_system:
                self.emit_status()
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Client disconnected")
    
    def start(self):
        """Start the web server"""
        try:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        except Exception as e:
            self.logger.error(f"Failed to start web server: {str(e)}")
            # Try alternative port if default is in use
            if 'Address already in use' in str(e):
                try:
                    self.port += 1
                    self.logger.info(f"Retrying with port {self.port}")
                    self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
                except Exception as e2:
                    self.logger.error(f"Failed to start web server on alternative port: {str(e2)}")
                    raise
            else:
                raise
    
    def emit_frame(self, frame):
        """Emit a video frame to connected clients"""
        try:
            if frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                # Convert to base64
                frame_data = base64.b64encode(buffer).decode('utf-8')
                self.socketio.emit('frame', {'data': frame_data})
        except Exception as e:
            self.logger.error(f"Error emitting frame: {str(e)}")
    
    def emit_audio(self, audio_data):
        """Emit audio data to connected clients"""
        try:
            if audio_data is not None:
                self.socketio.emit('audio', {'data': audio_data.tolist()})
        except Exception as e:
            self.logger.error(f"Error emitting audio: {str(e)}")
    
    def emit_status(self):
        """Emit current system status"""
        if not self.monitor_system:
            return
        
        try:
            status = {
                'type': 'status',
                'data': {
                    'camera_enabled': self.monitor_system.camera_enabled,
                    'audio_enabled': self.monitor_system.audio_enabled
                }
            }
            self.socketio.emit('status', status)
        except Exception as e:
            self.logger.error(f"Error emitting status: {str(e)}")
    
    def emit_alert(self, level, message):
        """Emit an alert message to connected clients"""
        try:
            self.socketio.emit('alert', {'level': level, 'message': message})
        except Exception as e:
            self.logger.error(f"Error emitting alert: {str(e)}")


class BabyMonitorWeb:
    def __init__(self, host='0.0.0.0', port=5000, dev_mode=False):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        self.monitor_system = None
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30.0  # Target 30 FPS
        self.dev_mode = dev_mode
        self.setup_routes()
        self.setup_socketio()

    def set_monitor_system(self, monitor_system):
        """Set reference to the monitor system."""
        self.monitor_system = monitor_system

    def setup_routes(self):
        """Setup Flask routes."""
        @self.app.route('/')
        def index():
            return render_template('index.html', dev_mode=self.dev_mode)

        @self.app.route('/status')
        def status():
            if not self.monitor_system:
                return jsonify({'error': 'Monitor system not initialized'})
            return jsonify({
                'camera_enabled': self.monitor_system.camera_enabled,
                'audio_enabled': self.monitor_system.audio_enabled,
                'emotion_enabled': True if hasattr(self.monitor_system, 'emotion_recognizer') else False,
                'detection_enabled': True if hasattr(self.monitor_system, 'person_detector') else False,
                'dev_mode': self.dev_mode
            })

        @self.app.route('/control/<action>', methods=['POST'])
        def control(action):
            if not self.monitor_system:
                return jsonify({'error': 'Monitor system not initialized'})
            
            try:
                if action == 'toggle_camera':
                    self.monitor_system.toggle_camera()
                    return jsonify({
                        'status': 'success',
                        'camera_enabled': self.monitor_system.camera_enabled
                    })
                elif action == 'toggle_audio':
                    self.monitor_system.toggle_audio()
                    return jsonify({
                        'status': 'success',
                        'audio_enabled': self.monitor_system.audio_enabled
                    })
                else:
                    return jsonify({'error': f'Unknown action: {action}'})
            except Exception as e:
                self.logger.error(f"Error in control action {action}: {str(e)}")
                return jsonify({'error': str(e)})

    def setup_socketio(self):
        """Setup Socket.IO event handlers."""
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("Client connected")
            if self.monitor_system:
                self.emit_status()

        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Client disconnected")

    def emit_frame(self, frame):
        """Emit video frame to connected clients."""
        try:
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_interval:
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                # Convert to base64 string
                frame_data = base64.b64encode(buffer).decode('utf-8')
                self.socketio.emit('frame', {'data': frame_data})
                self.last_frame_time = current_time
        except Exception as e:
            self.logger.error(f"Error emitting frame: {str(e)}")

    def emit_audio(self, audio_data):
        """Emit audio data to connected clients."""
        try:
            if audio_data is not None:
                self.socketio.emit('audio', {'data': audio_data.tolist()})
        except Exception as e:
            self.logger.error(f"Error emitting audio: {str(e)}")

    def emit_status(self, status_data=None):
        """Emit status update to connected clients."""
        try:
            if status_data is None and self.monitor_system:
                status_data = {
                    'camera_enabled': self.monitor_system.camera_enabled,
                    'audio_enabled': self.monitor_system.audio_enabled,
                    'emotion_enabled': True if hasattr(self.monitor_system, 'emotion_recognizer') else False,
                    'detection_enabled': True if hasattr(self.monitor_system, 'person_detector') else False
                }
            self.socketio.emit('status', status_data)
        except Exception as e:
            self.logger.error(f"Error emitting status: {str(e)}")

    def emit_alert(self, level, message):
        """Emit alert to connected clients."""
        try:
            self.socketio.emit('alert', {
                'level': level,
                'message': message,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        except Exception as e:
            self.logger.error(f"Error emitting alert: {str(e)}")

    def emit_emotion(self, emotion, confidence):
        """Emit emotion detection results to connected clients."""
        try:
            self.socketio.emit('emotion', {
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        except Exception as e:
            self.logger.error(f"Error emitting emotion: {str(e)}")

    def emit_audio_data(self, audio_data):
        """Emit audio waveform data to connected clients."""
        try:
            if audio_data is not None:
                # Convert to list and normalize
                data = audio_data.tolist()
                # Only send a subset of points to reduce bandwidth
                step = max(1, len(data) // 100)  # Limit to ~100 points
                data = data[::step]
                self.socketio.emit('waveform', {'data': data})
        except Exception as e:
            self.logger.error(f"Error emitting audio data: {str(e)}")

    def start(self):
        """Start the web interface."""
        try:
            threading.Thread(target=self._run_server, daemon=True).start()
            self.logger.info(f"Baby Monitor web system started on http://{self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Error starting web interface: {str(e)}")
            raise

    def _run_server(self):
        """Run the Flask server."""
        try:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        except Exception as e:
            self.logger.error(f"Error running web server: {str(e)}")

    def stop(self):
        """Stop the web interface."""
        try:
            # Implement cleanup if needed
            pass
        except Exception as e:
            self.logger.error(f"Error stopping web interface: {str(e)}")
