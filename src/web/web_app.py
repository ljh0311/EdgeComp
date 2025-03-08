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
    def __init__(self, monitor_system):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.monitor_system = monitor_system
        self.logger = logging.getLogger(__name__)
        self.setup_routes()
        self.setup_socketio()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/status')
        def status():
            return jsonify({
                'camera_enabled': self.monitor_system.camera_enabled,
                'audio_enabled': self.monitor_system.audio_enabled,
                'motion_detected': self.monitor_system.motion_detected,
                'emotion_detected': self.monitor_system.emotion_detected
            })

    def setup_socketio(self):
        @self.socketio.on('connect')
        def handle_connect():
            self.emit_status()

        @self.socketio.on('get_cameras')
        def handle_get_cameras(data=None):
            try:
                if not self.monitor_system:
                    return {'success': False, 'error': 'Monitor system not initialized'}
                
                cameras = []
                camera_list = self.monitor_system.camera.get_camera_list()
                
                for camera_name in camera_list:
                    resolutions = self.monitor_system.camera.get_camera_resolutions(camera_name)
                    cameras.append({
                        'id': camera_name,
                        'name': camera_name,
                        'resolutions': [{'width': int(res.split('x')[0]), 'height': int(res.split('x')[1])} 
                                      for res in resolutions]
                    })
                
                return {'success': True, 'cameras': cameras}
            except Exception as e:
                self.logger.error(f"Error getting cameras: {str(e)}")
                return {'success': False, 'error': str(e)}

        @self.socketio.on('select_camera')
        def handle_select_camera(data):
            try:
                if not self.monitor_system:
                    return {'success': False, 'error': 'Monitor system not initialized'}
                
                camera_name = data.get('camera_name')
                if not camera_name:
                    return {'success': False, 'error': 'No camera name provided'}

                if self.monitor_system.camera.select_camera(camera_name):
                    return {'success': True, 'message': f'Selected camera: {camera_name}'}
                else:
                    return {'success': False, 'error': f'Failed to select camera: {camera_name}'}
            except Exception as e:
                self.logger.error(f"Error selecting camera: {str(e)}")
                return {'success': False, 'error': str(e)}

        @self.socketio.on('set_resolution')
        def handle_set_resolution(data):
            try:
                width = int(data.get('width'))
                height = int(data.get('height'))
                success = self.monitor_system.camera.set_resolution(width, height)
                return {'success': success}
            except Exception as e:
                return {'success': False, 'error': str(e)}

        @self.socketio.on('toggle_camera')
        def handle_toggle_camera():
            try:
                self.monitor_system.toggle_camera()
                self.emit_status()
                return {'success': True}
            except Exception as e:
                return {'success': False, 'error': str(e)}

        @self.socketio.on('toggle_audio')
        def handle_toggle_audio():
            try:
                self.monitor_system.toggle_audio()
                self.emit_status()
                return {'success': True}
            except Exception as e:
                return {'success': False, 'error': str(e)}

    def emit_frame(self, frame):
        """Emit a video frame to all connected clients."""
        if frame is not None:
            try:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                # Convert to base64 string
                b64_string = base64.b64encode(buffer).decode('utf-8')
                # Emit the frame
                self.socketio.emit('frame', {'data': b64_string})
            except Exception as e:
                self.logger.error(f"Error emitting frame: {str(e)}")

    def emit_audio_data(self, samples):
        """Emit audio data for visualization."""
        if samples is not None:
            try:
                # Normalize samples to [-1, 1] range
                normalized = samples.astype(float) / np.iinfo(samples.dtype).max
                # Downsample for visualization
                downsampled = signal.resample(normalized, 100)
                self.socketio.emit('audio_data', {'samples': downsampled.tolist()})
            except Exception as e:
                self.logger.error(f"Error emitting audio data: {str(e)}")

    def emit_status(self):
        """Emit current system status to all connected clients."""
        self.socketio.emit('status', {
            'camera_enabled': self.monitor_system.camera_enabled,
            'audio_enabled': self.monitor_system.audio_enabled,
            'motion_detected': self.monitor_system.motion_detected,
            'emotion_detected': self.monitor_system.emotion_detected
        })

    def start(self):
        """Start the web server."""
        host = '0.0.0.0'  # Listen on all network interfaces
        port = 5000
        self.logger.info(f"Starting web server on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=False, use_reloader=False)

    def emit_detection(self, detection_data):
        """Emit detection results to connected clients."""
        try:
            data = {
                'people_count': detection_data.get('people_count', 0),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            
            # Update motion status
            if detection_data.get('rapid_motion'):
                data['motion_status'] = 'Rapid Motion'
            elif detection_data.get('people_count', 0) > 0:
                data['motion_status'] = 'Motion Detected'
            else:
                data['motion_status'] = 'No Motion'
            
            # Add fall detection status
            if detection_data.get('fall_detected'):
                data['fall_detected'] = True
                # Emit a critical alert for fall detection
                self.emit_alert('critical', 'Fall detected!')
            
            self.socketio.emit('detection', data)
        except Exception as e:
            self.logger.error(f"Error emitting detection: {str(e)}")

    def emit_audio_detection(self, sound_type):
        """Emit audio detection results to connected clients."""
        try:
            self.socketio.emit('audio_detection', {
                'sound_type': sound_type,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        except Exception as e:
            self.logger.error(f"Error emitting audio detection: {str(e)}")

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
                data = audio_data.astype(float)
                data = data / np.abs(data).max() if np.abs(data).max() > 0 else data
                # Only send a subset of points to reduce bandwidth
                step = max(1, len(data) // 100)  # Limit to ~100 points
                data = data[::step]
                self.socketio.emit('waveform', {'data': data.tolist()})
        except Exception as e:
            self.logger.error(f"Error emitting audio data: {str(e)}")

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


class BabyMonitorWeb:
    def __init__(self, host='0.0.0.0', port=5000, dev_mode=False):
        self.host = host
        self.port = port
        self.dev_mode = dev_mode
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.logger = logging.getLogger(__name__)
        self.monitor_system = None
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30.0  # 30 FPS max
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

        @self.app.route('/cameras')
        def get_cameras():
            if not self.monitor_system:
                return jsonify({'error': 'Monitor system not initialized'})
            
            try:
                cameras = []
                available_cameras = self.monitor_system.camera.get_available_cameras()
                
                for idx in available_cameras:
                    cameras.append({
                        'id': str(idx),
                        'name': f'Camera {idx}',
                        'resolutions': [
                            {'width': 640, 'height': 480},
                            {'width': 1280, 'height': 720},
                            {'width': 1920, 'height': 1080}
                        ]
                    })
                
                return jsonify(cameras)
            except Exception as e:
                self.logger.error(f"Error getting cameras: {str(e)}")
                return jsonify({'error': str(e)})

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
                elif action == 'select_camera':
                    data = request.get_json()
                    camera_id = int(data.get('camera_id', 0))
                    success = self.monitor_system.camera.select_camera(camera_id)
                    if success:
                        self.monitor_system.camera.initialize()
                        return jsonify({
                            'status': 'success',
                            'message': f'Selected camera {camera_id}'
                        })
                    else:
                        return jsonify({
                            'status': 'error',
                            'message': f'Failed to select camera {camera_id}'
                        })
                elif action == 'set_resolution':
                    data = request.get_json()
                    width = int(data.get('width', 640))
                    height = int(data.get('height', 480))
                    self.monitor_system.camera.width = width
                    self.monitor_system.camera.height = height
                    success = self.monitor_system.camera.initialize()
                    return jsonify({
                        'status': 'success' if success else 'error',
                        'message': f'Set resolution to {width}x{height}'
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

        @self.socketio.on('get_cameras')
        def handle_get_cameras(data=None):
            try:
                if not self.monitor_system:
                    return {'success': False, 'error': 'Monitor system not initialized'}
                
                cameras = []
                camera_list = self.monitor_system.camera.get_camera_list()
                
                for camera_name in camera_list:
                    resolutions = self.monitor_system.camera.get_camera_resolutions(camera_name)
                    cameras.append({
                        'id': camera_name,
                        'name': camera_name,
                        'resolutions': [{'width': int(res.split('x')[0]), 'height': int(res.split('x')[1])} 
                                      for res in resolutions]
                    })
                
                return {'success': True, 'cameras': cameras}
            except Exception as e:
                self.logger.error(f"Error getting cameras: {str(e)}")
                return {'success': False, 'error': str(e)}

        @self.socketio.on('select_camera')
        def handle_select_camera(data):
            try:
                if not self.monitor_system:
                    return {'success': False, 'error': 'Monitor system not initialized'}
                
                camera_name = data.get('camera_name')
                if not camera_name:
                    return {'success': False, 'error': 'No camera name provided'}

                if self.monitor_system.camera.select_camera(camera_name):
                    return {'success': True, 'message': f'Selected camera: {camera_name}'}
                else:
                    return {'success': False, 'error': f'Failed to select camera: {camera_name}'}
            except Exception as e:
                self.logger.error(f"Error selecting camera: {str(e)}")
                return {'success': False, 'error': str(e)}

        @self.socketio.on('set_resolution')
        def handle_set_resolution(data):
            try:
                if not self.monitor_system:
                    return {'success': False, 'error': 'Monitor system not initialized'}
                width = int(data.get('width', 640))
                height = int(data.get('height', 480))
                resolution = f"{width}x{height}"
                success = self.monitor_system.camera.set_resolution(resolution)
                return {
                    'success': success,
                    'message': f'Resolution set to {resolution}' if success else 'Failed to set resolution'
                }
            except Exception as e:
                self.logger.error(f"Error setting resolution: {str(e)}")
                return {'success': False, 'error': str(e)}

        @self.socketio.on('toggle_camera')
        def handle_toggle_camera():
            try:
                if not self.monitor_system:
                    return {'success': False, 'error': 'Monitor system not initialized'}
                self.monitor_system.toggle_camera()
                self.emit_status()
                return {'success': True}
            except Exception as e:
                return {'success': False, 'error': str(e)}

        @self.socketio.on('toggle_audio')
        def handle_toggle_audio():
            try:
                if not self.monitor_system:
                    return {'success': False, 'error': 'Monitor system not initialized'}
                self.monitor_system.toggle_audio()
                self.emit_status()
                return {'success': True}
            except Exception as e:
                return {'success': False, 'error': str(e)}

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
                data = audio_data.astype(float)
                data = data / np.abs(data).max() if np.abs(data).max() > 0 else data
                # Only send a subset of points to reduce bandwidth
                step = max(1, len(data) // 100)  # Limit to ~100 points
                data = data[::step]
                self.socketio.emit('waveform', {'data': data.tolist()})
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

    def emit_detection(self, detection_data):
        """Emit detection results to connected clients."""
        try:
            data = {
                'people_count': detection_data.get('people_count', 0),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            
            # Update motion status
            if detection_data.get('rapid_motion'):
                data['motion_status'] = 'Rapid Motion'
            elif detection_data.get('people_count', 0) > 0:
                data['motion_status'] = 'Motion Detected'
            else:
                data['motion_status'] = 'No Motion'
            
            # Add fall detection status
            if detection_data.get('fall_detected'):
                data['fall_detected'] = True
                # Emit a critical alert for fall detection
                self.emit_alert('critical', 'Fall detected!')
            
            self.socketio.emit('detection', data)
        except Exception as e:
            self.logger.error(f"Error emitting detection: {str(e)}")
