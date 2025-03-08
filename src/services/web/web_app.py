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
from pathlib import Path


class BabyMonitorWeb:
    def __init__(self, host='0.0.0.0', port=5000, dev_mode=False, monitor_system=None):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        self.monitor_system = monitor_system
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30.0  # Target 30 FPS
        self.dev_mode = dev_mode
        self.model_switch_lock = threading.Lock()
        self.last_emotion_time = 0
        self.emotion_interval = 0.1  # Update emotion every 100ms
        self.setup_routes()
        self.setup_socketio()

    def setup_routes(self):
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
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("Client connected")
            if self.monitor_system:
                self.emit_status()
                self.emit_system_info()
                # Send current model info
                if hasattr(self.monitor_system, 'current_emotion_detector'):
                    self.emit_model_info()

        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Client disconnected")

        @self.socketio.on('get_system_info')
        def handle_get_system_info():
            """Handle request for system information."""
            try:
                if not self.monitor_system:
                    return {'success': False, 'error': 'Monitor system not initialized'}
                
                system_info = {
                    'platform': self.monitor_system.system_info,
                    'resources': self.monitor_system.get_resource_usage(),
                    'recommended_model': self.monitor_system.recommend_model()
                }
                return {'success': True, 'system_info': system_info}
            except Exception as e:
                self.logger.error(f"Error getting system info: {str(e)}")
                return {'success': False, 'error': str(e)}

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

        @self.socketio.on('select_camera')
        def handle_camera_selection(self, data):
            """Handle camera selection from client."""
            try:
                camera_id = data.get('camera_id')
                if camera_id is not None:
                    self.monitor_system.set_camera(int(camera_id))
                    self.emit_status()
            except Exception as e:
                self.logger.error(f"Error handling camera selection: {str(e)}")
                self.emit_alert('error', 'Failed to switch camera')

        @self.socketio.on('select_resolution')
        def handle_resolution_selection(self, data):
            """Handle resolution selection from client."""
            try:
                width = data.get('width')
                height = data.get('height')
                if width is not None and height is not None:
                    self.monitor_system.set_resolution(int(width), int(height))
                    self.emit_status()
            except Exception as e:
                self.logger.error(f"Error handling resolution selection: {str(e)}")
                self.emit_alert('error', 'Failed to change resolution')

        @self.socketio.on('select_model')
        def handle_model_selection(data):
            """Handle emotion model selection."""
            try:
                with self.model_switch_lock:
                    model_key = data.get('model_key')
                    if not model_key:
                        return {'success': False, 'error': 'No model key provided'}

                    # Emit loading state
                    self.socketio.emit('model_loading', {
                        'status': 'loading',
                        'message': f'Switching to {model_key} model...'
                    })

                    # Stop audio processing temporarily
                    was_audio_enabled = False
                    if hasattr(self.monitor_system, 'audio_enabled') and self.monitor_system.audio_enabled:
                        was_audio_enabled = True
                        self.monitor_system.toggle_audio()

                    try:
                        # Initialize new model
                        self.monitor_system.initialize_emotion_detector(model_key)
                        
                        # Restart audio if it was enabled
                        if was_audio_enabled:
                            self.monitor_system.toggle_audio()

                        # Emit success state
                        self.socketio.emit('model_loading', {
                            'status': 'success',
                            'message': f'Successfully switched to {model_key} model'
                        })
                        
                        # Update model info
                        self.emit_model_info()
                        
                        return {'success': True}
                    except Exception as e:
                        error_msg = f"Failed to initialize {model_key} model: {str(e)}"
                        self.logger.error(error_msg)
                        
                        # Try to restore audio
                        if was_audio_enabled:
                            self.monitor_system.toggle_audio()
                            
                        # Emit error state
                        self.socketio.emit('model_loading', {
                            'status': 'error',
                            'message': error_msg
                        })
                        return {'success': False, 'error': error_msg}

            except Exception as e:
                error_msg = f"Error handling model selection: {str(e)}"
                self.logger.error(error_msg)
                self.socketio.emit('model_loading', {
                    'status': 'error',
                    'message': error_msg
                })
                return {'success': False, 'error': error_msg}

        @self.socketio.on('get_activity_summary')
        def handle_get_activity_summary():
            """Handle request for activity summary."""
            try:
                self.emit_activity_summary()
            except Exception as e:
                self.logger.error(f"Error handling activity summary request: {str(e)}")
                return {'success': False, 'error': str(e)}

    def emit_frame(self, frame):
        """Emit video frame to connected clients."""
        try:
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_interval:
                # Resize frame to reduce bandwidth if too large
                height, width = frame.shape[:2]
                if width > 800:  # Reduced max width threshold
                    scale = 800 / width
                    new_width = 800
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

                # Convert frame to JPEG with optimized quality
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                # Convert to base64 string
                frame_data = base64.b64encode(buffer).decode('utf-8')
                self.socketio.emit('frame', {'data': frame_data})
                self.last_frame_time = current_time
        except Exception as e:
            if self.dev_mode:
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
            if status_data:
                self.socketio.emit('status', status_data)
        except Exception as e:
            self.logger.error(f"Error emitting status: {str(e)}")

    def emit_alert(self, level, message):
        """Emit alert to connected clients."""
        try:
            # Only emit non-critical alerts in dev mode
            if level != 'critical' and not self.dev_mode:
                return

            self.socketio.emit('alert', {
                'level': level,
                'message': message,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        except Exception as e:
            if self.dev_mode:
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

    def emit_camera_list(self):
        """Emit list of available cameras to clients."""
        try:
            cameras = self.monitor_system.get_available_cameras()
            self.socketio.emit('camera_list', {
                'cameras': cameras
            })
        except Exception as e:
            self.logger.error(f"Error emitting camera list: {str(e)}")

    def emit_resolution_list(self):
        """Emit list of supported resolutions to clients."""
        try:
            resolutions = [
                {'width': 640, 'height': 480},
                {'width': 800, 'height': 600},
                {'width': 1280, 'height': 720},
                {'width': 1920, 'height': 1080}
            ]
            self.socketio.emit('resolution_list', {
                'resolutions': resolutions
            })
        except Exception as e:
            self.logger.error(f"Error emitting resolution list: {str(e)}")

    def emit_system_info(self):
        """Emit system information to connected clients."""
        try:
            if self.monitor_system:
                system_info = {
                    'platform': self.monitor_system.system_info,
                    'resources': self.monitor_system.get_resource_usage(),
                    'recommended_model': self.monitor_system.recommend_model()
                }
                self.socketio.emit('system_info', system_info)
        except Exception as e:
            self.logger.error(f"Error emitting system info: {str(e)}")

    def emit_resource_usage(self):
        """Emit resource usage metrics to connected clients."""
        try:
            if self.monitor_system:
                usage = self.monitor_system.get_resource_usage()
                self.socketio.emit('resource_usage', usage)
        except Exception as e:
            self.logger.error(f"Error emitting resource usage: {str(e)}")

    def emit_model_info(self):
        """Emit current model information to clients."""
        try:
            if hasattr(self.monitor_system, 'current_emotion_detector'):
                detector = self.monitor_system.current_emotion_detector
                model_info = {
                    'name': detector.model_name if hasattr(detector, 'model_name') else 'Unknown',
                    'type': detector.__class__.__name__,
                    'supported_emotions': detector.supported_emotions if hasattr(detector, 'supported_emotions') else [],
                    'is_gpu': hasattr(detector, 'device') and 'cuda' in str(detector.device)
                }
                self.socketio.emit('model_info', model_info)
        except Exception as e:
            self.logger.error(f"Error emitting model info: {str(e)}")

    def emit_speech_emotion(self, emotion_data):
        """Emit speech emotion detection results to connected clients."""
        try:
            current_time = time.time()
            if current_time - self.last_emotion_time >= self.emotion_interval:
                if emotion_data and isinstance(emotion_data, dict):
                    self.socketio.emit('speech_emotion', {
                        'emotion': emotion_data.get('emotion', 'neutral'),
                        'confidence': emotion_data.get('confidence', 0.0)
                    })
                    self.last_emotion_time = current_time
        except Exception as e:
            self.logger.error(f"Error emitting speech emotion: {str(e)}")

    def emit_waveform(self, audio_data):
        """Emit waveform data to web clients."""
        try:
            # Convert audio data to list for JSON serialization
            data = audio_data.tolist() if isinstance(audio_data, np.ndarray) else list(audio_data)
            
            # Downsample data if too large (keep max 1000 points)
            if len(data) > 1000:
                step = len(data) // 1000
                data = data[::step][:1000]
            
            # Normalize data to [-1, 1] range
            max_val = max(abs(min(data)), abs(max(data)))
            if max_val > 0:
                data = [x / max_val for x in data]
            
            self.socketio.emit('waveform_data', {
                'data': data,
                'timestamp': time.time()
            })
        except Exception as e:
            self.logger.error(f"Error emitting waveform data: {str(e)}")

    def start(self):
        """Start the web interface."""
        try:
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                allow_unsafe_werkzeug=True,  # Allow development server
                log_output=False  # Disable default logging
            )
            self.logger.info(f"Baby Monitor web system started on http://{self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Error starting web interface: {str(e)}")
            raise

    def stop(self):
        """Stop the web interface."""
        try:
            # Implement cleanup if needed
            pass
        except Exception as e:
            self.logger.error(f"Error stopping web interface: {str(e)}")

    def emit_detection(self, detection_data):
        """Emit detection results and status tracking to connected clients."""
        try:
            if not detection_data:
                return

            # Get current status from person tracker
            person_tracker = self.monitor_system.person_detector
            current_status = person_tracker.get_status_summary()
            activity_timeline = person_tracker.get_activity_timeline()

            # Convert all values to native Python types
            data = {
                'timestamp': str(detection_data.get('timestamp', datetime.now().strftime('%H:%M:%S'))),
                'people_count': int(detection_data.get('people_count', 0)),
                'status': current_status,
                'timeline': [
                    {
                        'timestamp': entry['timestamp'].strftime('%H:%M:%S'),
                        'status': entry['primary_status'],
                        'counts': entry['status_counts']
                    }
                    for entry in activity_timeline[-10:]  # Send last 10 entries
                ],
                'position': str(detection_data.get('position', 'unknown'))
            }
            
            # Add motion status with more detailed information
            primary_status = max(current_status.items(), key=lambda x: x[1])[0] if current_status else 'no_detection'
            
            if primary_status == 'lying':
                data['motion_status'] = 'Baby is lying down'
                data['status_level'] = 'info'
            elif primary_status == 'moving':
                data['motion_status'] = 'Baby is moving'
                data['status_level'] = 'warning'
            elif primary_status == 'seated':
                data['motion_status'] = 'Baby is seated'
                data['status_level'] = 'info'
            else:
                data['motion_status'] = 'No detection'
                data['status_level'] = 'normal'

            # Ensure all values are JSON serializable
            data = {k: str(v) if isinstance(v, (np.bool_, np.integer, np.floating)) else v 
                   for k, v in data.items()}

            # Emit detection data
            self.socketio.emit('detection', data)

            # Emit alerts based on status
            if primary_status == 'moving' and current_status.get('moving', 0) > 0.8:
                # Alert if moving for more than 80% of recent detections
                self.emit_alert('warning', 'High activity detected!')
            elif primary_status == 'lying' and data['people_count'] > 0:
                # Informational alert when baby is lying down
                if self.dev_mode:
                    self.emit_alert('info', 'Baby is resting')
                
        except Exception as e:
            if self.dev_mode:
                self.logger.error(f"Error emitting detection data: {str(e)}")
                self.logger.debug(f"Detection data that caused error: {detection_data}")

    def emit_activity_summary(self):
        """Emit summary of baby's activity."""
        try:
            if not hasattr(self.monitor_system, 'person_detector'):
                return

            person_tracker = self.monitor_system.person_detector
            
            # Get 24-hour summary
            from datetime import timedelta
            day_summary = person_tracker.get_status_summary(timedelta(days=1))
            
            # Format summary data
            summary = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'daily_activity': day_summary,
                'current_status': person_tracker.get_status_summary(timedelta(minutes=5))
            }
            
            self.socketio.emit('activity_summary', summary)
            
        except Exception as e:
            self.logger.error(f"Error emitting activity summary: {str(e)}")
