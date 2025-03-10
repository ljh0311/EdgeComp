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
from scipy import signal as scipy_signal
from datetime import datetime
from queue import Queue, Empty
from threading import Lock


class BabyMonitorWeb:
    def __init__(self, host='0.0.0.0', port=5000, dev_mode=False):
        self.host = host
        self.port = port
        self.dev_mode = dev_mode
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        self.logger = logging.getLogger(__name__)
        self.monitor_system = None
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30.0  # 30 FPS max
        
        # Add thread safety
        self.frame_lock = Lock()
        self.audio_lock = Lock()
        self.thread_lock = Lock()  # For thread management
        
        # Add frame and audio queues with size limits
        self.frame_queue = Queue(maxsize=10)  # Increased but limited buffer
        self.audio_queue = Queue(maxsize=10)
        
        # Add client tracking
        self.connected_clients = set()
        self.client_lock = Lock()
        
        # Thread management
        self.should_run = False
        self.frame_thread = None
        self.audio_thread = None
        self.server_thread = None
        
        self.setup_routes()
        self.setup_socketio()

    def set_monitor_system(self, monitor_system):
        """Set reference to the monitor system."""
        self.monitor_system = monitor_system
        self.start_background_tasks()

    def start_background_tasks(self):
        """Start background tasks for frame and audio processing."""
        with self.thread_lock:
            if not self.frame_thread or not self.frame_thread.is_alive():
                self.frame_thread = threading.Thread(target=self._process_frames, daemon=True)
                self.frame_thread.start()
            
            if not self.audio_thread or not self.audio_thread.is_alive():
                self.audio_thread = threading.Thread(target=self._process_audio, daemon=True)
                self.audio_thread.start()

    def _process_frames(self):
        """Background task to process and emit frames."""
        while self.should_run:
            try:
                if not self.frame_queue.empty():
                    with self.frame_lock:
                        frame = self.frame_queue.get_nowait()
                        current_time = time.time()
                        if current_time - self.last_frame_time >= self.frame_interval:
                            if len(self.connected_clients) > 0:
                                try:
                                    # Resize frame if needed
                                    if frame.shape[1] > 1280:  # If width > 1280
                                        scale = 1280 / frame.shape[1]
                                        frame = cv2.resize(frame, None, fx=scale, fy=scale)
                                    
                                    # Convert frame to JPEG with quality control
                                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                    frame_data = base64.b64encode(buffer).decode('utf-8')
                                    
                                    # Emit with error handling
                                    try:
                                        self.socketio.emit('frame', {'data': frame_data})
                                        self.last_frame_time = current_time
                                        self.logger.debug("Frame processed and emitted")  # Debug log
                                    except Exception as e:
                                        self.logger.error(f"Socket.IO emission error: {str(e)}")
                                except Exception as e:
                                    self.logger.error(f"Frame processing error: {str(e)}")
                else:
                    time.sleep(0.01)  # Short sleep when queue is empty
            except Empty:
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Frame thread error: {str(e)}")
                time.sleep(0.1)

    def _process_audio(self):
        """Background task to process and emit audio data."""
        while self.should_run:
            try:
                if not self.audio_queue.empty():
                    with self.audio_lock:
                        audio_data = self.audio_queue.get_nowait()
                        if len(self.connected_clients) > 0:
                            try:
                                # Convert to float and normalize
                                data = audio_data.astype(float)
                                if np.abs(data).max() > 0:
                                    data = data / np.abs(data).max()
                                
                                # Apply bandpass filter
                                nyquist = 22050
                                low = 20.0 / nyquist
                                high = 4000.0 / nyquist
                                b, a = scipy_signal.butter(4, [low, high], btype='band')
                                data = scipy_signal.filtfilt(b, a, data)
                                
                                # Downsample for visualization
                                target_points = 100
                                if len(data) > target_points:
                                    data = scipy_signal.resample(data, target_points)
                                
                                try:
                                    self.socketio.emit('waveform', {'data': data.tolist()})
                                except Exception as e:
                                    self.logger.error(f"Socket.IO emission error: {str(e)}")
                            except Exception as e:
                                self.logger.error(f"Audio processing error: {str(e)}")
            except Empty:
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Audio thread error: {str(e)}")
                time.sleep(0.1)

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
                        'resolutions': ['640x480', '1280x720', '1920x1080']
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
                    success = self.monitor_system.camera.set_resolution(width, height)
                    if success:
                        return jsonify({
                            'status': 'success',
                            'message': f'Set resolution to {width}x{height}'
                        })
                    else:
                        return jsonify({
                            'status': 'error',
                            'message': 'Failed to set resolution'
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
            with self.client_lock:
                self.connected_clients.add(request.sid)
            self.logger.info(f"Client connected: {request.sid}")
            if self.monitor_system:
                self.emit_status()

        @self.socketio.on('disconnect')
        def handle_disconnect():
            with self.client_lock:
                self.connected_clients.discard(request.sid)
            self.logger.info(f"Client disconnected: {request.sid}")

        @self.socketio.on('get_cameras')
        def handle_get_cameras(data=None):
            try:
                if not self.monitor_system:
                    return {'success': False, 'error': 'Monitor system not initialized'}
                
                cameras = []
                camera_list = self.monitor_system.camera.get_camera_list()
                
                for camera_name in camera_list:
                    resolutions = self.monitor_system.camera.get_camera_resolutions(camera_name)
                    # Format resolutions as strings
                    formatted_resolutions = []
                    for res in resolutions:
                        if isinstance(res, str) and 'x' in res:
                            formatted_resolutions.append(res)
                        elif isinstance(res, (list, tuple)) and len(res) == 2:
                            formatted_resolutions.append(f"{res[0]}x{res[1]}")
                    
                    cameras.append({
                        'id': camera_name,  # Use camera name as ID
                        'name': f"Camera {camera_name}" if camera_name.isdigit() else camera_name,
                        'resolutions': formatted_resolutions
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
                
                # Stop current camera if it's running
                if self.monitor_system.camera_enabled:
                    self.monitor_system.toggle_camera()
                
                success = self.monitor_system.camera.select_camera(camera_name)
                if success:
                    # Clear frame queue
                    with self.frame_lock:
                        while not self.frame_queue.empty():
                            self.frame_queue.get_nowait()
                    
                    # Restart camera if it was enabled
                    if self.monitor_system.camera_enabled:
                        self.monitor_system.toggle_camera()
                    
                    # Get current resolution
                    current_res = self.monitor_system.camera.get_current_resolution()
                    return {
                        'success': True,
                        'message': f'Camera {camera_name} selected',
                        'current_resolution': current_res
                    }
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
                
                resolution = data.get('resolution')
                if not resolution:
                    return {'success': False, 'error': 'No resolution provided'}
                
                try:
                    # Parse resolution string (e.g., "640x480")
                    width, height = map(int, resolution.split('x'))
                    success = self.monitor_system.camera.set_resolution(f"{width}x{height}")
                    
                    if success:
                        # Reset frame processing
                        with self.frame_lock:
                            while not self.frame_queue.empty():
                                self.frame_queue.get_nowait()
                        current_res = self.monitor_system.camera.get_current_resolution()
                        return {'success': True, 'message': f'Resolution set to {current_res}', 'current_resolution': current_res}
                    else:
                        current_res = self.monitor_system.camera.get_current_resolution()
                        return {'success': False, 'error': 'Failed to set resolution', 'current_resolution': current_res}
                except (ValueError, AttributeError) as e:
                    return {'success': False, 'error': f'Invalid resolution format: {str(e)}'}
            except Exception as e:
                self.logger.error(f"Error setting resolution: {str(e)}")
                return {'success': False, 'error': str(e)}

        @self.socketio.on('toggle_camera')
        def handle_toggle_camera(data=None):
            """Handle camera toggle event."""
            if self.monitor_system:
                try:
                    self.monitor_system.toggle_camera()
                    self.emit_status()
                except Exception as e:
                    self.logger.error(f"Error toggling camera: {str(e)}")
                    self.emit_alert("error", f"Failed to toggle camera: {str(e)}")

        @self.socketio.on('toggle_audio')
        def handle_toggle_audio(data=None):
            """Handle audio toggle event."""
            if self.monitor_system:
                try:
                    self.monitor_system.toggle_audio()
                    self.emit_status()
                except Exception as e:
                    self.logger.error(f"Error toggling audio: {str(e)}")
                    self.emit_alert("error", f"Failed to toggle audio: {str(e)}")

    def emit_frame(self, frame):
        """Queue frame for emission to connected clients."""
        try:
            if frame is None:
                self.logger.debug("No frame to emit")
                if self.monitor_system and self.monitor_system.camera_enabled:
                    self.emit_alert('warning', 'Camera is enabled but no frames are being received', True)
                return
                
            if len(self.connected_clients) == 0:
                self.logger.debug("No connected clients to receive frame")
                return

            # Convert frame to JPEG with quality control
            try:
                # Resize frame if needed
                if frame.shape[1] > 1280:  # If width > 1280
                    scale = 1280 / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                    self.logger.debug(f"Resized frame to {frame.shape}")
                
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Emit directly to all connected clients
                self.socketio.emit('frame', {'data': frame_data})
                self.logger.debug(f"Frame emitted successfully to {len(self.connected_clients)} clients")
            except Exception as e:
                self.logger.error(f"Error processing frame: {str(e)}")
                self.emit_alert('warning', 'Error processing camera feed', True)
        except Exception as e:
            self.logger.error(f"Error in emit_frame: {str(e)}")
            self.emit_alert('warning', 'Error with camera feed', True)

    def emit_audio_data(self, audio_data):
        """Queue audio data for emission to connected clients."""
        try:
            if audio_data is None and self.monitor_system and self.monitor_system.audio_enabled:
                self.emit_alert('warning', 'Audio is enabled but no audio data is being received', True)
                return

            if len(self.connected_clients) > 0:
                # Clear old audio data if queue is full
                if self.audio_queue.full():
                    try:
                        self.audio_queue.get_nowait()
                    except Empty:
                        pass
                self.audio_queue.put_nowait(audio_data)
        except Exception as e:
            self.logger.error(f"Error queueing audio data: {str(e)}")
            self.emit_alert('warning', 'Error with audio feed', True)

    def emit_detection(self, detection_data):
        """Emit detection results to connected clients."""
        try:
            if len(self.connected_clients) > 0:
                data = {
                    'people_count': detection_data.get('people_count', 0),
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'detections': []
                }
                
                # Add individual detection information
                if 'detections' in detection_data:
                    for det in detection_data['detections']:
                        data['detections'].append({
                            'id': det.get('id', 1),  # Default to 1 if single person
                            'position': det.get('position', 'unknown'),
                            'confidence': det.get('confidence', 0.0),
                            'box': det.get('box', [])  # [x1, y1, x2, y2]
                        })
                
                # Update motion status and emit alerts for rapid motion
                if detection_data.get('rapid_motion'):
                    data['motion_status'] = 'Rapid Motion'
                    self.emit_alert('warning', 'Rapid motion detected!', True)
                elif detection_data.get('people_count', 0) > 0:
                    data['motion_status'] = 'Motion Detected'
                else:
                    data['motion_status'] = 'No Motion'
                
                # Add fall detection status
                if detection_data.get('fall_detected'):
                    data['fall_detected'] = True
                    self.emit_alert('critical', 'Fall detected! Please check immediately.', True)
                
                self.socketio.emit('detection', data)
        except Exception as e:
            self.logger.error(f"Error emitting detection: {str(e)}")

    def emit_emotion(self, emotion, confidence):
        """Emit emotion detection results to connected clients."""
        try:
            if len(self.connected_clients) > 0:
                self.socketio.emit('emotion', {
                    'emotion': emotion,
                    'confidence': confidence
                })
        except Exception as e:
            self.logger.error(f"Error emitting emotion: {str(e)}")

    def emit_alert(self, level, message, should_beep=False):
        """Emit alert to connected clients."""
        try:
            if len(self.connected_clients) > 0:
                self.socketio.emit('alert', {
                    'level': level,
                    'message': message,
                    'should_beep': should_beep,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
        except Exception as e:
            self.logger.error(f"Error emitting alert: {str(e)}")

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
            
            # Check for enabled features without data
            if status_data:
                if status_data.get('camera_enabled') and not hasattr(self.monitor_system, 'camera'):
                    self.emit_alert('warning', 'Camera is enabled but not properly initialized', True)
                if status_data.get('audio_enabled') and not hasattr(self.monitor_system, 'audio_processor'):
                    self.emit_alert('warning', 'Audio is enabled but not properly initialized', True)
            
            if len(self.connected_clients) > 0:
                self.socketio.emit('status', status_data)
        except Exception as e:
            self.logger.error(f"Error emitting status: {str(e)}")

    def start(self):
        """Start the web interface."""
        try:
            self.should_run = True  # Set this before starting threads
            self.start_background_tasks()  # Start processing threads first
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            self.logger.info(f"Baby Monitor web system started on http://{self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Error starting web interface: {str(e)}")
            self.should_run = False
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
            self.should_run = False
            
            # Stop background threads
            with self.thread_lock:
                if self.frame_thread and self.frame_thread.is_alive():
                    self.frame_thread.join(timeout=2.0)
                if self.audio_thread and self.audio_thread.is_alive():
                    self.audio_thread.join(timeout=2.0)
            
            # Clear queues
            with self.frame_lock:
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        break
            
            with self.audio_lock:
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except Empty:
                        break
            
            # Clear client set
            with self.client_lock:
                self.connected_clients.clear()
            
            self.logger.info("Web interface stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping web interface: {str(e)}")
            raise
