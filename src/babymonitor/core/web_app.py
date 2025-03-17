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
from ..alerts.alert_manager import AlertManager, AlertLevel, AlertType
from .webrtc_streamer import WebRTCStreamer
from aiortc import RTCSessionDescription, RTCIceCandidate
import asyncio


class BabyMonitorWeb:
    def __init__(self, host="0.0.0.0", port=5000, dev_mode=False):
        self.host = host
        self.port = port
        self.dev_mode = dev_mode
        
        # Set up Flask with correct template directory
        template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web', 'templates'))
        static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web', 'static'))
        self.app = Flask(__name__, 
                        template_folder=template_dir,
                        static_folder=static_dir)

        # Update Socket.IO configuration
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode="threading",
            ping_timeout=60,
            ping_interval=25,
            max_http_buffer_size=10e6,
            engineio_logger=dev_mode,
            logger=dev_mode,
        )

        # Add CORS headers
        @self.app.after_request
        def after_request(response):
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add(
                "Access-Control-Allow-Headers", "Content-Type,Authorization"
            )
            response.headers.add(
                "Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS"
            )
            return response

        self.logger = logging.getLogger(__name__)
        self.monitor_system = None
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30.0  # 30 FPS max

        # Add thread safety
        self.frame_lock = Lock()
        self.audio_lock = Lock()
        self.thread_lock = Lock()  # For thread management
        self.metrics_lock = Lock()  # For performance metrics

        # Add frame and audio queues with size limits
        self.frame_queue = Queue(maxsize=10)
        self.audio_queue = Queue(maxsize=10)

        # Performance metrics storage
        self.metrics_history = {
            "fps": [],
            "frame_time": [],
            "cpu_usage": [],
            "memory_usage": [],
        }
        self.metrics_max_history = 60  # Store 1 minute of history

        # Add client tracking
        self.connected_clients = set()
        self.client_lock = Lock()

        # Thread management
        self.should_run = False
        self.frame_thread = None
        self.audio_thread = None
        self.server_thread = None

        # Initialize alert manager
        self.alert_manager = AlertManager()
        self.alert_manager.add_alert_handler(self._handle_alert)
        
        # Initialize WebRTC streamer
        self.webrtc_streamer = WebRTCStreamer()
        
        # Create asyncio event loop for WebRTC
        self.loop = asyncio.new_event_loop()
        self.webrtc_thread = None

        self.setup_routes()
        self.setup_socketio()

    def set_monitor_system(self, monitor_system):
        """Set reference to the monitor system."""
        self.monitor_system = monitor_system
        self.start_background_tasks()

    def setup_routes(self):
        """Setup Flask routes."""
        @self.app.route('/')
        def index():
            return render_template('index.html', dev_mode=self.dev_mode)

        @self.app.route('/metrics')
        def metrics_page():
            """Render the metrics page."""
            return render_template('metrics.html')

        @self.app.route('/api/metrics')
        def get_metrics():
            """Get current performance metrics."""
            with self.metrics_lock:
                return jsonify(self.metrics_history)

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
                    success = self.monitor_system.camera.set_resolution(f"{width}x{height}")
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
        """Set up Socket.IO event handlers."""
        @self.socketio.on('connect')
        def handle_connect():
            client_id = request.sid
            self.logger.info(f"Client connected: {client_id}")
            with self.client_lock:
                self.connected_clients.add(client_id)
            self.emit_status()
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = request.sid
            self.logger.info(f"Client disconnected: {client_id}")
            with self.client_lock:
                if client_id in self.connected_clients:
                    self.connected_clients.remove(client_id)
            
            # Close WebRTC connection
            asyncio.run_coroutine_threadsafe(
                self.webrtc_streamer.close_peer_connection(client_id),
                self.loop
            )
            
        @self.socketio.on('webrtc_request')
        def handle_webrtc_request():
            client_id = request.sid
            self.logger.info(f"WebRTC request from client: {client_id}")
            
            # Create WebRTC offer
            async def create_and_send_offer():
                try:
                    offer = await self.webrtc_streamer.create_offer(client_id)
                    self.socketio.emit('webrtc_offer', {'offer': offer.sdp}, room=client_id)
                except Exception as e:
                    self.logger.error(f"Error creating WebRTC offer: {str(e)}")
                    self.emit_alert('error', f"Failed to create WebRTC connection: {str(e)}")
            
            asyncio.run_coroutine_threadsafe(create_and_send_offer(), self.loop)
            
        @self.socketio.on('webrtc_answer')
        def handle_webrtc_answer(data):
            client_id = request.sid
            self.logger.info(f"WebRTC answer from client: {client_id}")
            
            # Process WebRTC answer
            async def process_answer():
                try:
                    answer = RTCSessionDescription(sdp=data['answer']['sdp'], type=data['answer']['type'])
                    await self.webrtc_streamer.process_answer(client_id, answer)
                except Exception as e:
                    self.logger.error(f"Error processing WebRTC answer: {str(e)}")
                    self.emit_alert('error', f"Failed to establish WebRTC connection: {str(e)}")
            
            asyncio.run_coroutine_threadsafe(process_answer(), self.loop)
            
        @self.socketio.on('ice_candidate')
        def handle_ice_candidate(data):
            client_id = request.sid
            
            # Process ICE candidate
            async def process_ice_candidate():
                try:
                    candidate = RTCIceCandidate(
                        sdpMid=data['candidate'].get('sdpMid'),
                        sdpMLineIndex=data['candidate'].get('sdpMLineIndex'),
                        candidate=data['candidate'].get('candidate')
                    )
                    await self.webrtc_streamer.add_ice_candidate(client_id, candidate)
                except Exception as e:
                    self.logger.error(f"Error processing ICE candidate: {str(e)}")
            
            asyncio.run_coroutine_threadsafe(process_ice_candidate(), self.loop)
            
        @self.socketio.on('request_metrics')
        def handle_request_metrics():
            """Handle client request for current metrics."""
            with self.metrics_lock:
                self.socketio.emit('metrics_update', self.metrics_history)

        @self.socketio.on('toggle_camera')
        def handle_toggle_camera():
            """Handle camera toggle event."""
            if self.monitor_system:
                try:
                    self.monitor_system.toggle_camera()
                    self.emit_status()
                except Exception as e:
                    self.logger.error(f"Error toggling camera: {str(e)}")
                    self.emit_alert("error", f"Failed to toggle camera: {str(e)}")

        @self.socketio.on('toggle_audio')
        def handle_toggle_audio():
            """Handle audio toggle event."""
            if self.monitor_system:
                try:
                    self.monitor_system.toggle_audio()
                    self.emit_status()
                except Exception as e:
                    self.logger.error(f"Error toggling audio: {str(e)}")
                    self.emit_alert("error", f"Failed to toggle audio: {str(e)}")

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
        resized_frame = None
        
        while self.should_run:
            try:
                if not self.frame_queue.empty():
                    with self.frame_lock:
                        frame = self.frame_queue.get_nowait()
                        current_time = time.time()
                        
                        # Add frame to WebRTC streamer
                        self.webrtc_streamer.add_frame(frame)
                        
                        # Also send via Socket.IO as fallback (at a lower rate)
                        if current_time - self.last_frame_time >= self.frame_interval * 3:  # 3x slower for Socket.IO
                            if len(self.connected_clients) > 0:
                                try:
                                    # Resize frame if needed
                                    if frame.shape[1] > 640:  # Reduced from 1280 to 640 for Socket.IO
                                        scale = 640 / frame.shape[1]
                                        new_width = int(frame.shape[1] * scale)
                                        new_height = int(frame.shape[0] * scale)
                                        
                                        if (resized_frame is None or 
                                            resized_frame.shape[0] != new_height or 
                                            resized_frame.shape[1] != new_width):
                                            resized_frame = cv2.resize(frame, (new_width, new_height),
                                                                     interpolation=cv2.INTER_NEAREST)
                                        else:
                                            cv2.resize(frame, (new_width, new_height),
                                                      dst=resized_frame,
                                                      interpolation=cv2.INTER_NEAREST)
                                        frame_to_encode = resized_frame
                                    else:
                                        frame_to_encode = frame
                                    
                                    # Convert frame to JPEG with lower quality for Socket.IO
                                    _, buffer = cv2.imencode('.jpg', frame_to_encode, 
                                                           [cv2.IMWRITE_JPEG_QUALITY, 70])
                                    frame_data = base64.b64encode(buffer).decode('utf-8')
                                    
                                    self.socketio.emit('frame', {'data': frame_data})
                                    self.last_frame_time = current_time
                                except Exception as e:
                                    self.logger.error(f"Frame processing error: {str(e)}")
                else:
                    time.sleep(0.001)
            except Empty:
                time.sleep(0.001)
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
                                
                                self.socketio.emit('waveform', {'data': data.tolist()})
                            except Exception as e:
                                self.logger.error(f"Audio processing error: {str(e)}")
            except Empty:
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Audio thread error: {str(e)}")
                time.sleep(0.1)

    def emit_frame(self, frame):
        """Queue frame for emission to connected clients."""
        try:
            if frame is None:
                self.logger.debug("No frame to emit")
                if self.monitor_system and self.monitor_system.camera_enabled:
                    self.emit_alert('warning', 'Camera is enabled but no frames are being received', True)
                return
                
            if len(self.connected_clients) == 0:
                return

            if self.frame_queue.qsize() < self.frame_queue.maxsize - 1:
                self.frame_queue.put_nowait(frame)
            else:
                try:
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except Empty:
                    pass
                
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
                if self.audio_queue.full():
                    try:
                        self.audio_queue.get_nowait()
                    except Empty:
                        pass
                self.audio_queue.put_nowait(audio_data)
        except Exception as e:
            self.logger.error(f"Error queueing audio data: {str(e)}")
            self.emit_alert('warning', 'Error with audio feed', True)

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
                    'detection_enabled': True if hasattr(self.monitor_system, 'person_detector') else False,
                    'dev_mode': self.dev_mode
                }
            
            if len(self.connected_clients) > 0:
                self.socketio.emit('status', status_data)
        except Exception as e:
            self.logger.error(f"Error emitting status: {str(e)}")

    def _handle_alert(self, alert):
        """Handle new alerts from AlertManager."""
        try:
            if len(self.connected_clients) > 0:
                self.socketio.emit('alert', {
                    'type': alert.type.value,
                    'level': alert.level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.strftime('%H:%M:%S'),
                    'details': alert.details,
                    'should_notify': alert.should_notify
                })
        except Exception as e:
            self.logger.error(f"Error handling alert: {str(e)}")

    def emit_metrics(self, metrics_data):
        """Emit performance metrics to connected clients."""
        try:
            if len(self.connected_clients) > 0:
                self.socketio.emit('metrics', metrics_data)
                
                # Update metrics history
                with self.metrics_lock:
                    for key, value in metrics_data.items():
                        if key in self.metrics_history:
                            self.metrics_history[key].append(value)
                            # Keep only the last N values
                            if len(self.metrics_history[key]) > self.metrics_max_history:
                                self.metrics_history[key] = self.metrics_history[key][-self.metrics_max_history:]
        except Exception as e:
            self.logger.error(f"Error emitting metrics: {str(e)}")

    def _run_webrtc_loop(self):
        """Run the asyncio event loop for WebRTC."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self):
        """Start the web server."""
        self.should_run = True
        
        # Start WebRTC event loop
        self.webrtc_thread = threading.Thread(target=self._run_webrtc_loop, daemon=True)
        self.webrtc_thread.start()
        
        # Start background tasks
        self.start_background_tasks()
        
        # Start the server
        self.server_thread = threading.Thread(
            target=self.socketio.run,
            args=(self.app,),
            kwargs={
                "host": self.host,
                "port": self.port,
                "debug": self.dev_mode,
                "use_reloader": False,
                "allow_unsafe_werkzeug": True,
            },
            daemon=True,
        )
        self.server_thread.start()
        self.logger.info(f"Web server started on http://{self.host}:{self.port}")
        
        return self.server_thread

    def stop(self):
        """Stop the web server."""
        self.should_run = False
        
        # Stop WebRTC connections
        asyncio.run_coroutine_threadsafe(self.webrtc_streamer.close_all(), self.loop)
        
        # Stop the event loop
        self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
            
        if self.webrtc_thread and self.webrtc_thread.is_alive():
            self.webrtc_thread.join(timeout=5)
            
        self.logger.info("Web server stopped")


# Create instances for the application
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
monitor = BabyMonitorWeb()
