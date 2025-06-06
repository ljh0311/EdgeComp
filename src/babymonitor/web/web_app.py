"""
Baby Monitor Web Application
=========================
Web interface for the Baby Monitor System using Flask and Flask-SocketIO.
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO
import cv2
import numpy as np
import threading
import logging
import time
import base64
import os
import signal
import psutil
import datetime
from scipy import signal as scipy_signal
from datetime import datetime
from queue import Queue, Empty
from threading import Lock
import sys
import platform
import subprocess
from pathlib import Path


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
        self.metrics_lock = Lock()  # For performance metrics
        
        # Add frame and audio queues with size limits
        self.frame_queue = Queue(maxsize=10)
        self.audio_queue = Queue(maxsize=10)
        
        # Performance metrics storage
        self.metrics_history = {
            'fps': [],
            'frame_time': [],
            'cpu_usage': [],
            'memory_usage': []
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
        # Pre-allocate reusable buffer for frame processing
        resized_frame = None
        
        while self.should_run:
            try:
                if not self.frame_queue.empty():
                    with self.frame_lock:
                        frame = self.frame_queue.get_nowait()
                        current_time = time.time()
                        if current_time - self.last_frame_time >= self.frame_interval:
                            if len(self.connected_clients) > 0:
                                try:
                                    # Validate frame
                                    if not isinstance(frame, np.ndarray):
                                        continue

                                    # Resize frame if needed
                                    if frame.shape[1] > 1280:  # If width > 1280
                                        scale = 1280 / frame.shape[1]
                                        new_width = int(frame.shape[1] * scale)
                                        new_height = int(frame.shape[0] * scale)
                                        
                                        # Reuse or create resized frame buffer
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

                                    # Ensure frame is in correct format (BGR)
                                    if len(frame_to_encode.shape) == 2:
                                        frame_to_encode = cv2.cvtColor(frame_to_encode, cv2.COLOR_GRAY2BGR)
                                    elif frame_to_encode.shape[2] == 4:
                                        frame_to_encode = cv2.cvtColor(frame_to_encode, cv2.COLOR_RGBA2BGR)
                                    
                                    # Convert frame to JPEG with quality control
                                    _, buffer = cv2.imencode('.jpg', frame_to_encode, 
                                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                                    frame_data = base64.b64encode(buffer).decode('utf-8')
                                    
                                    # Emit with error handling
                                    try:
                                        self.socketio.emit('frame', {'data': frame_data})
                                        self.last_frame_time = current_time
                                    except Exception as e:
                                        self.logger.error(f"Socket.IO emission error: {str(e)}")
                                except Exception as e:
                                    self.logger.error(f"Frame processing error: {str(e)}")
                else:
                    time.sleep(0.001)  # Short sleep when queue is empty
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

        @self.app.route('/repair')
        def repair_tools():
            """Render the repair tools page."""
            return render_template('repair_tools.html', dev_mode=self.dev_mode)

        @self.app.route('/metrics')
        def metrics_page():
            """Render the metrics page."""
            return render_template('metrics.html', dev_mode=self.dev_mode)

        @self.app.route('/dev/logs')
        def dev_logs():
            """Render the logs page (dev mode only)."""
            if not self.dev_mode:
                return redirect(url_for('index'))
            return render_template('logs.html', dev_mode=self.dev_mode)

        @self.app.route('/repair/api/cameras')
        def get_cameras():
            """Get available cameras and their capabilities."""
            if not self.monitor_system:
                return jsonify({'error': 'Monitor system not initialized'}), 503
            
            try:
                cameras = []
                # Get available cameras
                try:
                    available_cameras = self.monitor_system.camera.get_available_cameras()
                except (AttributeError, Exception) as e:
                    available_cameras = [0]  # Default to camera 0
                    logging.warning(f"Error getting available cameras: {str(e)}")
                
                # Get current camera
                try:
                    current_camera = self.monitor_system.camera.get_current_camera()
                except (AttributeError, Exception) as e:
                    current_camera = 0
                    logging.warning(f"Error getting current camera: {str(e)}")
                
                # Add available cameras with resolutions
                for idx in available_cameras:
                    try:
                        resolutions = self.monitor_system.camera.get_camera_resolutions(idx)
                    except (AttributeError, Exception):
                        resolutions = ['640x480', '1280x720', '1920x1080']  # Default resolutions
                    
                    cameras.append({
                        'id': str(idx),
                        'name': f'Camera {idx}',
                        'resolutions': resolutions
                    })
                
                return jsonify({
                    'cameras': cameras,
                    'current_camera': str(current_camera) if current_camera is not None else '0'
                })
            except Exception as e:
                logging.error(f"Error in get_cameras: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/repair/api/camera/configure', methods=['POST'])
        def configure_camera():
            """Configure camera settings."""
            if not self.monitor_system:
                return jsonify({'error': 'Monitor system not initialized'}), 503
            
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                width = data.get('width')
                height = data.get('height')
                
                if not all([width, height]):
                    return jsonify({'error': 'Missing required parameters'}), 400
                
                # Try to set resolution
                try:
                    success = self.monitor_system.camera.set_resolution(width, height)
                    if not success:
                        return jsonify({'error': 'Failed to set resolution'}), 400
                except Exception as e:
                    return jsonify({'error': f'Error setting resolution: {str(e)}'}), 500
                
                return jsonify({
                    'status': 'success',
                    'message': f'Camera configured with resolution {width}x{height}'
                })
            except Exception as e:
                self.logger.error(f"Error configuring camera: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/repair/api/system/memory')
        def api_system_memory():
            """Get system memory information."""
            try:
                memory = psutil.virtual_memory()
                current_resolution = None
                
                # Safely get current resolution
                if self.monitor_system and hasattr(self.monitor_system, 'camera'):
                    try:
                        current_resolution = self.monitor_system.camera.get_current_resolution()
                    except:
                        pass
                
                return jsonify({
                    'total_memory': memory.total / (1024 * 1024),  # Convert to MB
                    'available_memory': memory.available / (1024 * 1024),  # Convert to MB
                    'used_memory': memory.used / (1024 * 1024),  # Convert to MB
                    'memory_percent': memory.percent,
                    'current_resolution': current_resolution
                })
            except Exception as e:
                self.logger.error(f"Error getting system memory: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/repair/api/system/info')
        def api_system_info():
            """Get system information."""
            if not self.monitor_system:
                return jsonify({'error': 'Monitor system not initialized'}), 503
            
            try:
                # Get system info
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Get camera and audio status
                try:
                    camera_status = "Active" if self.monitor_system.camera and self.monitor_system.camera.is_active() else "Inactive"
                except (AttributeError, Exception):
                    camera_status = "Unknown"
                
                try:
                    audio_status = "Active" if self.monitor_system.audio_processor and self.monitor_system.audio_processor.is_active() else "Inactive"
                except (AttributeError, Exception):
                    audio_status = "Unknown"
                
                # Calculate uptime
                uptime = datetime.now() - datetime.fromtimestamp(psutil.boot_time())
                uptime_str = f"{uptime.days}d {uptime.seconds // 3600}h {(uptime.seconds // 60) % 60}m"
                
                return jsonify({
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory.percent,
                    'memory_available': memory.available / (1024 * 1024),  # MB
                    'disk_usage': disk.percent,
                    'camera_status': camera_status,
                    'audio_status': audio_status,
                    'uptime': uptime_str
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/repair/api/launch_desktop_app', methods=['POST'])
        def launch_desktop_app():
            """Launch the desktop repair and installation tool."""
            try:
                # Get project root directory
                base_dir = Path(__file__).resolve().parents[3]  # Go up 3 levels from web_app.py
                scripts_dir = base_dir / "scripts"
                
                if platform.system() == "Windows":
                    script_path = scripts_dir / "scripts_manager.bat"
                    # Use subprocess.Popen to avoid blocking
                    subprocess.Popen(str(script_path), shell=True, cwd=str(base_dir))
                else:
                    script_path = scripts_dir / "scripts_manager_gui.py"
                    # Use subprocess.Popen to avoid blocking
                    subprocess.Popen([sys.executable, str(script_path)], cwd=str(base_dir))
                
                return jsonify({
                    'status': 'success',
                    'message': 'Desktop application launched successfully'
                })
            except Exception as e:
                self.logger.error(f"Error launching desktop app: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to launch desktop application: {str(e)}'
                }), 500

        @self.app.route('/repair/run', methods=['POST'])
        def repair_run():
            """Run repair tools."""
            if not self.monitor_system:
                return jsonify({'error': 'Monitor system not initialized'}), 503
            
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                tool = data.get('tool')
                if not tool:
                    return jsonify({'error': 'No tool specified'}), 400
                
                if tool == 'restart_camera':
                    # Handle camera restart
                    if hasattr(self.monitor_system, 'camera'):
                        try:
                            self.monitor_system.camera.release()
                            time.sleep(1)
                            self.monitor_system.camera.initialize()
                            return jsonify({'status': 'success', 'message': 'Camera restarted successfully'})
                        except Exception as e:
                            return jsonify({'error': f'Failed to restart camera: {str(e)}'}), 500
                
                elif tool == 'restart_audio':
                    # Handle audio restart
                    if hasattr(self.monitor_system, 'audio_processor'):
                        try:
                            self.monitor_system.audio_processor.restart()
                            return jsonify({'status': 'success', 'message': 'Audio system restarted successfully'})
                        except Exception as e:
                            return jsonify({'error': f'Failed to restart audio: {str(e)}'}), 500
                
                elif tool == 'check_system':
                    # Perform system check
                    try:
                        status = {
                            'cpu': psutil.cpu_percent(interval=0.1),
                            'memory': psutil.virtual_memory().percent,
                            'disk': psutil.disk_usage('/').percent,
                            'camera': 'OK' if self.monitor_system.camera_enabled else 'Inactive',
                            'audio': 'OK' if self.monitor_system.audio_enabled else 'Inactive'
                        }
                        return jsonify({'status': 'success', 'message': 'System check completed', 'data': status})
                    except Exception as e:
                        return jsonify({'error': f'System check failed: {str(e)}'}), 500
                
                elif tool == 'restart_system':
                    # Handle full system restart
                    try:
                        if hasattr(self.monitor_system, 'restart'):
                            self.monitor_system.restart()
                            return jsonify({'status': 'success', 'message': 'System restarted successfully'})
                        else:
                            return jsonify({'error': 'System restart not supported'}), 501
                    except Exception as e:
                        return jsonify({'error': f'Failed to restart system: {str(e)}'}), 500
                
                else:
                    return jsonify({'error': f'Unknown tool: {tool}'}), 400
                
            except Exception as e:
                self.logger.error(f"Error running repair tool: {str(e)}")
                return jsonify({'error': str(e)}), 500

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

        @self.app.route('/video-feed')
        def video_feed():
            """Direct MJPEG camera feed."""
            if not self.monitor_system or not self.monitor_system.camera_enabled:
                return "Camera not available", 404
            return self._stream_camera()

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
                return jsonify({'error': str(e)}), 500

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

        @self.socketio.on('request_metrics')
        def handle_request_metrics():
            """Handle client request for current metrics."""
            with self.metrics_lock:
                self.socketio.emit('metrics_update', self.metrics_history)

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
                was_enabled = self.monitor_system.camera_enabled
                if was_enabled:
                    self.monitor_system.toggle_camera()
                
                success = self.monitor_system.camera.select_camera(camera_name)
                if success:
                    # Clear frame queue
                    with self.frame_lock:
                        while not self.frame_queue.empty():
                            self.frame_queue.get_nowait()
                    
                    # Reset motion detector
                    if hasattr(self.monitor_system, 'motion_detector'):
                        self.monitor_system.motion_detector.reset()
                    
                    # Restart camera if it was enabled
                    if was_enabled:
                        # Small delay to ensure camera is properly initialized
                        time.sleep(0.1)
                        self.monitor_system.toggle_camera()
                    
                    # Get current resolution
                    current_res = self.monitor_system.camera.get_current_resolution()
                    return {
                        'success': True,
                        'message': f'Camera {camera_name} selected',
                        'current_resolution': current_res
                    }
                else:
                    # If camera selection failed and camera was enabled, try to re-enable it
                    if was_enabled:
                        self.monitor_system.toggle_camera()
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
                    
                    # Stop camera if running
                    was_enabled = self.monitor_system.camera_enabled
                    if was_enabled:
                        self.monitor_system.toggle_camera()
                    
                    success = self.monitor_system.camera.set_resolution(f"{width}x{height}")
                    
                    if success:
                        # Clear frame queue
                        with self.frame_lock:
                            while not self.frame_queue.empty():
                                self.frame_queue.get_nowait()
                        
                        # Reset motion detector
                        if hasattr(self.monitor_system, 'motion_detector'):
                            self.monitor_system.motion_detector.reset()
                        
                        # Restart camera if it was enabled
                        if was_enabled:
                            # Small delay to ensure camera is properly initialized
                            time.sleep(0.1)
                            self.monitor_system.toggle_camera()
                        
                        current_res = self.monitor_system.camera.get_current_resolution()
                        return {'success': True, 'message': f'Resolution set to {current_res}', 'current_resolution': current_res}
                    else:
                        # If resolution change failed and camera was enabled, try to re-enable it
                        if was_enabled:
                            self.monitor_system.toggle_camera()
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

        @self.socketio.on('run_test')
        def handle_run_test(data):
            """Handle test execution requests."""
            try:
                test_type = data.get('type')
                if not test_type:
                    return {'success': False, 'error': 'No test type specified'}

                # Emit test started event
                self.socketio.emit('test_started', {'type': test_type})

                # Import test module
                from babymonitor.tests.test_camera_performance import test_camera_only, test_motion_detection

                # Run the requested test
                if test_type == 'camera':
                    # Run in a separate thread to not block
                    thread = threading.Thread(target=test_camera_only)
                    thread.daemon = True
                    thread.start()
                    return {'success': True, 'message': 'Camera test started'}
                
                elif test_type == 'motion':
                    thread = threading.Thread(target=test_motion_detection)
                    thread.daemon = True
                    thread.start()
                    return {'success': True, 'message': 'Motion detection test started'}
                
                elif test_type == 'full':
                    def run_full_test():
                        test_camera_only()
                        time.sleep(1)  # Brief pause between tests
                        test_motion_detection()
                    
                    thread = threading.Thread(target=run_full_test)
                    thread.daemon = True
                    thread.start()
                    return {'success': True, 'message': 'Full performance test started'}
                
                elif test_type == 'debug':
                    # Toggle debug mode
                    self.dev_mode = not self.dev_mode
                    return {
                        'success': True,
                        'message': f'Debug mode {"enabled" if self.dev_mode else "disabled"}',
                        'dev_mode': self.dev_mode
                    }
                
                else:
                    return {'success': False, 'error': f'Unknown test type: {test_type}'}

            except Exception as e:
                self.logger.error(f"Error running test: {str(e)}")
                self.socketio.emit('test_error', {'error': str(e)})
                return {'success': False, 'error': str(e)}

    def emit_frame(self, frame):
        """Queue frame for emission to connected clients."""
        try:
            if frame is None:
                self.logger.debug("No frame to emit")
                if self.monitor_system and self.monitor_system.camera_enabled:
                    self.emit_alert('warning', 'Camera is enabled but no frames are being received', True)
                return
                
            if len(self.connected_clients) == 0:
                return  # Skip processing if no clients

            # Validate frame format
            if not isinstance(frame, np.ndarray):
                self.logger.error("Invalid frame format")
                return

            # Ensure frame is in correct format (BGR)
            if len(frame.shape) == 2:  # If grayscale, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # If RGBA, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Only queue frame if we're not too backed up
            if self.frame_queue.qsize() < self.frame_queue.maxsize - 1:
                self.frame_queue.put_nowait(frame)
            else:
                # Clear queue if backed up
                try:
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)  # Add latest frame
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
                # Format emotion data
                emotion_data = {
                    'emotion': emotion.lower(),  # Ensure lowercase to match frontend
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                
                # Emit emotion update
                self.socketio.emit('emotion', emotion_data)
                
                # Emit alert for concerning emotions
                if emotion.lower() in ['anger', 'fear', 'sadness'] and confidence > 0.7:
                    self.emit_alert('warning', f'High confidence {emotion.lower()} emotion detected', True)
                elif emotion.lower() == 'worried' and confidence > 0.8:
                    self.emit_alert('warning', 'High distress detected', True)
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
                    'emotion_enabled': True if (hasattr(self.monitor_system, 'emotion_recognizer') and 
                                             self.monitor_system.emotion_recognizer is not None) else False,
                    'detection_enabled': True if hasattr(self.monitor_system, 'person_detector') else False
                }
            
            # Check for enabled features without data
            if status_data:
                if status_data.get('camera_enabled') and not hasattr(self.monitor_system, 'camera'):
                    self.emit_alert('warning', 'Camera is enabled but not properly initialized', True)
                if status_data.get('audio_enabled'):
                    if not hasattr(self.monitor_system, 'audio_processor'):
                        self.emit_alert('warning', 'Audio is enabled but not properly initialized', True)
                    if not hasattr(self.monitor_system, 'emotion_recognizer'):
                        self.emit_alert('warning', 'Emotion recognition is not available', False)
            
            if len(self.connected_clients) > 0:
                self.socketio.emit('status', status_data)
        except Exception as e:
            self.logger.error(f"Error emitting status: {str(e)}")

    def emit_metrics(self, metrics_data):
        """Emit performance metrics to connected clients."""
        try:
            with self.metrics_lock:
                # Update metrics history
                for key, value in metrics_data.items():
                    if key in self.metrics_history:
                        self.metrics_history[key].append(value)
                        # Keep only recent history
                        if len(self.metrics_history[key]) > self.metrics_max_history:
                            self.metrics_history[key].pop(0)

                # Emit to connected clients
                if len(self.connected_clients) > 0:
                    self.socketio.emit('metrics_update', {
                        'current': metrics_data,
                        'history': self.metrics_history
                    })
        except Exception as e:
            self.logger.error(f"Error emitting metrics: {str(e)}")

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

    def _stream_camera(self):
        """Generate MJPEG stream from camera."""
        def generate():
            while self.monitor_system and self.monitor_system.camera_enabled:
                try:
                    frame = self.monitor_system.camera.get_frame()
                    if frame is not None and isinstance(frame, np.ndarray):
                        # Process frame with person detector if available
                        if hasattr(self.monitor_system, 'person_detector'):
                            processed_frame = self.monitor_system.person_detector.process_frame(frame)
                            if processed_frame is not None and isinstance(processed_frame, np.ndarray):
                                frame = processed_frame
                        
                        # Ensure frame is in correct format (BGR)
                        if len(frame.shape) == 2:  # If grayscale, convert to BGR
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        elif frame.shape[2] == 4:  # If RGBA, convert to BGR
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                        
                        # Encode frame to JPEG with quality control and error handling
                        try:
                            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if jpeg is not None:
                                frame_bytes = jpeg.tobytes()
                                yield (b'--frame\r\n'
                                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        except cv2.error as e:
                            self.logger.error(f"Error encoding frame: {str(e)}")
                            continue
                            
                    time.sleep(0.016)  # ~60 FPS max
                except Exception as e:
                    self.logger.error(f"Error in video feed: {str(e)}")
                    time.sleep(0.1)
                    continue
        
        return self.app.response_class(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
