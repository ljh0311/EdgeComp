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
        
        # System status
        self.system_status = {
            'uptime': '00:00:00',
            'cpu_usage': 0,
            'memory_usage': 0,
            'camera_status': 'connected' if self.camera else 'disconnected',
            'person_detector_status': 'running' if self.person_detector else 'stopped',
            'emotion_detector_status': 'running' if self.emotion_detector else 'stopped',
        }
        
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
        
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page"""
            if self.mode == "dev":
                return redirect(url_for('metrics'))
            return render_template('index.html', mode=self.mode)
        
        @self.app.route('/metrics')
        def metrics():
            """Metrics page"""
            return render_template('metrics.html', mode=self.mode)
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video feed endpoint"""
            return Response(self._generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for metrics"""
            return jsonify(self.metrics)
        
        @self.app.route('/api/system_info')
        def api_system_info():
            """API endpoint for system info"""
            return jsonify(self.system_status)
        
        @self.app.route('/repair')
        def repair_tools():
            """Repair tools page"""
            return render_template('repair_tools.html', mode=self.mode)
        
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
                
                return jsonify({'status': 'error', 'message': 'Invalid repair tool'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
        if self.mode == "dev":
            @self.app.route('/dev/tools')
            def dev_tools():
                """Developer tools page"""
                return render_template('dev_tools.html')
            
            @self.app.route('/dev/logs')
            def dev_logs():
                """Logs page"""
                return render_template('logs.html')
    
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
    
    def _process_frames(self):
        """Process frames from camera"""
        import cv2
        
        # Initialize detection history for smoothing
        detection_history = []
        max_history = 5
        last_fps_time = time.time()
        frame_count = 0
        
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
                
                # Get emotion detection results
                if self.emotion_detector:
                    try:
                        emotion_results = self.emotion_detector.get_current_state()
                        if emotion_results:
                            # Update current emotion metrics
                            self.metrics['current']['emotion'] = emotion_results['emotion']
                            self.metrics['current']['emotion_confidence'] = emotion_results['confidence']
                            
                            # Update emotion history
                            emotion = emotion_results['emotion']
                            if emotion in self.metrics['history']['emotions']:
                                self.metrics['history']['emotions'][emotion] += 1
                            
                            # Emit emotion update via Socket.IO
                            self.socketio.emit('emotion_update', {
                                'emotion': emotion_results['emotion'],
                                'confidence': emotion_results['confidence'],
                                'confidences': emotion_results['confidences']
                            })
                    except Exception as e:
                        logger.error(f"Error getting emotion state: {e}")
                
                # Update detection metrics and emit update
                if 'detections' in results:
                    detection_count = len(results['detections'])
                    self.socketio.emit('detection_update', {
                        'count': detection_count,
                        'fps': self.metrics['current']['fps']
                    })
                
                # Create a copy of the frame for drawing
                display_frame = frame.copy()
                
                # Draw detection results on the frame with improved visualization
                if 'detections' in results:
                    # Count detections by type
                    face_count = 0
                    upper_body_count = 0
                    full_body_count = 0
                    
                    for detection in results['detections']:
                        if 'bbox' in detection:
                            x1, y1, x2, y2 = detection['bbox']
                            confidence = detection.get('confidence', 0.0)
                            detection_class = detection.get('class', 'person')
                            
                            # Skip low confidence detections
                            if confidence < 0.5:
                                continue
                                
                            # Count by type
                            if detection_class == 'face':
                                face_count += 1
                            elif detection_class == 'upper_body':
                                upper_body_count += 1
                            elif detection_class == 'full_body':
                                full_body_count += 1
                            
                            # Draw bounding box with different colors based on type
                            if detection_class == 'face':
                                color = (0, 255, 0)  # Green for face
                            elif detection_class == 'upper_body':
                                color = (255, 0, 0)  # Blue for upper body
                            elif detection_class == 'full_body':
                                color = (0, 0, 255)  # Red for full body
                            else:
                                color = (255, 255, 0)  # Yellow for other
                            
                            # Draw bounding box with thickness based on confidence
                            thickness = max(1, int(confidence * 3))
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                            
                            # Draw label with confidence
                            label = f"{detection_class}: {confidence:.2f}"
                            
                            # Get text size for background
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            
                            # Draw semi-transparent background for text
                            cv2.rectangle(
                                display_frame, 
                                (x1, y1 - text_size[1] - 10), 
                                (x1 + text_size[0], y1), 
                                color, 
                                -1
                            )
                            
                            # Draw text
                            cv2.putText(
                                display_frame, 
                                label, 
                                (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 0, 0), 
                                1, 
                                cv2.LINE_AA
                            )
                    
                    # Update metrics with detection counts
                    total_detections = face_count + upper_body_count + full_body_count
                    self.metrics['current']['detections'] = total_detections
                    
                    # Update detection types
                    if face_count > 0:
                        self.metrics['detection_types']['face'] += face_count
                    if upper_body_count > 0:
                        self.metrics['detection_types']['upper_body'] += upper_body_count
                    if full_body_count > 0:
                        self.metrics['detection_types']['full_body'] += full_body_count
                    
                    # Update total detections if any were found
                    if total_detections > 0:
                        self.metrics['total_detections'] += 1
                
                # Update FPS
                if 'fps' in results:
                    self.metrics['current']['fps'] = results['fps']
                
                # Add monitoring status indicator
                cv2.rectangle(display_frame, (display_frame.shape[1] - 120, 10), (display_frame.shape[1] - 10, 40), (0, 0, 0), -1)
                cv2.putText(
                    display_frame, 
                    "Monitoring", 
                    (display_frame.shape[1] - 115, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    1, 
                    cv2.LINE_AA
                )
                
                # Add timestamp to the frame
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    display_frame, 
                    timestamp, 
                    (10, display_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )
                
                # Add current emotion to the frame
                if self.emotion_detector:
                    emotion_text = f"Emotion: {self.metrics['current']['emotion']} ({self.metrics['current']['emotion_confidence']:.2f})"
                    cv2.putText(
                        display_frame,
                        emotion_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', display_frame)
                
                # Update frame buffer
                with self.frame_lock:
                    self.frame_buffer = buffer.tobytes()
                
                # Sleep to control processing rate
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(0.1)
    
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
                if self.emotion_detector and hasattr(self.emotion_detector, 'current_emotion'):
                    current_emotion = self.emotion_detector.current_emotion
                
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
                        'fps': self.system_status['fps']
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