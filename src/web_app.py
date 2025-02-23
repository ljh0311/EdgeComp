"""
Baby Monitor Web Application
=========================
Web interface for the Baby Monitor System using Flask and Flask-SocketIO.
"""

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import threading
import logging
import time
import base64
import os
import signal
from datetime import datetime

from utils.config import Config
from utils.camera import Camera
from detectors.person_detector import PersonDetector
from detectors.motion_detector import MotionDetector
from detectors.emotion_detector import EmotionDetector

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

class BabyMonitorWeb:
    def __init__(self):
        """Initialize the Baby Monitor web application."""
        # Setup logging
        logging.basicConfig(**Config.LOGGING)
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self.is_running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.camera_error = False
        self.last_frame_time = 0
        self.frame_timeout = 5  # Seconds before considering camera as failed
        
        # Initialize components
        self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
        self.person_detector = PersonDetector(Config.PERSON_DETECTION)
        self.motion_detector = MotionDetector(Config.MOTION_DETECTION)
        
        # Try to initialize emotion detector (optional)
        try:
            self.emotion_detector = EmotionDetector(Config.EMOTION_DETECTION)
            self.has_emotion_detector = True
            self.setup_audio()
        except Exception as e:
            self.logger.warning(f"Emotion detection disabled: {str(e)}")
            self.has_emotion_detector = False
        
        # Start processing thread
        self.start()
        
    def setup_audio(self):
        """Setup audio capture for emotion detection."""
        try:
            import pyaudio
            
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=Config.EMOTION_DETECTION['sampling_rate'],
                input=True,
                frames_per_buffer=Config.EMOTION_DETECTION['chunk_size'],
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            self.logger.info("Audio capture initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio capture: {str(e)}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Handle incoming audio data."""
        try:
            # Skip processing if emotion detection is not available
            if not self.has_emotion_detector:
                return (in_data, pyaudio.paContinue)
                
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Process emotions
            emotion, confidence = self.emotion_detector.detect(audio_data)
            
            if emotion:
                # Get alert level
                level = self.emotion_detector.get_emotion_level(emotion, confidence)
                
                if level:
                    # Emit alert via WebSocket
                    message = f"Detected {emotion.lower()} emotion (confidence: {confidence:.2f})"
                    socketio.emit('alert', {'message': message, 'level': level})
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
        
        return (in_data, pyaudio.paContinue)
    
    def start(self):
        """Start the monitoring system."""
        try:
            # Initialize camera
            if not self.camera.initialize():
                self.logger.error("Failed to initialize camera")
                self.camera_error = True
                socketio.emit('camera_error', {'message': 'Failed to initialize camera. Please check your camera connection.'})
                return
            
            self.is_running = True
            self.camera_error = False
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self.process_frames)
            self.process_thread.daemon = True  # Make thread daemon so it stops when main thread stops
            self.process_thread.start()
            
            self.logger.info("Baby Monitor system started successfully")
            socketio.emit('status', {
                'camera_error': False,
                'has_emotion_detector': self.has_emotion_detector,
                'is_running': True
            })
            
        except Exception as e:
            self.logger.error(f"Error starting application: {str(e)}")
            self.camera_error = True
            socketio.emit('camera_error', {'message': f'Failed to start system: {str(e)}'})
    
    def process_frames(self):
        """Process video frames in a separate thread."""
        consecutive_errors = 0
        max_consecutive_errors = 5
        error_cooldown = 0
        
        while self.is_running:
            try:
                # Get frame from camera
                ret, frame = self.camera.get_frame()
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        if time.time() > error_cooldown:
                            self.logger.error("Multiple consecutive camera errors detected")
                            self.camera_error = True
                            socketio.emit('camera_error', {'message': 'Camera connection lost! Please check your camera connection.'})
                            error_cooldown = time.time() + 30
                        time.sleep(0.5)
                        continue
                    time.sleep(0.1)
                    continue
                
                # Reset error counter on successful frame
                consecutive_errors = 0
                self.camera_error = False
                self.last_frame_time = time.time()
                
                # Create a copy for processing
                process_frame = frame.copy()
                
                try:
                    # Detect people
                    frame, person_count, boxes = self.person_detector.detect(process_frame)
                    
                    # Emit person count update
                    socketio.emit('person_count', {'count': person_count})
                    
                    # Detect motion and falls
                    frame, rapid_motion, fall_detected = self.motion_detector.detect(frame, boxes)
                    
                    # Update frame
                    with self.frame_lock:
                        self.frame = frame
                    
                    # Convert frame for streaming
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit frame via WebSocket
                    socketio.emit('frame', {'frame': frame_bytes})
                    
                    # Handle alerts
                    if fall_detected:
                        socketio.emit('alert', {
                            'message': "FALL DETECTED! Check on person immediately!",
                            'level': "critical"
                        })
                    elif rapid_motion:
                        socketio.emit('alert', {
                            'message': "RAPID MOTION DETECTED! Monitor situation",
                            'level': "warning"
                        })
                
                except Exception as e:
                    self.logger.error(f"Error processing detection: {str(e)}")
                    continue
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error processing frame: {str(e)}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    if time.time() > error_cooldown:
                        self.logger.error("Multiple consecutive camera errors detected")
                        self.camera_error = True
                        socketio.emit('camera_error', {'message': 'Camera connection lost!'})
                        error_cooldown = time.time() + 30
                    time.sleep(0.5)
                    continue
                time.sleep(0.1)
    
    def stop(self):
        """Stop the monitoring system."""
        self.is_running = False
        
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
        
        # Stop audio
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        self.camera.release()
        self.logger.info("Baby Monitor system stopped")

# Create monitor instance
monitor = BabyMonitorWeb()

@app.route('/')
def index():
    """Render the main monitoring page."""
    return render_template('index.html')

@app.route('/status')
def status():
    """Get the current system status."""
    # Check if we haven't received frames for too long
    if monitor.last_frame_time > 0 and time.time() - monitor.last_frame_time > monitor.frame_timeout:
        monitor.camera_error = True
    
    return jsonify({
        'camera_error': monitor.camera_error,
        'has_emotion_detector': monitor.has_emotion_detector,
        'is_running': monitor.is_running
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    # Check if we haven't received frames for too long
    if monitor.last_frame_time > 0 and time.time() - monitor.last_frame_time > monitor.frame_timeout:
        monitor.camera_error = True
    
    emit('status', {
        'camera_error': monitor.camera_error,
        'has_emotion_detector': monitor.has_emotion_detector,
        'is_running': monitor.is_running
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    pass

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the application gracefully."""
    try:
        # Stop the monitor
        monitor.stop()
        
        # Get the parent process ID
        pid = os.getppid()
        
        # Schedule the shutdown after sending response
        def shutdown_server():
            time.sleep(1)  # Give time for response to be sent
            os.kill(pid, signal.SIGTERM)
        
        threading.Thread(target=shutdown_server).start()
        
        return jsonify({'status': 'success', 'message': 'Application shutting down...'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000) 