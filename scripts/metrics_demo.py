#!/usr/bin/env python
"""
Standalone Metrics Dashboard Demo

This script runs a standalone Flask server with Socket.IO that
generates demo data for the metrics dashboard. It doesn't require
any imports from the babymonitor package.

Usage:
    python scripts/metrics_demo.py
"""

import os
import sys
import time
import random
import threading
import logging
import platform
from pathlib import Path
from collections import deque

try:
    import psutil
except ImportError:
    print("Installing psutil...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO
except ImportError:
    print("Installing Flask and Flask-SocketIO...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-socketio"])
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class MetricsCollector:
    """Collects and stores metrics from the baby monitor system."""
    
    def __init__(self, history_size=20):
        """Initialize metrics collector with empty histories."""
        self.start_time = time.time()
        self.frame_count = 0
        self.fps_history = deque(maxlen=history_size)
        self.detection_count_history = deque(maxlen=history_size)
        self.cpu_usage_history = deque(maxlen=history_size)
        self.memory_usage_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        
        # Emotion tracking
        self.emotions = {
            'crying': deque(maxlen=history_size),
            'laughing': deque(maxlen=history_size),
            'babbling': deque(maxlen=history_size),
            'silence': deque(maxlen=history_size)
        }
        
        # Current emotion distribution (percentages)
        self.current_emotions = {
            'crying': 0,
            'laughing': 0,
            'babbling': 0,
            'silence': 100  # Default to silence
        }
        
        # System info
        self.system_info = {
            'os': f"{platform.system()} {platform.release()}",
            'python_version': platform.python_version(),
            'opencv_version': "4.5.4",  # Default value
            'uptime': '00:00:00',
            'person_detector': 'YOLOv8',
            'detector_model': 'YOLOv8n',
            'emotion_detector': 'Active',
            'detection_threshold': '0.7',
            'camera_resolution': '640x480',
            'audio_sample_rate': '16000 Hz',
            'frame_skip': '2',
            'process_resolution': '640x480',
            'confidence_threshold': '0.7',
            'detection_history_size': f"{history_size} frames"
        }
        
        # Initialize with empty data
        for _ in range(history_size):
            self.fps_history.append(0)
            self.detection_count_history.append(0)
            self.cpu_usage_history.append(0)
            self.memory_usage_history.append(0)
            self.confidence_history.append(0)
            
            # Initialize emotion histories
            for emotion in self.emotions:
                self.emotions[emotion].append(0)
    
    def update_frame_metrics(self, frame_time, detection_results=None):
        """Update metrics with new frame data."""
        # Update frame count
        self.frame_count += 1
        
        # Calculate FPS
        fps = 1.0 / frame_time if frame_time > 0 else 0
        self.fps_history.append(round(fps, 2))
        
        # Update detection count
        detection_count = len(detection_results) if detection_results else 0
        self.detection_count_history.append(detection_count)
        
        # Calculate average confidence if detections exist
        if detection_results and len(detection_results) > 0:
            # Assuming detection_results is a list of dicts with a 'confidence' key
            avg_confidence = sum(d.get('confidence', 0) * 100 for d in detection_results) / len(detection_results)
            self.confidence_history.append(round(avg_confidence, 2))
        else:
            self.confidence_history.append(0)
        
        # Update system resource usage
        self.cpu_usage_history.append(round(psutil.cpu_percent(), 2))
        self.memory_usage_history.append(round(psutil.virtual_memory().percent, 2))
        
        # Update uptime
        uptime_seconds = time.time() - self.start_time
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.system_info['uptime'] = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    
    def update_emotion_metrics(self, emotion_result=None):
        """Update emotion metrics based on detection results."""
        if not emotion_result:
            # If no emotion data, assume silence
            self.current_emotions = {
                'crying': 0,
                'laughing': 0,
                'babbling': 0,
                'silence': 100
            }
            
            # Update emotion histories
            for emotion, value in self.current_emotions.items():
                self.emotions[emotion].append(value)
            return
        
        # Process emotion results
        # emotion_result should be a dict with percentages for each emotion
        # Example: {'crying': 75, 'laughing': 0, 'babbling': 0, 'silence': 25}
        self.current_emotions = emotion_result
        
        # Ensure all emotions are present
        for emotion in ['crying', 'laughing', 'babbling', 'silence']:
            if emotion not in self.current_emotions:
                self.current_emotions[emotion] = 0
        
        # Ensure percentages sum to 100%
        total = sum(self.current_emotions.values())
        if total > 0 and total != 100:
            for emotion in self.current_emotions:
                self.current_emotions[emotion] = (self.current_emotions[emotion] / total) * 100
        
        # Update emotion histories
        for emotion, value in self.current_emotions.items():
            self.emotions[emotion].append(value)
    
    def get_metrics(self):
        """Get current metrics data."""
        # Ensure all arrays are the same length (pad if necessary)
        max_len = max(
            len(self.fps_history),
            len(self.detection_count_history),
            len(self.cpu_usage_history),
            len(self.memory_usage_history),
            len(self.confidence_history),
            max(len(self.emotions[e]) for e in self.emotions)
        )
        
        # Convert deques to lists and pad if needed
        def pad_array(array, target_len, pad_value=0):
            array_list = list(array)
            if len(array_list) < target_len:
                return [pad_value] * (target_len - len(array_list)) + array_list
            return array_list
        
        # Build metrics object expected by frontend
        metrics = {
            'fps': pad_array(self.fps_history, max_len),
            'detectionCount': pad_array(self.detection_count_history, max_len),
            'cpuUsage': pad_array(self.cpu_usage_history, max_len),
            'memoryUsage': pad_array(self.memory_usage_history, max_len),
            'detectionConfidence': pad_array(self.confidence_history, max_len),
            'emotions': {
                'crying': self.current_emotions['crying'],
                'laughing': self.current_emotions['laughing'],
                'babbling': self.current_emotions['babbling'],
                'silence': self.current_emotions['silence']
            }
        }
        
        return {
            'metrics': metrics,
            'system_info': self.system_info
        }

def demo_metrics_generation(metrics_collector):
    """Generate demo metrics data for testing."""
    def generate_demo_data():
        while True:
            try:
                # Simulate frame processing time (0.05-0.1 seconds)
                frame_time = random.uniform(0.05, 0.1)
                
                # Simulate detections (0-2 people)
                num_detections = random.randint(0, 2)
                detections = []
                for i in range(num_detections):
                    detections.append({
                        'id': i,
                        'x': random.randint(0, 640),
                        'y': random.randint(0, 480),
                        'width': random.randint(50, 150),
                        'height': random.randint(100, 200),
                        'confidence': random.uniform(0.7, 0.99)
                    })
                
                # Update frame metrics
                metrics_collector.update_frame_metrics(frame_time, detections)
                
                # Simulate emotion detection (10% chance of crying)
                if random.random() < 0.1:
                    emotion_result = {
                        'crying': random.uniform(60, 90),
                        'laughing': random.uniform(0, 10),
                        'babbling': random.uniform(0, 10),
                        'silence': random.uniform(0, 20)
                    }
                else:
                    # Normal distribution
                    emotion_result = {
                        'crying': random.uniform(0, 10),
                        'laughing': random.uniform(0, 30),
                        'babbling': random.uniform(0, 30),
                        'silence': random.uniform(40, 90)
                    }
                
                # Normalize to 100%
                total = sum(emotion_result.values())
                for key in emotion_result:
                    emotion_result[key] = (emotion_result[key] / total) * 100
                
                # Update emotion metrics
                metrics_collector.update_emotion_metrics(emotion_result)
                
                # Sleep to simulate real-time processing
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error generating demo data: {e}")
    
    # Start demo generation in a separate thread
    demo_thread = threading.Thread(target=generate_demo_data)
    demo_thread.daemon = True
    demo_thread.start()
    
    return demo_thread

class MetricsDemoServer:
    """A demo server for testing the metrics dashboard."""
    
    def __init__(self, host='127.0.0.1', port=5000):
        """Initialize the demo server."""
        self.host = host
        self.port = port
        
        template_folder = os.path.join(PROJECT_ROOT, 'src', 'babymonitor', 'web', 'templates')
        static_folder = os.path.join(PROJECT_ROOT, 'src', 'babymonitor', 'web', 'static')
        
        # Check if template directory exists
        if not os.path.exists(template_folder):
            logger.warning(f"Template folder not found at {template_folder}")
            logger.warning("Checking alternative locations...")
            
            # Try to find templates elsewhere
            for root, dirs, files in os.walk(PROJECT_ROOT):
                if 'templates' in dirs and 'metrics.html' in os.listdir(os.path.join(root, 'templates')):
                    template_folder = os.path.join(root, 'templates')
                    logger.info(f"Found template folder at {template_folder}")
                    break
        
        # Create Flask app and SocketIO instance
        self.app = Flask(__name__, 
                         template_folder=template_folder,
                         static_folder=static_folder)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Create metrics collector
        self.metrics_collector = MetricsCollector()
        
        # Setup routes and socket events
        self._setup_routes()
        self._setup_socketio()
        
        # Start metrics emission thread
        self.running = False
        self.metrics_thread = None
        
        # Print startup info
        logger.info(f"Metrics demo server initialized on http://{self.host}:{self.port}")
        logger.info(f"Template folder: {self.app.template_folder}")
        logger.info(f"Static folder: {self.app.static_folder}")
        
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/metrics')
        def metrics():
            return render_template('metrics.html')
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for metrics."""
            # Get metrics data
            metrics_data = self.metrics_collector.get_metrics()
            return jsonify(metrics_data)
        
        @self.app.route('/api/system_info')
        def api_system_info():
            """API endpoint for system info."""
            return jsonify(self.metrics_collector.system_info)
    
    def _setup_socketio(self):
        """Setup Socket.IO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            
            # Send initial metrics data
            metrics_data = self.metrics_collector.get_metrics()
            self.socketio.emit('metrics_update', metrics_data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_metrics')
        def handle_request_metrics(data=None):
            """Handle client request for metrics data."""
            time_range = '1h'
            if data and 'timeRange' in data:
                time_range = data['timeRange']
            
            logger.info(f"Metrics requested with time range: {time_range}")
            
            # Get current metrics
            metrics_data = self.metrics_collector.get_metrics()
            
            # Emit metrics update
            self.socketio.emit('metrics_update', metrics_data)
    
    def _emit_metrics(self):
        """Periodically emit metrics updates."""
        while self.running:
            try:
                # Get current metrics
                metrics_data = self.metrics_collector.get_metrics()
                
                # Emit metrics update
                self.socketio.emit('metrics_update', metrics_data)
                
                # Randomly emit crying events (5% chance)
                if random.random() < 0.05:
                    self.socketio.emit('crying_detected', {
                        'timestamp': time.time() * 1000  # Use milliseconds
                    })
                    
                    # Also emit a general alert
                    self.socketio.emit('alert', {
                        'message': 'Crying detected!',
                        'level': 'warning',
                        'timestamp': time.time() * 1000
                    })
                
                # Sleep for 2 seconds
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error emitting metrics: {e}")
    
    def start(self):
        """Start the demo server."""
        try:
            # Start demo metrics generation
            self.demo_thread = demo_metrics_generation(self.metrics_collector)
            
            # Start metrics emission thread
            self.running = True
            self.metrics_thread = threading.Thread(target=self._emit_metrics)
            self.metrics_thread.daemon = True
            self.metrics_thread.start()
            
            # Start the Flask server
            logger.info(f"Starting Metrics Demo Server on http://{self.host}:{self.port}")
            self.socketio.run(self.app, host=self.host, port=self.port, debug=True, use_reloader=False)
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            raise

def main():
    """Main function to run the metrics demo server."""
    try:
        print("=" * 60)
        print("Baby Monitor System - Metrics Dashboard Demo")
        print("=" * 60)
        print("\nThis script runs a standalone Flask server with Socket.IO")
        print("that generates demo data for the metrics dashboard.\n")
        print(f"Project root: {PROJECT_ROOT}")
        
        # Create and start the demo server
        server = MetricsDemoServer()
        server.start()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main() 