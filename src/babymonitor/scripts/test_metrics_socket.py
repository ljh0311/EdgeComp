#!/usr/bin/env python
"""
Test script for metrics Socket.IO communication.
This script creates a simple Flask server with Socket.IO that emits test metrics data
to help diagnose issues with the metrics page.
"""

import os
import sys
import time
import random
import logging
import argparse
from threading import Thread
from datetime import datetime

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('metrics_test')

class MetricsTestServer:
    def __init__(self, host='0.0.0.0', port=5000, debug=False):
        self.host = host
        self.port = port
        self.debug = debug
        
        # Create Flask app
        self.app = Flask(__name__, 
                         template_folder='../web/templates',
                         static_folder='../web/static')
        
        # Configure app
        self.app.config['SECRET_KEY'] = 'test-metrics-secret-key'
        self.app.config['DEBUG'] = debug
        
        # Initialize Socket.IO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Setup routes
        self.setup_routes()
        
        # Setup Socket.IO events
        self.setup_socketio_events()
        
        # Test metrics data thread
        self.test_thread = None
        self.running = False
        
        logger.info(f"MetricsTestServer initialized with host={host}, port={port}")
    
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/metrics')
        def metrics():
            return render_template('metrics.html')
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for metrics data"""
            return jsonify(self.generate_test_metrics())
        
        @self.app.route('/api/system_info')
        def api_system_info():
            """API endpoint for system information"""
            return jsonify(self.generate_test_system_info())
    
    def setup_socketio_events(self):
        """Setup Socket.IO event handlers"""
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {self.socketio.request.sid}")
            # Send initial metrics data
            self.socketio.emit('metrics_update', self.generate_test_metrics(), room=self.socketio.request.sid)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {self.socketio.request.sid}")
        
        @self.socketio.on('request_metrics')
        def handle_request_metrics(data=None):
            logger.info(f"Metrics requested: {data}")
            # Send metrics data
            metrics_data = self.generate_test_metrics()
            self.socketio.emit('metrics_update', metrics_data, room=self.socketio.request.sid)
            logger.info("Metrics data sent")
        
        @self.socketio.on('start_demo')
        def handle_start_demo():
            logger.info("Demo mode started")
        
        @self.socketio.on('stop_demo')
        def handle_stop_demo():
            logger.info("Demo mode stopped")
    
    def generate_test_metrics(self):
        """Generate test metrics data"""
        # Generate random metrics data
        fps_data = [random.uniform(15, 30) for _ in range(20)]
        detection_data = [random.randint(0, 3) for _ in range(20)]
        cpu_data = [random.uniform(10, 80) for _ in range(20)]
        memory_data = [random.uniform(20, 70) for _ in range(20)]
        confidence_data = [random.uniform(70, 95) for _ in range(20)]
        
        # Generate emotion distribution
        crying = random.uniform(0, 30)
        laughing = random.uniform(0, 30)
        babbling = random.uniform(0, 30)
        total = crying + laughing + babbling
        silence = 100 - total if total < 100 else 0
        
        # Ensure they add up to 100%
        if total > 100:
            scale = 100 / total
            crying *= scale
            laughing *= scale
            babbling *= scale
        
        emotions = {
            'crying': crying,
            'laughing': laughing,
            'babbling': babbling,
            'silence': silence
        }
        
        # Create metrics data structure
        metrics_data = {
            'metrics': {
                'fps': fps_data,
                'detectionCount': detection_data,
                'cpuUsage': cpu_data,
                'memoryUsage': memory_data,
                'detectionConfidence': confidence_data,
                'emotions': emotions
            },
            'system_info': self.generate_test_system_info(),
            'current': {
                'fps': fps_data[-1],
                'detections': detection_data[-1],
                'cpu': cpu_data[-1],
                'memory': memory_data[-1]
            }
        }
        
        return metrics_data
    
    def generate_test_system_info(self):
        """Generate test system information"""
        import platform
        
        # Calculate uptime
        current_time = time.time()
        uptime_seconds = current_time % 86400  # Random uptime within a day
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        
        # Random status options
        statuses = ['running', 'connected', 'initializing', 'error']
        camera_status = random.choice(statuses)
        detector_status = 'running' if random.random() < 0.9 else 'error'
        
        system_info = {
            'os': platform.system() + ' ' + platform.release(),
            'python_version': platform.python_version(),
            'opencv_version': '4.5.4',
            'uptime': uptime_str,
            'person_detector': 'YOLOv8',
            'detector_model': 'YOLOv8n',
            'emotion_detector': 'Active',
            'detection_threshold': '0.7',
            'camera_resolution': '640x480',
            'audio_sample_rate': '16000 Hz',
            'frame_skip': '2',
            'process_resolution': '640x480',
            'confidence_threshold': '0.7',
            'detection_history_size': '20 frames',
            'camera_status': camera_status,
            'person_detector_status': detector_status,
            'emotion_detector_status': 'running'
        }
        
        return system_info
    
    def emit_test_data(self):
        """Emit test metrics data periodically"""
        logger.info("Starting test data emission thread")
        self.running = True
        
        counter = 0
        while self.running:
            metrics_data = self.generate_test_metrics()
            self.socketio.emit('metrics_update', metrics_data)
            
            # Occasionally emit detection events
            if counter % 5 == 0:
                detection_event = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'count': random.randint(1, 3)
                }
                self.socketio.emit('detection_event', detection_event)
                logger.info(f"Emitted detection event: {detection_event}")
            
            counter += 1
            logger.info(f"Emitted test metrics data (#{counter})")
            time.sleep(3)  # Emit every 3 seconds
    
    def start_test_thread(self):
        """Start a thread to emit test metrics data"""
        if self.test_thread is None or not self.test_thread.is_alive():
            self.test_thread = Thread(target=self.emit_test_data)
            self.test_thread.daemon = True
            self.test_thread.start()
            logger.info("Test data thread started")
    
    def run(self):
        """Run the test server"""
        logger.info(f"Starting metrics test server on {self.host}:{self.port}")
        self.start_test_thread()
        self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug, allow_unsafe_werkzeug=True)

def main():
    parser = argparse.ArgumentParser(description='Metrics Socket.IO Test Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    server = MetricsTestServer(host=args.host, port=args.port, debug=args.debug)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        server.running = False
        if server.test_thread and server.test_thread.is_alive():
            server.test_thread.join(timeout=1)
        logger.info("Server shutdown complete")

if __name__ == '__main__':
    main() 