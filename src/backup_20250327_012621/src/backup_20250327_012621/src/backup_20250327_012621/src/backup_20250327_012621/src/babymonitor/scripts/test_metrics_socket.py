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

from flask import Flask, render_template, jsonify, request
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
        
        # Current camera resolution
        self.camera_resolution = "640x480"
        self.camera_device = "Default Camera (Webcam)"
        self.microphone_device = "Default Microphone"
        
        # Current emotion model
        self.current_model_id = 'emotion_model'
        
        # Initialize camera manager
        from babymonitor.camera.camera_manager import CameraManager
        self.camera_manager = CameraManager()
        
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
        
        @self.app.route('/repair')
        def repair():
            return render_template('repair_tools.html')
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for metrics data"""
            return jsonify(self.generate_test_metrics())
        
        @self.app.route('/api/system_info')
        def api_system_info():
            """API endpoint for system information"""
            return jsonify(self.generate_test_system_info())
        
        @self.app.route('/api/camera/list')
        def api_camera_list():
            """API endpoint to list all cameras"""
            try:
                cameras = self.camera_manager.get_camera_list()
                return jsonify({
                    'status': 'success',
                    'cameras': cameras
                })
            except Exception as e:
                logger.error(f"Error listing cameras: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/camera/add', methods=['POST'])
        def api_camera_add():
            """API endpoint to add a new camera"""
            try:
                data = request.get_json()
                if not data or 'name' not in data or 'source' not in data:
                    return jsonify({'error': 'Missing required parameters'}), 400
                
                name = data['name']
                source = data['source']
                url = data.get('url')
                
                # Convert source to int for USB cameras
                if source != 'ip':
                    source = int(source)
                else:
                    source = url
                
                success = self.camera_manager.add_camera(name, source)
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': f'Camera "{name}" added successfully'
                    })
                else:
                    return jsonify({'error': 'Failed to add camera'}), 500
                
            except Exception as e:
                logger.error(f"Error adding camera: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/camera/remove', methods=['POST'])
        def api_camera_remove():
            """API endpoint to remove a camera"""
            try:
                data = request.get_json()
                if not data or 'name' not in data:
                    return jsonify({'error': 'Missing camera name'}), 400
                
                name = data['name']
                success = self.camera_manager.remove_camera(name)
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': f'Camera "{name}" removed successfully'
                    })
                else:
                    return jsonify({'error': 'Failed to remove camera'}), 500
                
            except Exception as e:
                logger.error(f"Error removing camera: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/camera/restart', methods=['POST'])
        def api_camera_restart():
            """API endpoint to restart a camera"""
            try:
                data = request.get_json()
                if not data or 'name' not in data:
                    return jsonify({'error': 'Missing camera name'}), 400
                
                name = data['name']
                success = self.camera_manager.restart_camera(name)
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': f'Camera "{name}" restarted successfully'
                    })
                else:
                    return jsonify({'error': 'Failed to restart camera'}), 500
                
            except Exception as e:
                logger.error(f"Error restarting camera: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system/check')
        def api_system_check():
            """API endpoint to check system status"""
            try:
                # Get camera status
                cameras = self.camera_manager.get_camera_list()
                camera_status = "ok" if any(c['status'] == 'Active' for c in cameras) else "error"
                
                # Get audio status
                audio_status = random.choice(["ok", "warning", "error"])
                
                # Get detection status
                detection_status = random.choice(["ok", "warning"])
                
                # Get emotion status
                emotion_status = "ok" if self.current_model_id else "error"
                
                return jsonify({
                    'camera': {
                        'status': camera_status,
                        'message': f'Found {len(cameras)} camera(s)'
                    },
                    'audio': {
                        'status': audio_status,
                        'message': 'Audio system running'
                    },
                    'detection': {
                        'status': detection_status,
                        'message': 'Person detection active'
                    },
                    'emotion': {
                        'status': emotion_status,
                        'message': f'Using model: {self.current_model_id}'
                    }
                })
            except Exception as e:
                logger.error(f"Error checking system: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system/restart', methods=['POST'])
        def api_system_restart():
            """API endpoint to restart the system"""
            try:
                # Clean up resources
                self.camera_manager.cleanup()
                
                # Simulate system restart
                time.sleep(2)
                
                # Reinitialize camera manager
                self.camera_manager = CameraManager()
                
                return jsonify({
                    'status': 'success',
                    'message': 'System restarted successfully'
                })
            except Exception as e:
                logger.error(f"Error restarting system: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system/stop', methods=['POST'])
        def api_system_stop():
            """API endpoint to stop the system"""
            try:
                # Clean up resources
                self.camera_manager.cleanup()
                
                return jsonify({
                    'status': 'success',
                    'message': 'System stopped successfully'
                })
            except Exception as e:
                logger.error(f"Error stopping system: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system/memory')
        def api_system_memory():
            """API endpoint for system memory information"""
            return jsonify({
                'total_memory': 8192,  # 8GB total
                'available_memory': 4096,  # 4GB available
                'current_resolution': self.camera_resolution
            })
        
        @self.app.route('/api/system/info')
        def api_system_info_alternate():
            """Alternative API endpoint for system information"""
            try:
                # Get camera status
                cameras = self.camera_manager.get_camera_list()
                camera_status = "ok" if any(c['status'] == 'Active' for c in cameras) else "error"
                
                # Get current model info
                from babymonitor.detectors.emotion_detector import EmotionDetector
                model_info = EmotionDetector.AVAILABLE_MODELS.get(self.current_model_id, {})
                
                return jsonify({
                    'cpu_usage': random.uniform(10, 80),
                    'memory_usage': random.uniform(20, 70),
                    'memory_available': 4096,
                    'disk_usage': random.uniform(30, 90),
                    'camera_status': camera_status,
                    'audio_status': random.choice(['running', 'connected', 'error']),
                    'uptime': self._get_random_uptime(),
                    'system_info': {
                        'os': 'Windows 10',
                        'python_version': '3.8.10',
                        'opencv_version': '4.5.4',
                        'camera_resolution': self.camera_resolution,
                        'camera_device': self.camera_device,
                        'microphone_device': self.microphone_device,
                        'audio_sample_rate': '16000 Hz'
                    },
                    'model_info': {
                        'id': self.current_model_id,
                        'name': model_info.get('name', 'Unknown'),
                        'type': model_info.get('type', 'Unknown'),
                        'emotions': model_info.get('emotions', [])
                    }
                })
            except Exception as e:
                logger.error(f"Error getting system info: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/camera/configure', methods=['POST'])
        def api_configure_camera():
            """API endpoint to configure camera resolution"""
            try:
                data = request.get_json()
                if not data or 'width' not in data or 'height' not in data:
                    return jsonify({'error': 'Missing width/height parameters'}), 400
                
                width = data['width']
                height = data['height']
                
                # Set new resolution
                self.camera_resolution = f"{width}x{height}"
                logger.info(f"Camera resolution set to {self.camera_resolution}")
                
                # Broadcast the resolution update to all clients
                self.socketio.emit('system_info', {
                    'camera_resolution': self.camera_resolution,
                    'camera_device': self.camera_device,
                    'microphone_device': self.microphone_device
                })
                
                return jsonify({
                    'status': 'success',
                    'message': f'Camera resolution set to {self.camera_resolution}'
                })
            except Exception as e:
                logger.error(f"Error setting camera resolution: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/repair/run', methods=['POST'])
        def api_repair_run():
            """API endpoint for running repair tools"""
            try:
                data = request.get_json()
                tool = data.get('tool', '')
                
                logger.info(f"Running repair tool: {tool}")
                
                # Simulate repair operation
                time.sleep(1)
                
                return jsonify({
                    'status': 'success',
                    'message': f'Successfully ran repair tool: {tool}'
                })
            except Exception as e:
                logger.error(f"Error running repair tool: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/emotion/models')
        def api_emotion_models():
            """API endpoint for available emotion models"""
            from babymonitor.detectors.emotion_detector import EmotionDetector
            return jsonify(EmotionDetector.AVAILABLE_MODELS)
        
        @self.app.route('/api/emotion/apply_model', methods=['POST'])
        def api_apply_model():
            """API endpoint to switch emotion models"""
            try:
                data = request.get_json()
                if not data or 'model_id' not in data:
                    return jsonify({'error': 'Missing model_id parameter'}), 400
                
                model_id = data['model_id']
                
                # Update current model
                self.current_model_id = model_id
                logger.info(f"Switching to emotion model: {model_id}")
                
                # Get model info
                from babymonitor.detectors.emotion_detector import EmotionDetector
                model_info = EmotionDetector.AVAILABLE_MODELS.get(model_id)
                if not model_info:
                    return jsonify({'error': f'Unknown model ID: {model_id}'}), 400
                
                # Broadcast model change to all clients
                self.socketio.emit('emotion_model_changed', {
                    'model_id': model_id,
                    'name': model_info['name'],
                    'type': model_info['type'],
                    'emotions': model_info['emotions']
                })
                
                return jsonify({
                    'status': 'success',
                    'message': f'Switched to model: {model_info["name"]}',
                    'emotions': {emotion: 0.0 for emotion in model_info['emotions']}
                })
                
            except Exception as e:
                logger.error(f"Error applying emotion model: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/emotion/model_info')
        def api_model_info():
            """API endpoint for current model information"""
            from babymonitor.detectors.emotion_detector import EmotionDetector
            model_info = EmotionDetector.AVAILABLE_MODELS.get(self.current_model_id)
            if not model_info:
                return jsonify({'error': 'Current model not found'}), 404
                
            return jsonify({
                'model_id': self.current_model_id,
                'name': model_info['name'],
                'type': model_info['type'],
                'emotions': model_info['emotions']
            })
        
        @self.app.route('/api/audio/test', methods=['POST'])
        def api_audio_test():
            """API endpoint to test audio system"""
            try:
                # Simulate audio test
                time.sleep(1)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Audio test completed successfully'
                })
            except Exception as e:
                logger.error(f"Error testing audio: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/audio/restart', methods=['POST'])
        def api_audio_restart():
            """API endpoint to restart audio system"""
            try:
                # Simulate audio system restart
                time.sleep(2)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Audio system restarted successfully'
                })
            except Exception as e:
                logger.error(f"Error restarting audio: {str(e)}")
                return jsonify({'error': str(e)}), 500
    
    def _get_random_uptime(self):
        """Generate a random uptime string"""
        hours = random.randint(0, 23)
        minutes = random.randint(0, 59)
        seconds = random.randint(0, 59)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def setup_socketio_events(self):
        """Setup Socket.IO event handlers"""
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info(f"Client connected: {request.sid}")
            
            # Send initial camera list
            cameras = self.camera_manager.get_camera_list()
            self.socketio.emit('camera_list', {
                'cameras': cameras
            })
            
            # Send initial system status
            self.emit_system_status()
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_camera_list')
        def handle_request_camera_list():
            """Handle request for camera list"""
            cameras = self.camera_manager.get_camera_list()
            self.socketio.emit('camera_list', {
                'cameras': cameras
            })
        
        @self.socketio.on('request_system_status')
        def handle_request_system_status():
            """Handle request for system status"""
            self.emit_system_status()
    
    def emit_system_status(self):
        """Emit current system status to all clients"""
        try:
            # Get camera status
            cameras = self.camera_manager.get_camera_list()
            camera_status = "ok" if any(c['status'] == 'Active' for c in cameras) else "error"
            
            # Get audio status
            audio_status = random.choice(["ok", "warning", "error"])
            
            # Get detection status
            detection_status = random.choice(["ok", "warning"])
            
            # Get emotion status
            emotion_status = "ok" if self.current_model_id else "error"
            
            self.socketio.emit('system_status', {
                'camera': {
                    'status': camera_status,
                    'message': f'Found {len(cameras)} camera(s)'
                },
                'audio': {
                    'status': audio_status,
                    'message': 'Audio system running'
                },
                'detection': {
                    'status': detection_status,
                    'message': 'Person detection active'
                },
                'emotion': {
                    'status': emotion_status,
                    'message': f'Using model: {self.current_model_id}'
                }
            })
        except Exception as e:
            logger.error(f"Error emitting system status: {str(e)}")
    
    def generate_test_metrics(self):
        """Generate test metrics data"""
        # Generate random metrics data
        fps_data = [random.uniform(15, 30) for _ in range(20)]
        detection_data = [random.randint(0, 3) for _ in range(20)]
        cpu_data = [random.uniform(10, 80) for _ in range(20)]
        memory_data = [random.uniform(20, 70) for _ in range(20)]
        confidence_data = [random.uniform(70, 95) for _ in range(20)]
        
        # Get current model's emotions
        from babymonitor.detectors.emotion_detector import EmotionDetector
        model_info = EmotionDetector.AVAILABLE_MODELS.get(self.current_model_id)
        available_emotions = model_info['emotions'] if model_info else EmotionDetector.DEFAULT_EMOTIONS
        
        # Generate emotion distribution based on current model
        emotion_values = []
        for _ in range(len(available_emotions)):
            emotion_values.append(random.uniform(0, 30))
        
        total = sum(emotion_values)
        if total > 100:
            scale = 100 / total
            emotion_values = [v * scale for v in emotion_values]
        elif total < 100:
            # Add remaining probability to a random emotion
            remaining = 100 - total
            emotion_values[random.randint(0, len(emotion_values)-1)] += remaining
        
        emotions = dict(zip(available_emotions, emotion_values))
        
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
            'camera_resolution': self.camera_resolution,
            'camera_device': self.camera_device,
            'microphone_device': self.microphone_device,
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
            
            # Also emit system_info periodically to update camera resolution
            if counter % 3 == 0:
                system_info = self.generate_test_system_info()
                self.socketio.emit('system_info', system_info)
            
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