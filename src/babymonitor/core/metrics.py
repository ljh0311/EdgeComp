import time
import psutil
from collections import deque
import logging
import numpy as np
import platform
import cv2
import os
import socket
import datetime

logger = logging.getLogger(__name__)

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
        self.history_size = history_size
        
        # Track total and peak detections
        self.total_detections = 0
        self.peak_detections = 0
        
        # System status tracking
        self.camera_status = "disconnected"
        self.person_detector_status = "initializing"
        self.emotion_detector_status = "initializing"
        
        # Alerts storage
        self.alerts = deque(maxlen=50)  # Store the last 50 alerts
        
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
            'os': platform.system() + ' ' + platform.release(),
            'python_version': platform.python_version(),
            'opencv_version': cv2.__version__,
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
            'detection_history_size': f"{history_size} frames",
            'camera_status': self.camera_status,
            'person_detector_status': self.person_detector_status,
            'emotion_detector_status': self.emotion_detector_status,
            'hostname': socket.gethostname(),
            'cpu_usage': 0,
            'memory_usage': 0
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
    
    def set_camera_status(self, status):
        """Update camera status."""
        self.camera_status = status
        self.system_info['camera_status'] = status
    
    def set_person_detector_status(self, status):
        """Update person detector status."""
        self.person_detector_status = status
        self.system_info['person_detector_status'] = status
    
    def set_emotion_detector_status(self, status):
        """Update emotion detector status."""
        self.emotion_detector_status = status
        self.system_info['emotion_detector_status'] = status
    
    def add_alert(self, message, level="info"):
        """Add an alert to the alerts history."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.alerts.append({
            "timestamp": timestamp,
            "message": message,
            "level": level
        })
        logger.info(f"Alert: [{level}] {message}")
    
    def update_frame_metrics(self, frame_time, detection_results=None):
        """Update metrics with new frame data."""
        try:
            current_time = time.time()
            
            # Update frame count
            self.frame_count += 1
            
            # Calculate FPS
            fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_history.append(round(fps, 2))
            
            # Update detection count
            detection_count = len(detection_results) if detection_results else 0
            self.detection_count_history.append(detection_count)
            
            # Update total and peak detections
            self.total_detections += detection_count
            self.peak_detections = max(self.peak_detections, detection_count)
            
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
            
            # Update system info values
            self.system_info['cpu_usage'] = round(psutil.cpu_percent(), 2)
            self.system_info['memory_usage'] = round(psutil.virtual_memory().percent, 2)
            
            # Update uptime
            uptime_seconds = current_time - self.start_time
            hours, remainder = divmod(uptime_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
            self.system_info['uptime'] = uptime_str
            
        except Exception as e:
            logger.error(f"Error updating frame metrics: {e}")
    
    def update_emotion_metrics(self, emotion_result=None):
        """Update emotion metrics based on detection results."""
        try:
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
            
            # Generate crying alert if crying percentage is high
            if self.current_emotions['crying'] > 70:
                self.add_alert("High crying probability detected!", "warning")
            
        except Exception as e:
            logger.error(f"Error updating emotion metrics: {e}")
    
    def get_metrics(self, time_range='1h'):
        """Get current metrics data based on time range."""
        try:
            # Determine subset of data based on time range
            # Note: This is a simplified version; in a real app, you'd need 
            # to store timestamps with metrics to properly filter by time range
            if time_range == '24h':
                subset_size = min(self.history_size, 20)
            elif time_range == '3h':
                subset_size = min(self.history_size, 15)
            else:  # 1h default
                subset_size = min(self.history_size, 10)
            
            # Ensure all arrays are the same length (pad if necessary)
            def get_recent_data(data_deque, count):
                """Get the most recent count items from a deque."""
                items = list(data_deque)
                return items[-count:] if len(items) >= count else items
            
            # Get recent data for each metric
            subset_fps = get_recent_data(self.fps_history, subset_size)
            subset_detections = get_recent_data(self.detection_count_history, subset_size)
            subset_cpu = get_recent_data(self.cpu_usage_history, subset_size)
            subset_memory = get_recent_data(self.memory_usage_history, subset_size)
            subset_confidence = get_recent_data(self.confidence_history, subset_size)
            
            # Build metrics object expected by frontend
            metrics = {
                'fps': subset_fps,
                'detectionCount': subset_detections,
                'cpuUsage': subset_cpu,
                'memoryUsage': subset_memory,
                'detectionConfidence': subset_confidence,
                'emotions': self.current_emotions,
                'total_detections': self.total_detections,
                'peak_detections': self.peak_detections
            }
            
            # Calculate detection confidence average if available
            if subset_confidence and len(subset_confidence) > 0:
                metrics['detection_confidence_avg'] = sum(subset_confidence) / len(subset_confidence)
                
            # Get the most recent values for current metrics
            current = {
                'fps': subset_fps[-1] if subset_fps else 0,
                'detections': subset_detections[-1] if subset_detections else 0,
                'cpu': subset_cpu[-1] if subset_cpu else 0,
                'memory': subset_memory[-1] if subset_memory else 0
            }
            
            return {
                'metrics': metrics,
                'system_info': self.system_info,
                'current': current,
                'alerts': list(self.alerts)
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            # Return a minimal valid response on error
            return {
                'metrics': {
                    'fps': list(self.fps_history),
                    'detectionCount': list(self.detection_count_history),
                    'cpuUsage': list(self.cpu_usage_history),
                    'memoryUsage': list(self.memory_usage_history),
                    'detectionConfidence': list(self.confidence_history),
                    'emotions': self.current_emotions
                },
                'system_info': self.system_info,
                'current': {
                    'fps': 0,
                    'detections': 0,
                    'cpu': 0,
                    'memory': 0
                },
                'alerts': list(self.alerts)
            }
    
    def get_alerts(self, count=10):
        """Get the most recent alerts."""
        return list(self.alerts)[:count]

def demo_metrics_generation(metrics_collector):
    """Generate demo metrics data for testing."""
    import random
    import time
    from threading import Thread
    
    def generate_demo_data():
        while True:
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
                # Add crying alert
                metrics_collector.add_alert("Crying detected with high confidence!", "warning")
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
            
            # Randomly update system status
            if random.random() < 0.01:  # 1% chance to change status
                statuses = ["connected", "disconnected", "initializing", "running", "error"]
                metrics_collector.set_camera_status(random.choice(statuses))
                metrics_collector.set_person_detector_status(random.choice(statuses))
                metrics_collector.set_emotion_detector_status(random.choice(statuses))
                
                # Add system status alert
                metrics_collector.add_alert(f"System status changed: Camera: {metrics_collector.camera_status}", "info")
            
            # Sleep to simulate real-time processing
            time.sleep(0.5)
    
    # Start demo generation in a separate thread
    demo_thread = Thread(target=generate_demo_data)
    demo_thread.daemon = True
    demo_thread.start()
    
    return demo_thread 