import time
import psutil
from collections import deque
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, history_size=60):
        self.history_size = history_size
        self.fps_history = deque(maxlen=history_size)
        self.frame_time_history = deque(maxlen=history_size)
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.detection_count_history = deque(maxlen=history_size)
        self.detection_types_history = deque(maxlen=history_size)
        
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_update_interval = 1.0  # Update FPS every second
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
        # Add smoothing
        self.smoothing_window = 5
        self.last_cpu = 0
        self.last_memory = 0
        self.last_metrics_update = 0
        self.min_update_interval = 0.5  # Minimum time between updates in seconds
        
        # Detection metrics
        self.current_detection_count = 0
        self.current_detection_types = {}
    
    def update_frame_metrics(self, frame_time, detection_results=None):
        """Update frame processing metrics.
        
        Args:
            frame_time: Time taken to process the frame in seconds
            detection_results: Optional detection results from the detector
        """
        try:
            current_time = time.time()
            
            # Update frame time
            self.frame_time_history.append(frame_time * 1000)  # Convert to ms
            
            # Update frame count and FPS
            self.frame_count += 1
            if current_time - self.last_fps_update >= self.fps_update_interval:
                self.current_fps = self.frame_count / (current_time - self.last_fps_update)
                self.fps_history.append(self.current_fps)
                self.frame_count = 0
                self.last_fps_update = current_time
            
            # Update detection metrics if provided
            if detection_results is not None:
                detections = detection_results.get('detections', [])
                self.current_detection_count = len(detections)
                self.detection_count_history.append(self.current_detection_count)
                
                # Count detection types
                detection_types = {}
                for detection in detections:
                    detection_class = detection.get('class', 'unknown')
                    if detection_class in detection_types:
                        detection_types[detection_class] += 1
                    else:
                        detection_types[detection_class] = 1
                
                self.current_detection_types = detection_types
                self.detection_types_history.append(detection_types)
                
        except Exception as e:
            logger.error(f"Error updating frame metrics: {e}")
    
    def update_system_metrics(self):
        try:
            current_time = time.time()
            
            # Only update if enough time has passed
            if current_time - self.last_metrics_update < self.min_update_interval:
                return
                
            # Get CPU usage with smoothing
            cpu_percent = psutil.cpu_percent(interval=None)
            if cpu_percent is not None and cpu_percent >= 0:
                # Apply exponential smoothing
                alpha = 0.3  # Smoothing factor
                self.last_cpu = alpha * cpu_percent + (1 - alpha) * self.last_cpu
                self.cpu_history.append(self.last_cpu)
            
            # Get memory usage with smoothing
            memory = psutil.virtual_memory()
            if memory is not None and memory.percent >= 0:
                # Apply exponential smoothing
                self.last_memory = alpha * memory.percent + (1 - alpha) * self.last_memory
                self.memory_history.append(self.last_memory)
            
            self.last_metrics_update = current_time
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def get_metrics(self):
        try:
            # Calculate smoothed values using a moving average
            def smooth_data(data, window=5):
                if not data:
                    return 0.0
                window = min(window, len(data))
                return float(np.mean(list(data)[-window:]))
            
            # Get the latest detection types
            detection_types = {}
            if self.detection_types_history:
                # Combine the last few detection type counts
                for types_dict in list(self.detection_types_history)[-self.smoothing_window:]:
                    for detection_type, count in types_dict.items():
                        if detection_type in detection_types:
                            detection_types[detection_type] += count
                        else:
                            detection_types[detection_type] = count
                
                # Average the counts
                for detection_type in detection_types:
                    detection_types[detection_type] /= min(self.smoothing_window, len(self.detection_types_history))
            
            current = {
                'fps': smooth_data(self.fps_history, self.smoothing_window),
                'frame_time': smooth_data(self.frame_time_history, self.smoothing_window),
                'cpu_usage': self.last_cpu,
                'memory_usage': self.last_memory,
                'detection_count': smooth_data(self.detection_count_history, self.smoothing_window),
                'detection_types': detection_types
            }
            
            history = {
                'fps': list(self.fps_history),
                'frame_time': list(self.frame_time_history),
                'cpu_usage': list(self.cpu_history),
                'memory_usage': list(self.memory_history),
                'detection_count': list(self.detection_count_history)
            }
            
            return {
                'current': current,
                'history': history
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {
                'current': {
                    'fps': self.current_fps,
                    'frame_time': self.frame_time_history[-1] if self.frame_time_history else 0.0,
                    'cpu_usage': self.last_cpu,
                    'memory_usage': self.last_memory,
                    'detection_count': self.current_detection_count,
                    'detection_types': self.current_detection_types
                },
                'history': {
                    'fps': list(self.fps_history),
                    'frame_time': list(self.frame_time_history),
                    'cpu_usage': list(self.cpu_history),
                    'memory_usage': list(self.memory_history),
                    'detection_count': list(self.detection_count_history)
                }
            } 