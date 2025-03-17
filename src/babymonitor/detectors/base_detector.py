"""
Base Detector Module
==================
Base class for detector implementations.
"""

import time
import logging
import numpy as np
import cv2
import psutil
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

class BaseDetector(ABC):
    """Base class for all detector implementations."""
    
    def __init__(self, threshold: float = 0.5):
        """Initialize the base detector.
        
        Args:
            threshold: Detection confidence threshold
        """
        self.logger = logging.getLogger(__name__)
        self.threshold = threshold
        self.frame_count = 0
        self.fps = 0.0
        self.frame_times = []
        self.max_frame_history = 30
        self.last_time = time.time()
        self.last_frame = None
        self.last_detections = None
        
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Dict containing processed frame and detection results
        """
        pass
    
    def get_fps(self) -> float:
        """Get the current FPS."""
        if not self.frame_times:
            return 0.0
        
        # Avoid division by zero
        total_time = sum(self.frame_times)
        if total_time <= 0:
            return 0.0
            
        return len(self.frame_times) / total_time
    
    def update_fps(self, frame_time: float):
        """Update FPS calculation with new frame time."""
        # Ensure frame_time is positive to avoid division by zero
        if frame_time <= 0:
            frame_time = 0.001  # Use a small positive value instead
            
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        self.fps = self.get_fps()
    
    def get_processing_time(self) -> float:
        """Get the average processing time per frame.
        
        Returns:
            Average processing time in seconds
        """
        if not self.frame_times:
            return 0.0
        return sum(self.frame_times) / len(self.frame_times)
    
    def get_cpu_usage(self) -> float:
        """Get the current CPU usage.
        
        Returns:
            CPU usage percentage
        """
        return psutil.cpu_percent()
    
    def get_memory_usage(self) -> float:
        """Get the current memory usage.
        
        Returns:
            Memory usage in MB
        """
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    def get_device_info(self) -> Dict:
        """Get information about the device.
        
        Returns:
            Dict containing device information
        """
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024 * 1024),  # MB
            'memory_available': psutil.virtual_memory().available / (1024 * 1024),  # MB
        }
    
    def cleanup(self):
        """Clean up resources."""
        pass 