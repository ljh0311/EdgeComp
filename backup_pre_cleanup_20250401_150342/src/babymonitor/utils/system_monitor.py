"""
System Performance Monitor
======================
Monitors system resources and performance metrics in real-time.
"""

import psutil
import time
import threading
import logging
import platform
import torch
import numpy as np
from typing import Dict, Optional
from pathlib import Path

class SystemMonitor:
    """Real-time system resource and performance monitor."""
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize the system monitor.
        
        Args:
            update_interval: How often to update metrics (in seconds)
        """
        self.logger = logging.getLogger(__name__)
        self.update_interval = update_interval
        self.is_running = False
        self.metrics = {}
        self.metrics_lock = threading.Lock()
        self._monitor_thread = None
        
        # Initialize system info
        self.system_info = self._get_system_info()
        self.logger.info(f"System info: {self.system_info}")
    
    def _get_system_info(self) -> Dict:
        """Get static system information."""
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'total_memory': psutil.virtual_memory().total / (1024 * 1024),  # MB
            'gpu_available': torch.cuda.is_available(),
            'python_version': platform.python_version()
        }
        
        # Check if running on Raspberry Pi
        try:
            with open('/proc/cpuinfo', 'r') as f:
                if 'Raspberry Pi' in f.read():
                    info['is_raspberry_pi'] = True
                    # Try to determine Pi model
                    if 'BCM2711' in platform.release():
                        info['pi_model'] = '4'
                    elif 'BCM2837' in platform.release():
                        info['pi_model'] = '3'
                    else:
                        info['pi_model'] = 'unknown'
                else:
                    info['is_raspberry_pi'] = False
        except:
            info['is_raspberry_pi'] = False
        
        if info['gpu_available']:
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
        
        return info
    
    def _update_metrics(self):
        """Update system metrics."""
        try:
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=None),
                'per_cpu_percent': psutil.cpu_percent(interval=None, percpu=True),
                'memory_used': psutil.virtual_memory().used / (1024 * 1024),  # MB
                'memory_percent': psutil.virtual_memory().percent
            }
            
            # Add GPU metrics if available
            if self.system_info['gpu_available']:
                metrics.update({
                    'gpu_memory_used': torch.cuda.memory_allocated(0) / (1024 * 1024),  # MB
                    'gpu_memory_cached': torch.cuda.memory_reserved(0) / (1024 * 1024)  # MB
                })
            
            # Add disk metrics
            disk = psutil.disk_usage('/')
            metrics.update({
                'disk_used': disk.used / (1024 * 1024 * 1024),  # GB
                'disk_free': disk.free / (1024 * 1024 * 1024),  # GB
                'disk_percent': disk.percent
            })
            
            # Calculate performance metrics
            if hasattr(self, 'prev_metrics'):
                dt = metrics['timestamp'] - self.prev_metrics['timestamp']
                if dt > 0:
                    metrics['cpu_usage_change'] = (metrics['cpu_percent'] - self.prev_metrics['cpu_percent']) / dt
                    metrics['memory_usage_change'] = (metrics['memory_used'] - self.prev_metrics['memory_used']) / dt
            
            self.prev_metrics = metrics.copy()
            
            # Update metrics thread-safely
            with self.metrics_lock:
                self.metrics = metrics
                
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            self._update_metrics()
            time.sleep(self.update_interval)
    
    def start(self):
        """Start the system monitor."""
        if not self.is_running:
            self.is_running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            self.logger.info("System monitor started")
    
    def stop(self):
        """Stop the system monitor."""
        self.is_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self.update_interval * 2)
            self.logger.info("System monitor stopped")
    
    def get_metrics(self) -> Dict:
        """Get current system metrics thread-safely."""
        with self.metrics_lock:
            return self.metrics.copy()
    
    def get_resource_usage(self) -> Dict:
        """Get current resource usage with warnings."""
        metrics = self.get_metrics()
        
        # Define warning thresholds
        thresholds = {
            'cpu_critical': 90,
            'cpu_warning': 75,
            'memory_critical': 90,
            'memory_warning': 75,
            'disk_critical': 90,
            'disk_warning': 75
        }
        
        warnings = []
        if metrics.get('cpu_percent', 0) > thresholds['cpu_critical']:
            warnings.append('CRITICAL: CPU usage extremely high')
        elif metrics.get('cpu_percent', 0) > thresholds['cpu_warning']:
            warnings.append('WARNING: CPU usage high')
            
        if metrics.get('memory_percent', 0) > thresholds['memory_critical']:
            warnings.append('CRITICAL: Memory usage extremely high')
        elif metrics.get('memory_percent', 0) > thresholds['memory_warning']:
            warnings.append('WARNING: Memory usage high')
            
        if metrics.get('disk_percent', 0) > thresholds['disk_critical']:
            warnings.append('CRITICAL: Disk usage extremely high')
        elif metrics.get('disk_percent', 0) > thresholds['disk_warning']:
            warnings.append('WARNING: Disk usage high')
        
        return {
            'metrics': metrics,
            'warnings': warnings,
            'status': 'critical' if any('CRITICAL' in w for w in warnings) else 
                     'warning' if warnings else 'normal'
        }
    
    def log_metrics(self):
        """Log current metrics."""
        metrics = self.get_metrics()
        self.logger.info(f"System Metrics:")
        self.logger.info(f"  CPU Usage: {metrics.get('cpu_percent', 0):.1f}%")
        self.logger.info(f"  Memory Usage: {metrics.get('memory_percent', 0):.1f}%")
        self.logger.info(f"  Disk Usage: {metrics.get('disk_percent', 0):.1f}%")
        if self.system_info['gpu_available']:
            self.logger.info(f"  GPU Memory Used: {metrics.get('gpu_memory_used', 0):.1f} MB")
    
    def recommend_model(self) -> str:
        """Recommend the best emotion detection model based on system capabilities."""
        if self.system_info.get('is_raspberry_pi', False):
            pi_model = self.system_info.get('pi_model', 'unknown')
            if pi_model == '4':
                return 'wav2vec2'  # Balanced model for Pi 4
            else:
                return 'basic'     # Lightweight model for Pi 3 and older
        elif self.system_info['gpu_available']:
            return 'hubert'        # Full model for systems with GPU
        else:
            # Check CPU and memory capabilities
            if (self.system_info['cpu_count'] >= 4 and 
                self.system_info['total_memory'] > 4000):  # More than 4GB RAM
                return 'wav2vec2'
            else:
                return 'basic' 