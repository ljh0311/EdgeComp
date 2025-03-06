import psutil
import platform
import time
import logging
from datetime import datetime

class SystemMonitor:
    def __init__(self, log_file='baby_monitor.log'):
        self.log_file = log_file
        self.setup_logging()
        self.is_raspberry_pi = self._check_if_raspberry_pi()
        self.start_time = time.time()
        self.last_check = self.start_time
        self.metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'temperature': [],
            'fps': []
        }
    
    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _check_if_raspberry_pi(self):
        if platform.system() == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    return 'Raspberry Pi' in f.read()
            except:
                return False
        return False
    
    def get_cpu_temperature(self):
        if self.is_raspberry_pi:
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    return float(f.read()) / 1000.0
            except:
                return None
        return None
    
    def update_metrics(self, fps=None):
        current_time = time.time()
        
        # Only update every 5 seconds
        if current_time - self.last_check < 5:
            return
            
        self.last_check = current_time
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        temperature = self.get_cpu_temperature()
        
        # Update metrics history
        self.metrics['cpu_percent'].append(cpu_percent)
        self.metrics['memory_percent'].append(memory_percent)
        if temperature:
            self.metrics['temperature'].append(temperature)
        if fps:
            self.metrics['fps'].append(fps)
        
        # Keep only last hour of metrics
        max_samples = 720  # 1 hour at 5-second intervals
        for key in self.metrics:
            if len(self.metrics[key]) > max_samples:
                self.metrics[key] = self.metrics[key][-max_samples:]
        
        # Log current status
        log_msg = (
            f"CPU: {cpu_percent}% | "
            f"Memory: {memory_percent}% | "
            f"FPS: {fps if fps else 'N/A'}"
        )
        if temperature:
            log_msg += f" | Temperature: {temperature}°C"
        
        logging.info(log_msg)
        
        # Check for warning conditions
        self._check_warning_conditions(cpu_percent, memory_percent, temperature, fps)
    
    def _check_warning_conditions(self, cpu_percent, memory_percent, temperature, fps):
        warnings = []
        
        if cpu_percent > 90:
            warnings.append("High CPU usage")
        if memory_percent > 90:
            warnings.append("High memory usage")
        if temperature and temperature > 80:
            warnings.append("High temperature")
        if fps and fps < 10:
            warnings.append("Low FPS")
        
        if warnings:
            warning_msg = " | ".join(warnings)
            logging.warning(warning_msg)
    
    def get_performance_summary(self):
        if not any(self.metrics.values()):
            return "No metrics collected yet"
            
        summary = {
            'cpu_avg': sum(self.metrics['cpu_percent']) / len(self.metrics['cpu_percent']),
            'memory_avg': sum(self.metrics['memory_percent']) / len(self.metrics['memory_percent']),
            'fps_avg': sum(self.metrics['fps']) / len(self.metrics['fps']) if self.metrics['fps'] else None,
            'temp_avg': sum(self.metrics['temperature']) / len(self.metrics['temperature']) if self.metrics['temperature'] else None,
            'uptime': time.time() - self.start_time
        }
        
        return summary 