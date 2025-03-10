"""
Platform Checker and System Configurator
===================================
Checks system requirements and configures the application accordingly.
"""

import os
import sys
import logging
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import psutil
from .system_monitor import SystemMonitor

class PlatformChecker:
    """Checks and configures system for optimal performance."""
    
    MINIMUM_REQUIREMENTS = {
        'cpu_count': 2,
        'memory_mb': 1024,  # 1GB
        'disk_gb': 1,
        'python_version': (3, 7)
    }
    
    RECOMMENDED_REQUIREMENTS = {
        'cpu_count': 4,
        'memory_mb': 2048,  # 2GB
        'disk_gb': 5,
        'python_version': (3, 8)
    }
    
    REQUIRED_PACKAGES = [
        'torch',
        'numpy',
        'librosa',
        'soundfile',
        'transformers',
        'cv2'
    ]
    
    def __init__(self):
        """Initialize the platform checker."""
        self.logger = logging.getLogger(__name__)
        self.system_monitor = SystemMonitor()
        self.system_info = self.system_monitor.system_info
        
    def check_python_version(self) -> Tuple[bool, str]:
        """Check if Python version meets requirements."""
        current = tuple(map(int, platform.python_version().split('.')))
        min_version = self.MINIMUM_REQUIREMENTS['python_version']
        
        if current < min_version:
            return False, f"Python {'.'.join(map(str, min_version))} or higher required"
        return True, "Python version OK"
    
    def check_system_resources(self) -> List[Dict]:
        """Check if system meets resource requirements."""
        results = []
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        results.append({
            'component': 'CPU Cores',
            'current': cpu_count,
            'minimum': self.MINIMUM_REQUIREMENTS['cpu_count'],
            'recommended': self.RECOMMENDED_REQUIREMENTS['cpu_count'],
            'status': 'ok' if cpu_count >= self.MINIMUM_REQUIREMENTS['cpu_count'] else 'warning'
        })
        
        # Check Memory
        memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        results.append({
            'component': 'Memory (MB)',
            'current': memory_mb,
            'minimum': self.MINIMUM_REQUIREMENTS['memory_mb'],
            'recommended': self.RECOMMENDED_REQUIREMENTS['memory_mb'],
            'status': 'ok' if memory_mb >= self.MINIMUM_REQUIREMENTS['memory_mb'] else 'warning'
        })
        
        # Check Disk Space
        disk_gb = psutil.disk_usage('/').free / (1024 * 1024 * 1024)
        results.append({
            'component': 'Free Disk Space (GB)',
            'current': disk_gb,
            'minimum': self.MINIMUM_REQUIREMENTS['disk_gb'],
            'recommended': self.RECOMMENDED_REQUIREMENTS['disk_gb'],
            'status': 'ok' if disk_gb >= self.MINIMUM_REQUIREMENTS['disk_gb'] else 'warning'
        })
        
        return results
    
    def check_camera(self) -> Tuple[bool, str]:
        """Check if camera is available and working."""
        try:
            import cv2
            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                ret, frame = camera.read()
                camera.release()
                if ret:
                    return True, "Camera OK"
                else:
                    return False, "Camera not working properly"
            else:
                return False, "Camera not found"
        except Exception as e:
            return False, f"Error checking camera: {str(e)}"
    
    def check_audio(self) -> Tuple[bool, str]:
        """Check if audio input/output is available."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            output_devices = [d for d in devices if d['max_output_channels'] > 0]
            
            if not input_devices:
                return False, "No audio input devices found"
            if not output_devices:
                return False, "No audio output devices found"
                
            return True, "Audio devices OK"
        except Exception as e:
            return False, f"Error checking audio devices: {str(e)}"
    
    def check_packages(self) -> List[Dict]:
        """Check if required packages are installed."""
        results = []
        for package in self.REQUIRED_PACKAGES:
            try:
                __import__(package.replace('-', '_'))
                results.append({
                    'package': package,
                    'status': 'installed'
                })
            except ImportError:
                results.append({
                    'package': package,
                    'status': 'missing'
                })
        return results
    
    def get_optimal_config(self) -> Dict:
        """Generate optimal configuration based on system capabilities."""
        config = {
            'use_gpu': torch.cuda.is_available(),
            'use_quantization': True,
            'batch_size': 1,
            'audio_chunk_size': 4000,
            'feature_cache_size': 10,
            'confidence_threshold': 0.5
        }
        
        # Adjust settings based on platform
        if self.system_info.get('is_raspberry_pi', False):
            config.update({
                'use_quantization': True,
                'use_int8': True,
                'audio_chunk_size': 2000,  # Smaller chunks for Pi
                'feature_cache_size': 5     # Smaller cache for Pi
            })
            
            # Further optimize for Pi model
            if self.system_info.get('pi_model') == '4':
                config.update({
                    'batch_size': 2,
                    'confidence_threshold': 0.4
                })
            else:
                config.update({
                    'batch_size': 1,
                    'confidence_threshold': 0.3
                })
        else:
            # Adjust based on available memory
            memory_gb = self.system_info['total_memory'] / 1024
            if memory_gb >= 8:  # 8GB+ RAM
                config.update({
                    'batch_size': 4,
                    'audio_chunk_size': 8000,
                    'feature_cache_size': 20
                })
            elif memory_gb >= 4:  # 4-8GB RAM
                config.update({
                    'batch_size': 2,
                    'audio_chunk_size': 6000,
                    'feature_cache_size': 15
                })
        
        return config
    
    def check_and_configure(self) -> Dict:
        """Run all checks and generate configuration."""
        results = {
            'python_version': self.check_python_version(),
            'system_resources': self.check_system_resources(),
            'camera': self.check_camera(),
            'audio': self.check_audio(),
            'packages': self.check_packages(),
            'recommended_model': self.system_monitor.recommend_model(),
            'optimal_config': self.get_optimal_config()
        }
        
        # Log results
        self.logger.info("System Check Results:")
        self.logger.info(f"Python Version: {results['python_version'][1]}")
        self.logger.info("System Resources:")
        for resource in results['system_resources']:
            self.logger.info(f"  {resource['component']}: {resource['current']} "
                           f"({resource['status'].upper()})")
        self.logger.info(f"Camera: {results['camera'][1]}")
        self.logger.info(f"Audio: {results['audio'][1]}")
        self.logger.info("Packages:")
        for pkg in results['packages']:
            self.logger.info(f"  {pkg['package']}: {pkg['status'].upper()}")
        self.logger.info(f"Recommended Model: {results['recommended_model']}")
        
        return results
    
    def install_missing_packages(self) -> bool:
        """Attempt to install missing packages."""
        try:
            missing = [p['package'] for p in self.check_packages() 
                      if p['status'] == 'missing']
            
            if not missing:
                return True
                
            self.logger.info(f"Installing missing packages: {', '.join(missing)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            
            # Verify installation
            still_missing = [p['package'] for p in self.check_packages() 
                           if p['status'] == 'missing']
            
            if still_missing:
                self.logger.error(f"Failed to install: {', '.join(still_missing)}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error installing packages: {str(e)}")
            return False
    
    def setup_logging(self, log_dir: str = "logs"):
        """Setup logging configuration."""
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"system_{platform.node()}_{platform.machine()}.log"
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            
            self.logger.info("Logging configured successfully")
            
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            sys.exit(1)
    
    @staticmethod
    def get_camera_list() -> List[Dict]:
        """Get list of available cameras."""
        cameras = []
        try:
            import cv2
            for i in range(10):  # Check first 10 indexes
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cameras.append({
                            'index': i,
                            'resolution': (
                                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            ),
                            'fps': cap.get(cv2.CAP_PROP_FPS)
                        })
                    cap.release()
        except Exception as e:
            logging.error(f"Error getting camera list: {str(e)}")
        return cameras
    
    @staticmethod
    def get_audio_devices() -> Dict:
        """Get list of available audio devices."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return {
                'input': [d for d in devices if d['max_input_channels'] > 0],
                'output': [d for d in devices if d['max_output_channels'] > 0]
            }
        except Exception as e:
            logging.error(f"Error getting audio devices: {str(e)}")
            return {'input': [], 'output': []} 