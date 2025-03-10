"""
System Setup and Package Installation
=================================
Handles both package installation and system configuration.
"""

import os
import sys
import logging
from pathlib import Path
import urllib.request
import subprocess
import platform
import json
import psutil
try:
    import cv2
except ImportError:
    cv2 = None

class SystemSetup:
    """Handles system configuration and checks."""
    
    REQUIRED_PACKAGES = [
        'opencv-python',
        'numpy',
        'torch',
        'transformers',
        'sounddevice',
        'librosa'
    ]

    def __init__(self):
        self.logger = self._setup_logging()
        self.config = {
            'use_gpu': False,
            'batch_size': 1,
            'audio_chunk_size': 1024,
            'feature_cache_size': 1000,
            'use_quantization': False
        }

    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler(log_dir / "setup.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger

    def check_python_version(self):
        """Check Python version."""
        if sys.version_info < (3, 8):
            return False, "Python 3.8 or higher is required"
        return True, "Python version OK"

    def check_camera(self):
        """Check camera availability."""
        if cv2 is None:
            return False, "OpenCV not installed"
        
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return False, "No camera found"
            cap.release()
            return True, "Camera OK"
        except Exception as e:
            return False, f"Camera error: {str(e)}"

    def check_audio_devices(self):
        """Check audio devices."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            inputs = [d for d in devices if d['max_input_channels'] > 0]
            outputs = [d for d in devices if d['max_output_channels'] > 0]
            
            if not inputs:
                return False, "No input audio devices found"
            if not outputs:
                return False, "No output audio devices found"
            
            return True, "Audio devices OK"
        except Exception as e:
            return False, f"Audio device error: {str(e)}"

    def check_gpu(self):
        """Check GPU availability."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.config['use_gpu'] = True
                return True, f"GPU available: {gpu_name}"
            return False, "No GPU found"
        except Exception:
            return False, "Error checking GPU"

    def get_system_info(self):
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'total_memory': psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
            'free_disk': psutil.disk_usage('/').free / (1024 * 1024 * 1024),  # GB
            'gpu_name': self.check_gpu()[1] if self.check_gpu()[0] else "No GPU"
        }

    def optimize_config(self):
        """Optimize configuration based on system capabilities."""
        info = self.get_system_info()
        
        # Adjust batch size based on memory
        if info['total_memory'] > 16:  # More than 16GB RAM
            self.config['batch_size'] = 4
        elif info['total_memory'] > 8:  # More than 8GB RAM
            self.config['batch_size'] = 2
            
        # Enable quantization for lower-end systems
        if info['total_memory'] < 8:
            self.config['use_quantization'] = True
            
        # Adjust audio settings
        if info['cpu_count'] > 4:
            self.config['audio_chunk_size'] = 2048
            self.config['feature_cache_size'] = 2000

        return self.config

def download_file(url, filename):
    """Download a file with progress indicator."""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def setup_models():
    """Setup and download required model files."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Model URLs and their local paths
    MODEL_FILES = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    }

    # Download available models
    for model_name, url in MODEL_FILES.items():
        model_path = models_dir / model_name
        if not model_path.exists():
            download_file(url, model_path)

    # Check for missing custom models
    custom_models = ["emotion_model.pt", "audio_model.pt"]
    missing_custom = [model for model in custom_models if not (models_dir / model).exists()]
    if missing_custom:
        print("\nWarning: The following custom model files need to be downloaded separately:")
        for model in missing_custom:
            print(f"  - {model}")
        print("\nPlease follow the instructions in README.md to obtain these models.")

def setup_environment():
    """Setup virtual environment and install dependencies."""
    if not Path("venv").exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])

    # Determine the correct pip path
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"

    # Install/upgrade pip
    subprocess.run([pip_path, "install", "--upgrade", "pip"])

    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])

def create_env_template():
    """Create .env template file."""
    env_template = """# Environment Configuration
CAMERA_INDEX=0
AUDIO_DEVICE_INDEX=0
GPU_ENABLED=true
MODEL_PATH=models/
LOG_LEVEL=INFO
"""
    with open(".env.template", "w") as f:
        f.write(env_template)
    print("Created .env.template file")

def main():
    """Main setup function."""
    print("Starting Baby Monitor System setup...")
    
    # Initialize system setup
    system = SystemSetup()
    logger = system.logger
    
    # Create necessary directories
    for dir_name in ["models", "logs", "data", "config"]:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_name}")

    # Setup environment
    setup_environment()
    
    # Download and setup models
    setup_models()
    
    # Create .env template
    create_env_template()
    
    # Run system checks
    checks = {
        'python_version': system.check_python_version(),
        'camera': system.check_camera(),
        'audio': system.check_audio_devices(),
        'gpu': system.check_gpu()
    }
    
    # Optimize configuration
    config = system.optimize_config()
    
    # Save configuration
    config_path = Path('config/system_config.json')
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Print setup results
    print("\nSystem Setup Complete!")
    print("=====================")
    
    # System Requirements
    print("\nSystem Requirements Check:")
    for check_name, (status, message) in checks.items():
        print(f"{check_name}: {'✓' if status else '✗'} - {message}")
    
    # System Information
    info = system.get_system_info()
    print(f"\nSystem Information:")
    print(f"CPU Cores: {info['cpu_count']}")
    print(f"Total RAM: {info['total_memory']:.1f}GB")
    print(f"Free Disk Space: {info['free_disk']:.1f}GB")
    print(f"GPU: {info['gpu_name']}")
    
    # Configuration
    print("\nSystem Configuration:")
    print(f"Using GPU: {config['use_gpu']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Audio Chunk Size: {config['audio_chunk_size']}")
    print(f"Quantization Enabled: {config['use_quantization']}")
    
    # Package setup
    setup(
        name="babymonitor",
        version="1.0.0",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        install_requires=[
            "opencv-python>=4.5.0",
            "numpy>=1.19.0",
            "torch>=1.9.0",
            "matplotlib>=3.3.0",
            "pillow>=8.0.0",
            "flask>=2.0.0",
            "flask-socketio>=5.0.0",
            "transformers>=4.5.0",
            "sounddevice>=0.4.0",
            "scipy>=1.7.0",
            "librosa>=0.8.0",
            "ultralytics>=8.0.0",
            "python-dotenv>=0.19.0",
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0"
        ],
        entry_points={
            "console_scripts": [
                "babymonitor=babymonitor.core.main:main",
            ],
        },
        include_package_data=True,
        package_data={
            "babymonitor.web": ["templates/*", "static/*"],
            "edge_comp": ["models/*.pt", "models/*.pth"],
        },
        python_requires=">=3.8",
        author="Your Name",
        author_email="your.email@example.com",
        description="A comprehensive baby monitoring system with video, audio, and emotion detection",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/yourusername/babymonitor",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: End Users/Desktop",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
    )

if __name__ == "__main__":
    main() 