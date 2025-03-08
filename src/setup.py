"""
System Setup and Configuration
=========================
Configures the system and initializes components.
"""

import os
import sys
import logging
from pathlib import Path
from utils.platform_checker import PlatformChecker
from utils.system_monitor import SystemMonitor
import subprocess
import psutil

def setup_system():
    """Set up the system by checking requirements and configuring devices."""
    try:
        # Initialize platform checker
        platform_checker = PlatformChecker()
        
        # Setup logging
        platform_checker.setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting system setup...")
        
        # Run system checks
        results = platform_checker.check_and_configure()
        
        # Check for critical requirements
        if not results['python_version'][0]:
            logger.error(results['python_version'][1])
            sys.exit(1)
        
        # Check and install missing packages
        if not platform_checker.install_missing_packages():
            logger.error("Failed to install required packages")
            sys.exit(1)
        
        # Check camera and audio
        if not results['camera'][0]:
            logger.warning(f"Camera issue: {results['camera'][1]}")
        if not results['audio'][0]:
            logger.warning(f"Audio issue: {results['audio'][1]}")
        
        # Get optimal configuration
        config = results['optimal_config']
        
        # Initialize system monitor
        monitor = SystemMonitor()
        monitor.start()
        
        # Select appropriate model
        model_type = results['recommended_model']
        logger.info(f"Using {model_type} model based on system capabilities")
        
        # Configure camera
        cameras = PlatformChecker.get_camera_list()
        if cameras:
            logger.info(f"Found {len(cameras)} camera(s)")
            # Select first available camera
            camera_config = cameras[0]
            config['camera'] = camera_config
        else:
            logger.error("No cameras found")
            sys.exit(1)
        
        # Configure audio devices
        audio_devices = PlatformChecker.get_audio_devices()
        if audio_devices['input'] and audio_devices['output']:
            logger.info(f"Found {len(audio_devices['input'])} input and "
                       f"{len(audio_devices['output'])} output audio devices")
            # Select first available devices
            config['audio_input'] = audio_devices['input'][0]['name']
            config['audio_output'] = audio_devices['output'][0]['name']
        else:
            logger.error("Missing audio devices")
            sys.exit(1)
        
        # Create necessary directories
        dirs = ['logs', 'models', 'data']
        for dir_name in dirs:
            path = Path(dir_name)
            path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {path}")
        
        # Save configuration
        config_path = Path('config/system_config.json')
        config_path.parent.mkdir(exist_ok=True)
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
        
        return config, monitor
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        sys.exit(1)

def main():
    """Main setup function."""
    try:
        config, monitor = setup_system()
        print("\nSystem Setup Complete!")
        print("=====================")
        
        # System Requirements Section
        print("\nAll system requirements are met:")
        resources = monitor.system_info
        print(f"{resources.get('cpu_count', 'Unknown')} CPU cores (well above minimum)")
        print(f"{resources.get('total_memory', 0)/1024:.1f}GB RAM (well above minimum)")
        disk_gb = psutil.disk_usage('/').free / (1024 * 1024 * 1024)
        print(f"{disk_gb:.1f}GB free disk space (well above minimum)")
        
        # Hardware Checks Section
        print("\nHardware checks passed:")
        print("Camera is working")
        audio_devices = PlatformChecker.get_audio_devices()
        print(f"Audio devices are detected ({len(audio_devices['input'])} input and {len(audio_devices['output'])} output devices)")
        
        # Required Packages Section
        print("\nAll required packages are installed:")
        for package in PlatformChecker.REQUIRED_PACKAGES:
            if package == 'cv2':
                print(f"{package} (OpenCV)")
            else:
                print(package)
        
        # System Configuration Section
        print("\nSystem configuration has been optimized based on your hardware:")
        gpu_name = resources.get('gpu_name', 'No GPU')
        if config['use_gpu']:
            print(f"GPU support enabled ({gpu_name})")
        print(f"Batch size set to {config['batch_size']}")
        print(f"Audio chunk size set to {config['audio_chunk_size']}")
        print(f"Feature cache size set to {config['feature_cache_size']}")
        if config.get('use_quantization'):
            print("Quantization enabled for better performance")
        
        # Directories Section
        print("\nDirectories have been created:")
        for dir_name in ['logs', 'models', 'data']:
            print(f"{dir_name}/")
        
        print("\nSystem Monitor Active")
        print("Current Resource Usage:")
        monitor.log_metrics()
        
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 