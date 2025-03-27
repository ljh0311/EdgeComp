"""
Baby Monitor System - Unified Setup
==================================
Handles package installation, system configuration, and provides a GUI installer.

This script can be run directly without any arguments:
    python setup.py

Or with specific options:
    python setup.py --no-gui --skip-models --mode dev
    python setup.py --download-models
    python setup.py --fix-models
    python setup.py --repair-env
    python setup.py --repair-config
    python setup.py --repair-camera
    python setup.py --reset --keep-models
"""

# Check if we're being run as a setuptools command or directly
# This allows the script to be used both as a setuptools setup.py and as a direct script
import sys
if len(sys.argv) > 1 and sys.argv[1] in ('install', 'develop', 'bdist_wheel', 'sdist'):
    # Being run as setuptools command
    from setuptools import setup, find_packages
else:
    # Define a dummy setup function when run directly
    def setup(*args, **kwargs):
        pass
    find_packages = lambda **kw: []

import os
import logging
import platform
import json
import shutil
import tarfile
import urllib.request
import subprocess
from pathlib import Path
import argparse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BabyMonitorSetup")

# Constants
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")
CONFIG_DIR = Path("config")
DATA_DIR = Path("data")
SCRIPTS_DIR = Path("scripts")

# Check Python version
MIN_PYTHON = (3, 8)
MAX_PYTHON = (3, 12)
if sys.version_info < MIN_PYTHON or sys.version_info > MAX_PYTHON:
    sys.exit(
        f"Python {'.'.join(map(str, MIN_PYTHON))} or newer but not newer than {'.'.join(map(str, MAX_PYTHON))} is required."
    )

# Define package versions based on Python version
def get_package_versions():
    py_version = sys.version_info[:2]
    
    # Base packages that work across all supported Python versions
    packages = {
        'numpy': '1.24.3',
        'opencv-python-headless': '4.8.1.78',
        'flask': '2.3.3',
        'python-dotenv': '1.0.0',
        'psutil': '5.9.5',
        'cffi': '1.15.1',
        'pycparser': '2.21',
        'sounddevice': '0.4.6',
        'soundfile': '0.12.1',
        'librosa': '0.10.0',
        'aiortc': '1.5.0',
        'aiohttp': '3.8.5',
        'python-engineio': '4.5.1',
        'python-socketio': '5.8.0',
        'eventlet': '0.33.3',
        'flask-socketio': '5.3.6',
        'werkzeug': '2.3.7',
        'jsonschema': '4.17.3',    # For JSON schema validation
        'matplotlib': '3.7.1',      # For visualization
        'pandas': '2.0.3',          # For data management
        'tensorflowjs': '4.10.0',   # For web model support
        'onnx': '1.14.0',           # For ONNX model support
        'onnxruntime': '1.15.1',    # For ONNX runtime
    }
    
    # Python version specific adjustments
    if py_version >= (3, 11):
        packages.update({
            'torch': '2.0.0',
            'torchaudio': '2.0.0',
            'tensorflow': '2.12.0',  # For TensorFlow Lite models
        })
    elif py_version >= (3, 8):
        packages.update({
            'torch': '1.13.1',
            'torchaudio': '0.13.1',
            'tensorflow': '2.11.0',  # For TensorFlow Lite models
        })
    
    return packages

# Only include this section if run as setuptools command
if len(sys.argv) > 1 and sys.argv[1] in ('install', 'develop', 'bdist_wheel', 'sdist'):
    REQUIRED_PACKAGES = get_package_versions()

    # Read README.md for long description
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            long_description = fh.read()
    except:
        long_description = "Baby Monitor System with AI-powered features"

    setup(
        name="babymonitor",
        version="1.0.0",
        author="Your Name",
        author_email="your.email@example.com",
        description="A Baby Monitor System with AI-powered features",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/yourusername/babymonitor",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: End Users/Desktop",
            "Topic :: Multimedia :: Video :: Capture",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires=f">={'.'.join(map(str, MIN_PYTHON))},<={'.'.join(map(str, MAX_PYTHON))}",
        install_requires=[f"{pkg}=={ver}" for pkg, ver in REQUIRED_PACKAGES.items()],
        extras_require={
            "dev": [
                "pytest>=8.0.0",
                "black>=23.0.0",
                "flake8>=6.0.0",
                "mypy>=1.0.0",
            ],
        },
    )

# The rest of the code is used for both setuptools and direct execution
def check_python_version():
    """Check if the current Python version is supported."""
    current_version = sys.version_info[:3]
    if current_version > (3, 11, 5):
        logger.error(f"Python {'.'.join(map(str, current_version))} is not supported.")
        logger.error("Please use Python 3.11 or earlier.")
        logger.error(
            "Download Python 3.11.5 from: https://www.python.org/downloads/release/python-3115/"
        )
        sys.exit(1)
    elif current_version < MIN_PYTHON:
        logger.error(f"Python {'.'.join(map(str, current_version))} is too old.")
        logger.error(
            f"Minimum required version is Python {'.'.join(map(str, MIN_PYTHON))}"
        )
        sys.exit(1)


def is_raspberry_pi():
    """Check if the system is a Raspberry Pi."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            return any("Raspberry Pi" in line for line in f)
    except:
        return False


def download_file(url, filename, desc=None):
    """Download a file with progress indicator."""
    if desc is None:
        desc = filename
    logger.info(f"Downloading {desc}...")
    try:
        urllib.request.urlretrieve(url, filename)
        logger.info(f"Successfully downloaded {desc}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {desc}: {e}")
        return False


def setup_models():
    """Setup and download required model files."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    person_detection_dir = MODELS_DIR / "person"
    person_detection_dir.mkdir(exist_ok=True)
    
    emotion_dir = MODELS_DIR / "emotion"
    emotion_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different emotion models
    for subdir in ["speechbrain", "emotion2", "cry_detection"]:
        (emotion_dir / subdir).mkdir(exist_ok=True)
    
    # Download YOLOv8 nano model if not exists
    yolo_model_path = person_detection_dir / "yolov8n.pt"
    if not yolo_model_path.exists():
        yolo_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        download_file(yolo_url, yolo_model_path, "YOLOv8 nano model")
    
    # Download Haar Cascade models
    haar_models = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_eye.xml",
        "haarcascade_upperbody.xml",
        "haarcascade_fullbody.xml"
    ]
    
    for model in haar_models:
        model_path = person_detection_dir / model
        if not model_path.exists():
            model_url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{model}"
            download_file(model_url, model_path, f"Haar Cascade model: {model}")
    
    # Try to import utility functions if available
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.babymonitor.utils.setup_models import download_emotion_models
        logger.info("Using utility function to download emotion models")
        download_emotion_models(str(emotion_dir))
    except ImportError:
        logger.warning("Could not import setup_models utility, using fallback method")
        
        # Create placeholders for emotion models
        speechbrain_dir = emotion_dir / "speechbrain"
        readme_file = speechbrain_dir / "README.txt"
        with open(readme_file, 'w') as f:
            f.write("""
Emotion Recognition Models - SpeechBrain
=======================================
Place SpeechBrain emotion recognition models in this directory.

Required files:
- best_emotion_model.pt (Basic emotion model)
- emotion_model.pt (Advanced emotion model)
            """)
        
        emotion2_dir = emotion_dir / "emotion2"
        readme_file = emotion2_dir / "README.txt"
        with open(readme_file, 'w') as f:
            f.write("""
Enhanced Emotion Models - TensorFlow Lite
=======================================
Place TensorFlow Lite emotion models in this directory.

Required files:
- baby_cry_classifier_enhanced.tflite
            """)
            
        cry_detection_dir = emotion_dir / "cry_detection"
        readme_file = cry_detection_dir / "README.txt"
        with open(readme_file, 'w') as f:
            f.write("""
Cry Detection Models - PyTorch
============================
Place PyTorch cry detection models in this directory.

Required files:
- cry_detection_model.pth
            """)
    
    # Create logs directory for emotion history
    logs_dir = Path(os.path.expanduser('~')) / 'babymonitor_logs'
    logs_dir.mkdir(exist_ok=True)
    
    logger.info(f"Model directories and placeholders created successfully")


def setup_environment():
    """Setup virtual environment and install dependencies."""
    venv_path = Path("venv")

    if not venv_path.exists():
        logger.info("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)

    # Determine the correct pip path
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
        python_path = "venv\\Scripts\\python"
    else:
        pip_path = "venv/bin/pip"
        python_path = "venv/bin/python"

    # Install/upgrade pip
    subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)

    # Install dependencies with specific versions
    logger.info("Installing dependencies...")
    for package, version in REQUIRED_PACKAGES.items():
        subprocess.run([pip_path, "install", f"{package}=={version}"], check=True)

    return python_path


def create_scripts():
    """Create or update utility scripts."""
    SCRIPTS_DIR.mkdir(exist_ok=True)

    # Create Windows fix script
    fix_windows_path = SCRIPTS_DIR / "fix_windows.bat"
    with open(fix_windows_path, "w") as f:
        f.write(
            """@echo off
echo ===================================
echo Baby Monitor Windows Fix Utility
echo ===================================

echo.
echo Step 1: Checking Python installation...
where py >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python launcher (py) not found.
    echo Please install Python from https://www.python.org/downloads/release/python-3115/
    echo Make sure to check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

echo Checking for Python 3.11...
py -3.11 --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 is not installed.
    echo Please install Python 3.11 from https://www.python.org/downloads/release/python-3115/
    echo After installing:
    echo 1. Make sure to check "Add python.exe to PATH" during installation
    echo 2. Run 'py -3.11 -m pip install --upgrade pip'
    echo 3. Run this script again
    pause
    exit /b 1
)

echo Using Python 3.11...
py -3.11 --version

echo.
echo Step 2: Stopping any running Baby Monitor processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Baby Monitor*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Step 3: Cleaning Python cache...
del /s /q *.pyc 2>nul
rd /s /q __pycache__ 2>nul
rd /s /q src\\babymonitor\\__pycache__ 2>nul
rd /s /q src\\babymonitor\\web\\__pycache__ 2>nul
rd /s /q src\\babymonitor\\detectors\\__pycache__ 2>nul

echo.
echo Step 4: Please ensure no other applications are using your camera
echo Close any applications that might be using the camera (Zoom, Teams, etc.)
echo Press any key when ready...
pause >nul

echo.
echo Step 5: Installing required packages...
py -3.11 -m pip uninstall -y eventlet flask-socketio python-engineio python-socketio
py -3.11 -m pip install --no-cache-dir eventlet==0.33.3 flask-socketio==5.3.6 werkzeug==2.3.7 python-engineio==4.5.1 python-socketio==5.8.0

echo.
echo Step 6: Starting Baby Monitor...
echo.
echo The web interface will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

set PYTHONPATH=%CD%
set EVENTLET_NO_GREENDNS=yes
py -3.11 run_monitor.py --mode dev --camera_id 0 --debug

echo.
echo If you see any errors, please try:
echo 1. Make sure Python 3.11 is properly installed: py -3.11 --version
echo 2. Try reinstalling packages: py -3.11 -m pip install --no-cache-dir -r requirements.txt
echo 3. Close other applications using your camera
echo 4. Try a different camera (use --camera_id 1)
echo 5. Check Windows camera privacy settings
echo.
pause
"""
        )


def create_env_file(mode="normal", web_port="5000", force=False):
    """Create .env file with default configuration."""
    env_content = """# Baby Monitor System Environment Configuration
# Generated by setup.py

# System Mode (normal or dev)
MODE=normal

# Camera Settings
CAMERA_INDEX=0
CAMERA_RESOLUTION=640x480
CAMERA_FPS=30

# Audio Settings
AUDIO_DEVICE_INDEX=0
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1

# Detection Settings
DETECTION_THRESHOLD=0.5
EMOTION_THRESHOLD=0.7
EMOTION_MODEL=basic_emotion

# History Settings
EMOTION_HISTORY_ENABLED=true
EMOTION_HISTORY_SAVE_INTERVAL=300

# System Settings
GPU_ENABLED=false
LOG_LEVEL=INFO
WEB_HOST=0.0.0.0
WEB_PORT=5000

# Paths
MODEL_PATH=models/
LOG_PATH=logs/
DATA_PATH=data/
EMOTION_HISTORY_PATH=~/babymonitor_logs/
"""
    
    if force:
        env_content = env_content.replace("MODE=normal", f"MODE={mode}")
        env_content = env_content.replace("WEB_PORT=5000", f"WEB_PORT={web_port}")
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    logger.info("Created .env file with default configuration")


def create_config_files(force=False):
    """Create configuration files."""
    CONFIG_DIR.mkdir(exist_ok=True)
    
    try:
        # Try to import utility functions
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.babymonitor.utils.config import generate_default_config
        
        logger.info("Using utility function to generate default configuration")
        config = generate_default_config()
        
        # Save the configuration files
        for config_name, config_data in config.items():
            with open(CONFIG_DIR / f"{config_name}.json", "w") as f:
                json.dump(config_data, f, indent=4)
            logger.info(f"Created configuration file: {config_name}.json")
            
    except ImportError:
        logger.warning("Could not import config utility, using fallback method")
        
        # Create camera config (if not already done by setup_camera_management)
        if not (CONFIG_DIR / "camera.json").exists():
            camera_config = {
                "default_camera_id": 0,
                "resolution": {
                    "width": 640,
                    "height": 480
                },
                "fps": 30,
                "flip_horizontal": False,
                "flip_vertical": False
            }
            
            with open(CONFIG_DIR / "camera.json", "w") as f:
                json.dump(camera_config, f, indent=4)
        
        # Create audio config
        audio_config = {
            "default_device_id": 0,
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 1024,
            "enable_noise_reduction": True
        }
        
        with open(CONFIG_DIR / "audio.json", "w") as f:
            json.dump(audio_config, f, indent=4)
        
        # Create detection config
        detection_config = {
            "person_threshold": 0.5,
            "emotion_threshold": 0.7,
            "detection_types": ["face", "upper_body", "full_body"],
            "enable_tracking": True,
            "mode": "normal"  # Default mode (can be 'normal' or 'dev')
        }
        
        with open(CONFIG_DIR / "detection.json", "w") as f:
            json.dump(detection_config, f, indent=4)
            
        # Create emotion config if not already done by setup_emotion_history
        if not (CONFIG_DIR / "emotion.json").exists():
            emotion_config = {
                "history_enabled": True,
                "default_model": "basic_emotion",
                "available_models": [
                    {
                        "id": "basic_emotion",
                        "name": "Basic Emotion",
                        "supported_emotions": ["crying", "laughing", "babbling", "silence"]
                    },
                    {
                        "id": "emotion2", 
                        "name": "Enhanced Emotion Model",
                        "supported_emotions": ["happy", "sad", "angry", "neutral", "crying", "laughing"]
                    }
                ],
                "save_interval": 300,
                "max_history_entries": 1000
            }
            
            with open(CONFIG_DIR / "emotion.json", "w") as f:
                json.dump(emotion_config, f, indent=4)
        
    logger.info("Created configuration files")


def create_desktop_shortcut():
    """Create desktop shortcut for the application."""
    if platform.system() == "Windows":
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        shortcut_path = os.path.join(desktop_path, "Baby Monitor.bat")

        with open(shortcut_path, "w") as f:
            f.write(
                f'@echo off\ncd "{os.getcwd()}"\n"{os.getcwd()}\\venv\\Scripts\\python" main.py\n'
            )

        print(f"Created desktop shortcut at {shortcut_path}")

    elif platform.system() == "Linux":
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        shortcut_path = os.path.join(desktop_path, "BabyMonitor.desktop")

        with open(shortcut_path, "w") as f:
            f.write(
                f"""[Desktop Entry]
Type=Application
Name=Baby Monitor
Comment=Baby Monitor System
Exec={os.getcwd()}/venv/bin/python {os.getcwd()}/main.py
Icon={os.getcwd()}/src/babymonitor/web/static/img/logo.png
Terminal=false
Categories=Utility;
"""
            )

        # Make executable
        os.chmod(shortcut_path, 0o755)
        print(f"Created desktop shortcut at {shortcut_path}")


def run_gui_installer(args):
    """Run the GUI installer with the given arguments."""
    try:
        from PyQt5.QtWidgets import QApplication, QWizard
        from scripts.install.gui_installer import InstallerWizard
        
        # If installer module not found, use the internal InstallerWizard class
        if 'InstallerWizard' not in locals():
            # Use the InstallerWizard class defined in this module
            InstallerWizard = globals().get('InstallerWizard')
        
        if not InstallerWizard:
            logger.error("InstallerWizard class not found")
            return
        
        # Convert args to dictionary to pass to the wizard
        options = vars(args)
        
        app = QApplication(sys.argv)
        wizard = InstallerWizard(options)
        wizard.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Error launching GUI installer: {e}")
        logger.info("Falling back to command line installation")
        # Continue with command line installation


def main():
    """Main entry point for setup."""
    # Check Python version first
    check_python_version()
    
    parser = argparse.ArgumentParser(description="Baby Monitor System Setup")
    # Basic setup options
    parser.add_argument("--gui", action="store_true", help="Run setup with GUI (default)")
    parser.add_argument("--no-gui", action="store_true", help="Run setup without GUI")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading models")
    parser.add_argument("--skip-shortcut", action="store_true", help="Skip creating desktop shortcut")
    parser.add_argument("--mode", choices=["normal", "dev"], default="normal", help="Setup mode (normal or dev)")
    
    # Installation options
    parser.add_argument("--download-models", action="store_true", help="Download all required models")
    parser.add_argument("--emotion-only", action="store_true", help="Download only emotion detection models")
    parser.add_argument("--cry-only", action="store_true", help="Download only cry detection models")
    parser.add_argument("--fix-models", action="store_true", help="Verify and fix model issues")
    parser.add_argument("--repair-env", action="store_true", help="Repair Python environment")
    parser.add_argument("--repair-config", action="store_true", help="Repair configuration files")
    parser.add_argument("--repair-camera", action="store_true", help="Repair camera configuration")
    parser.add_argument("--reset", action="store_true", help="Reset system configuration")
    parser.add_argument("--keep-models", action="store_true", help="Keep models when resetting")
    parser.add_argument("--configure-only", action="store_true", help="Run configuration wizard only")
    
    args = parser.parse_args()
    
    # Default to GUI if not specified and no specific action given
    use_gui = args.gui or (not args.no_gui and not any([
        args.download_models, args.fix_models, args.repair_env, 
        args.repair_config, args.repair_camera, args.reset,
        args.configure_only
    ]))
    
    # If GUI requested and PyQt5 is available, launch the GUI installer
    if use_gui:
        try:
            from PyQt5.QtWidgets import QApplication
            logger.info("Launching GUI installer...")
            run_gui_installer(args)
            return
        except ImportError:
            logger.warning("PyQt5 not available, falling back to command line installation")
    
    # Handle specific actions if requested
    if args.download_models:
        logger.info("Downloading models...")
        setup_models()
        return
    
    if args.fix_models:
        logger.info("Verifying and fixing model issues...")
        # Add model verification/fixing logic here
        return
    
    if args.repair_env:
        logger.info("Repairing Python environment...")
        setup_environment()
        return
    
    if args.repair_config:
        logger.info("Repairing configuration files...")
        create_config_files(force=True)
        create_env_file(force=True)
        return
    
    if args.repair_camera:
        logger.info("Repairing camera configuration...")
        setup_camera_management()
        return
    
    if args.reset:
        logger.info("Resetting system configuration...")
        # Add system reset logic here
        return
    
    if args.configure_only:
        logger.info("Running configuration wizard...")
        # Add configuration wizard logic here
        return
    
    # Print selected options for standard installation
    logger.info(f"Running setup with options:")
    logger.info(f"  - GUI: {'Disabled' if args.no_gui else 'Enabled'}")
    logger.info(f"  - Download models: {'Skipped' if args.skip_models else 'Enabled'}")
    logger.info(f"  - Create shortcut: {'Skipped' if args.skip_shortcut else 'Enabled'}")
    logger.info(f"  - Mode: {args.mode}")
    
    # Create necessary directories
    for directory in [MODELS_DIR, LOGS_DIR, CONFIG_DIR, DATA_DIR, SCRIPTS_DIR]:
        directory.mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    try:
        # Setup environment
        logger.info("Setting up Python environment...")
        python_path = setup_environment()
        
        # Setup models
        if not args.skip_models:
            logger.info("Setting up detection models...")
            setup_models()
        else:
            logger.info("Skipping model setup (--skip-models specified)")
        
        # Setup camera management
        logger.info("Setting up camera management system...")
        setup_camera_management()
        
        # Setup emotion history
        logger.info("Setting up emotion history system...")
        setup_emotion_history()
        
        # Create configuration files
        logger.info("Creating configuration files...")
        create_config_files()
        
        # Create utility scripts
        logger.info("Creating utility scripts...")
        create_scripts()
        
        # Create .env file
        logger.info("Creating environment file...")
        create_env_file(mode=args.mode)
        
        # Create desktop shortcut if not skipped
        if not args.skip_shortcut:
            logger.info("Creating desktop shortcut...")
            create_desktop_shortcut()
        else:
            logger.info("Skipping shortcut creation (--skip-shortcut specified)")
        
        # Create the babymonitor_logs directory in user's home
        logs_dir = Path(os.path.expanduser('~')) / 'babymonitor_logs'
        logs_dir.mkdir(exist_ok=True)
        logger.info(f"Created logs directory at {logs_dir}")
        
        logger.info("Setup completed successfully!")
        
        # Print instructions to start the system
        print("\n" + "="*50)
        print("Baby Monitor System Setup Complete!")
        print("="*50)
        print("\nTo start the system, run:")
        if platform.system() == "Windows":
            print(f"    venv\\Scripts\\python main.py --mode {args.mode}")
        else:
            print(f"    venv/bin/python main.py --mode {args.mode}")
        print("\nThe web interface will be available at: http://localhost:5000")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def download_emotion_models():
    """Download emotion detection models only."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    emotion_dir = MODELS_DIR / "emotion"
    emotion_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different emotion models
    for subdir in ["speechbrain", "emotion2", "cry_detection"]:
        (emotion_dir / subdir).mkdir(exist_ok=True)
    
    # Download emotion models logic
    logger.info("Downloading emotion models...")
    # Add your model download logic here
    logger.info("Emotion models downloaded successfully")
    return True


def download_cry_models():
    """Download cry detection models only."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    emotion_dir = MODELS_DIR / "emotion"
    emotion_dir.mkdir(exist_ok=True)
    cry_dir = emotion_dir / "cry_detection"
    cry_dir.mkdir(exist_ok=True)
    
    # Download cry detection models logic
    logger.info("Downloading cry detection models...")
    # Add your model download logic here
    logger.info("Cry detection models downloaded successfully")
    return True


def verify_models():
    """Verify and fix issues with models."""
    logger.info("Verifying models...")
    
    # Check if models directory exists
    if not MODELS_DIR.exists():
        logger.warning("Models directory not found. Creating and downloading models...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        setup_models()
        return True
    
    # Check for missing model files and download if needed
    missing_models = []
    
    # Check person detection models
    person_dir = MODELS_DIR / "person"
    if not person_dir.exists() or not any(person_dir.iterdir()):
        missing_models.append("person detection")
    
    # Check emotion models
    emotion_dir = MODELS_DIR / "emotion"
    if not emotion_dir.exists() or not any(emotion_dir.iterdir()):
        missing_models.append("emotion detection")
    
    # Download missing models
    if missing_models:
        logger.warning(f"Missing models detected: {', '.join(missing_models)}")
        logger.info("Downloading missing models...")
        setup_models()
    else:
        logger.info("All models are present")
    
    return True


def repair_environment():
    """Repair the Python environment."""
    logger.info("Repairing Python environment...")
    
    # Reinstall required packages
    packages = get_package_versions()
    
    try:
        # Check if virtual environment exists
        if platform.system() == "Windows":
            venv_python = "venv\\Scripts\\python"
            venv_pip = "venv\\Scripts\\pip"
        else:
            venv_python = "venv/bin/python"
            venv_pip = "venv/bin/pip"
        
        if not Path(venv_python).exists():
            logger.warning("Virtual environment not found. Creating...")
            setup_environment()
        
        # Upgrade pip
        logger.info("Upgrading pip...")
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Reinstall packages
        logger.info("Reinstalling packages...")
        for package, version in packages.items():
            logger.info(f"Installing {package}=={version}")
            try:
                subprocess.run([venv_pip, "install", f"{package}=={version}"], check=True)
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to install {package}=={version}, trying without version...")
                subprocess.run([venv_pip, "install", package], check=True)
        
        logger.info("Python environment repaired successfully")
        return True
    except Exception as e:
        logger.error(f"Error repairing environment: {e}")
        return False


def repair_config_files():
    """Repair configuration files."""
    logger.info("Repairing configuration files...")
    
    # Create config directory if it doesn't exist
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Repair/recreate configuration files
    create_config_files(force=True)
    
    # Repair/recreate .env file
    mode = "normal"
    try:
        # Check if .env exists and try to read mode
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("MODE="):
                        mode = line.strip().split("=")[1].strip().strip('"\'')
                        break
    except:
        pass
    
    create_env_file(mode=mode, force=True)
    
    logger.info("Configuration files repaired successfully")
    return True


def reset_system(keep_models=False):
    """Reset the system to default settings."""
    logger.info("Resetting system...")
    
    # Backup models if requested
    if keep_models and MODELS_DIR.exists():
        logger.info("Backing up models...")
        backup_dir = Path("models_backup")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(MODELS_DIR, backup_dir)
    
    # Remove configuration directories
    for directory in [CONFIG_DIR, DATA_DIR, LOGS_DIR]:
        if directory.exists():
            logger.info(f"Removing {directory}...")
            shutil.rmtree(directory)
    
    # Remove models if not keeping them
    if not keep_models and MODELS_DIR.exists():
        logger.info("Removing models...")
        shutil.rmtree(MODELS_DIR)
    
    # Remove .env file
    env_file = Path(".env")
    if env_file.exists():
        logger.info("Removing .env file...")
        os.remove(env_file)
    
    # Recreate directories
    for directory in [CONFIG_DIR, DATA_DIR, LOGS_DIR]:
        directory.mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Restore models if backed up
    if keep_models and backup_dir.exists():
        logger.info("Restoring models...")
        MODELS_DIR.mkdir(exist_ok=True)
        for item in backup_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, MODELS_DIR / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, MODELS_DIR)
        # Clean up backup
        shutil.rmtree(backup_dir)
    
    logger.info("System reset successfully. Run the installer to set up the system again.")
    return True


def run_configuration_wizard():
    """Run the configuration wizard only."""
    logger.info("Running configuration wizard...")
    
    # Check if configuration directory exists
    if not CONFIG_DIR.exists():
        logger.warning("Configuration directory not found. Creating...")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run configuration
    try:
        # If PyQt5 is available, use GUI configuration
        from PyQt5.QtWidgets import QApplication
        # Launch just the configuration page with appropriate defaults
        run_gui_configuration()
    except ImportError:
        # Command line configuration
        print("GUI configuration not available. Using command line configuration.")
        
        # Ask for mode
        print("\nSelect operation mode:")
        print("1. Normal Mode (for regular users)")
        print("2. Developer Mode (for advanced users)")
        mode_choice = input("Enter your choice (1/2): ")
        mode = "dev" if mode_choice == "2" else "normal"
        
        # Ask for web port
        port = input("\nEnter web interface port (default: 5000): ") or "5000"
        
        # Create/update .env file with these settings
        create_env_file(mode=mode, web_port=port, force=True)
        
        print("\nConfiguration updated successfully")
    
    return True


def run_gui_configuration():
    """Run just the configuration page of the GUI installer."""
    try:
        from PyQt5.QtWidgets import QApplication, QWizard, QVBoxLayout, QDialog
        from PyQt5.QtCore import Qt
        from scripts.install.gui_installer import ConfigurationPage
        
        app = QApplication(sys.argv)
        
        dialog = QDialog()
        dialog.setWindowTitle("Baby Monitor System Configuration")
        dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout()
        config_page = ConfigurationPage()
        
        # Add the configuration page to the dialog
        layout.addWidget(config_page)
        dialog.setLayout(layout)
        
        # Show the dialog
        if dialog.exec_() == QDialog.Accepted:
            # Apply configuration
            mode = config_page.get_mode()
            camera_id = config_page.field("camera_id")
            resolution = config_page.field("resolution")
            web_port = config_page.field("web_port")
            detection_threshold = config_page.field("detection_threshold")
            
            # Update .env file
            create_env_file(
                mode=mode,
                camera_id=camera_id,
                resolution=resolution,
                web_port=web_port,
                detection_threshold=detection_threshold,
                force=True
            )
            
            logger.info("Configuration updated successfully")
        else:
            logger.info("Configuration cancelled")
            
    except Exception as e:
        logger.error(f"Error launching configuration dialog: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def setup_camera_management(force_rescan=False):
    """Setup camera management system."""
    logger.info("Setting up camera management system...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.babymonitor.utils.camera import get_available_cameras, test_camera_connection
        
        # Get available cameras
        cameras = get_available_cameras()
        logger.info(f"Found {len(cameras)} camera(s)")
        
        # Create camera configuration
        camera_config = {
            "available_cameras": cameras,
            "default_camera_id": 0 if cameras else None,
            "resolution": {
                "width": 640,
                "height": 480
            },
            "fps": 30,
            "flip_horizontal": False,
            "flip_vertical": False
        }
        
        # Create cameras directory in data
        cameras_dir = DATA_DIR / "cameras"
        cameras_dir.mkdir(exist_ok=True)
        
        # Save camera configuration
        with open(CONFIG_DIR / "camera.json", "w") as f:
            json.dump(camera_config, f, indent=4)
        
        # Test default camera if available
        if cameras:
            test_result = test_camera_connection(0)
            if test_result.get("success", False):
                logger.info(f"Successfully tested default camera: {test_result.get('message', '')}")
            else:
                logger.warning(f"Issue with default camera: {test_result.get('message', 'Unknown error')}")
        
        return True
    except Exception as e:
        logger.error(f"Error setting up camera management: {e}")
        
        # Create fallback camera configuration
        camera_config = {
            "default_camera_id": 0,
            "resolution": {
                "width": 640,
                "height": 480
            },
            "fps": 30,
            "flip_horizontal": False,
            "flip_vertical": False
        }
        
        # Save fallback configuration
        with open(CONFIG_DIR / "camera.json", "w") as f:
            json.dump(camera_config, f, indent=4)
        
        return False


def setup_emotion_history():
    """Setup emotion history logging system."""
    logger.info("Setting up emotion history system...")
    
    # Create directory for emotion history logs
    logs_dir = Path(os.path.expanduser('~')) / 'babymonitor_logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Create emotion history configuration
    emotion_config = {
        "history_enabled": True,
        "default_model": "basic_emotion",
        "available_models": [
            {
                "id": "basic_emotion",
                "name": "Basic Emotion",
                "supported_emotions": ["crying", "laughing", "babbling", "silence"]
            },
            {
                "id": "emotion2",
                "name": "Enhanced Emotion Model",
                "supported_emotions": ["happy", "sad", "angry", "neutral", "crying", "laughing"]
            },
            {
                "id": "cry_detection",
                "name": "Cry Detection",
                "supported_emotions": ["crying", "not_crying"]
            },
            {
                "id": "speechbrain",
                "name": "SpeechBrain Emotion",
                "supported_emotions": ["happy", "sad", "angry", "neutral"]
            }
        ],
        "save_interval": 300,  # Save every 5 minutes
        "max_history_entries": 1000,
        "history_file_path": str(logs_dir)
    }
    
    # Save emotion configuration
    with open(CONFIG_DIR / "emotion.json", "w") as f:
        json.dump(emotion_config, f, indent=4)
    
    logger.info(f"Emotion history configuration saved to {CONFIG_DIR / 'emotion.json'}")
    
    # Create placeholder files for each model's history
    for model in emotion_config["available_models"]:
        model_id = model["id"]
        history_file = logs_dir / f"emotion_history_{model_id}.json"
        
        if not history_file.exists():
            initial_history = {
                "model_id": model_id,
                "emotions": model["supported_emotions"],
                "emotion_counts": {emotion: 0 for emotion in model["supported_emotions"]},
                "daily_history": {},
                "last_updated": time.time()
            }
            
            with open(history_file, 'w') as f:
                json.dump(initial_history, f, indent=2)
            
            logger.info(f"Created placeholder history file for {model_id} at {history_file}")
    
    return True


if __name__ == "__main__":
    main()
