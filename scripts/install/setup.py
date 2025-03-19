"""
Baby Monitor System - Unified Setup
==================================
Handles package installation, system configuration, and provides a GUI installer.
"""

from setuptools import setup, find_packages
import os
import sys
import logging
import platform
import json
import shutil
import tarfile
import urllib.request
import subprocess
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BabyMonitorSetup')

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
    sys.exit(f"Python {'.'.join(map(str, MIN_PYTHON))} or newer but not newer than {'.'.join(map(str, MAX_PYTHON))} is required.")

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
    }
    
    # Python version specific adjustments
    if py_version >= (3, 11):
        packages.update({
            'torch': '2.0.0',
            'torchaudio': '2.0.0',
        })
    elif py_version >= (3, 8):
        packages.update({
            'torch': '1.13.1',
            'torchaudio': '0.13.1',
        })
    
    return packages

REQUIRED_PACKAGES = get_package_versions()

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
        'dev': [
            'pytest>=8.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
    }
)

def check_python_version():
    """Check if the current Python version is supported."""
    current_version = sys.version_info[:3]
    if current_version > (3, 11, 5):
        logger.error(f"Python {'.'.join(map(str, current_version))} is not supported.")
        logger.error("Please use Python 3.11 or earlier.")
        logger.error("Download Python 3.11.5 from: https://www.python.org/downloads/release/python-3115/")
        sys.exit(1)
    elif current_version < MIN_PYTHON:
        logger.error(f"Python {'.'.join(map(str, current_version))} is too old.")
        logger.error(f"Minimum required version is Python {'.'.join(map(str, MIN_PYTHON))}")
        sys.exit(1)

def is_raspberry_pi():
    """Check if the system is a Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
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
    
    # Download YOLOv8 nano model if not exists
    yolo_model_path = MODELS_DIR / "yolov8n.pt"
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
        model_path = MODELS_DIR / model
        if not model_path.exists():
            model_url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{model}"
            download_file(model_url, model_path, f"Haar Cascade model: {model}")
    
    # Create emotion models directory
    emotion_dir = MODELS_DIR / "emotion"
    emotion_dir.mkdir(exist_ok=True)
    
    # Create a placeholder for emotion model
    emotion_model_info = emotion_dir / "README.txt"
    if not emotion_model_info.exists():
        with open(emotion_model_info, 'w') as f:
            f.write("""
Emotion Recognition Models
=========================
Place your emotion recognition models in this directory.

Supported formats:
- PyTorch (.pt, .pth)
- TensorFlow (.pb, .tflite)
- ONNX (.onnx)

The system will automatically detect and use the appropriate model.
            """)

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
    with open(fix_windows_path, 'w') as f:
        f.write("""@echo off
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
""")

def create_env_file():
    """Create .env file with default configuration."""
    env_content = """# Baby Monitor System Environment Configuration
# Generated by setup.py

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

# System Settings
GPU_ENABLED=false
LOG_LEVEL=INFO
WEB_HOST=0.0.0.0
WEB_PORT=5000

# Paths
MODEL_PATH=models/
LOG_PATH=logs/
DATA_PATH=data/
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("Created .env file with default configuration")

def create_config_files():
    """Create configuration files."""
    CONFIG_DIR.mkdir(exist_ok=True)
    
    # Create camera config
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
        "enable_tracking": True
    }
    
    with open(CONFIG_DIR / "detection.json", "w") as f:
        json.dump(detection_config, f, indent=4)
    
    print("Created configuration files")

def create_desktop_shortcut():
    """Create desktop shortcut for the application."""
    if platform.system() == "Windows":
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        shortcut_path = os.path.join(desktop_path, "Baby Monitor.bat")
        
        with open(shortcut_path, "w") as f:
            f.write(f'@echo off\ncd "{os.getcwd()}"\n"{os.getcwd()}\\venv\\Scripts\\python" main.py\n')
        
        print(f"Created desktop shortcut at {shortcut_path}")
    
    elif platform.system() == "Linux":
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        shortcut_path = os.path.join(desktop_path, "BabyMonitor.desktop")
        
        with open(shortcut_path, "w") as f:
            f.write(f"""[Desktop Entry]
Type=Application
Name=Baby Monitor
Comment=Baby Monitor System
Exec={os.getcwd()}/venv/bin/python {os.getcwd()}/main.py
Icon={os.getcwd()}/src/babymonitor/web/static/img/logo.png
Terminal=false
Categories=Utility;
""")
        
        # Make executable
        os.chmod(shortcut_path, 0o755)
        print(f"Created desktop shortcut at {shortcut_path}")

def run_gui_installer():
    """Run the graphical installer."""
    try:
        from PyQt5.QtWidgets import QApplication, QWizard, QWizardPage, QLabel, QVBoxLayout, QCheckBox, QProgressBar, QComboBox, QPushButton, QLineEdit, QFileDialog, QHBoxLayout, QRadioButton, QButtonGroup, QGroupBox
        from PyQt5.QtCore import Qt, QThread, pyqtSignal
        from PyQt5.QtGui import QFont, QPixmap, QIcon
    except ImportError:
        print("PyQt5 not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "PyQt5"])
        from PyQt5.QtWidgets import QApplication, QWizard, QWizardPage, QLabel, QVBoxLayout, QCheckBox, QProgressBar, QComboBox, QPushButton, QLineEdit, QFileDialog, QHBoxLayout, QRadioButton, QButtonGroup, QGroupBox
        from PyQt5.QtCore import Qt, QThread, pyqtSignal
        from PyQt5.QtGui import QFont, QPixmap, QIcon
    
    class SetupThread(QThread):
        progress_update = pyqtSignal(int, str)
        finished_signal = pyqtSignal()
        
        def __init__(self, options):
            super().__init__()
            self.options = options
        
        def run(self):
            total_steps = 7
            current_step = 0
            
            # Create directories
            self.progress_update.emit(int(current_step / total_steps * 100), "Creating directories...")
            for directory in [MODELS_DIR, LOGS_DIR, CONFIG_DIR, DATA_DIR]:
                directory.mkdir(exist_ok=True)
            current_step += 1
            
            # Setup environment
            self.progress_update.emit(int(current_step / total_steps * 100), "Setting up environment...")
            python_path = setup_environment()
            current_step += 1
            
            # Setup models
            if self.options.get('download_models', True):
                self.progress_update.emit(int(current_step / total_steps * 100), "Downloading models...")
                setup_models()
            current_step += 1
            
            # Raspberry Pi specific setup
            if is_raspberry_pi() and self.options.get('setup_raspberry_pi', True):
                self.progress_update.emit(int(current_step / total_steps * 100), "Setting up Raspberry Pi...")
                setup_raspberry_pi()
            current_step += 1
            
            # Create configuration files
            self.progress_update.emit(int(current_step / total_steps * 100), "Creating configuration files...")
            create_config_files()
            current_step += 1
            
            # Create .env file
            self.progress_update.emit(int(current_step / total_steps * 100), "Creating environment file...")
            create_env_file()
            current_step += 1
            
            # Create desktop shortcut
            if self.options.get('create_shortcut', True):
                self.progress_update.emit(int(current_step / total_steps * 100), "Creating desktop shortcut...")
                create_desktop_shortcut()
            
            self.progress_update.emit(100, "Setup completed!")
            self.finished_signal.emit()
    
    class WelcomePage(QWizardPage):
        def __init__(self):
            super().__init__()
            self.setTitle("Welcome to Baby Monitor System Setup")
            
            layout = QVBoxLayout()
            
            # Add logo
            logo_label = QLabel()
            logo_path = Path("src/babymonitor/web/static/img/logo.png")
            if logo_path.exists():
                pixmap = QPixmap(str(logo_path))
                logo_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
                logo_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(logo_label)
            
            # Welcome text
            welcome_text = QLabel(
                "This wizard will guide you through the installation and setup of the Baby Monitor System.\n\n"
                "The setup will:\n"
                "• Install required dependencies\n"
                "• Download necessary models\n"
                "• Configure the system\n"
                "• Create desktop shortcuts\n\n"
                "Click Next to continue."
            )
            welcome_text.setWordWrap(True)
            layout.addWidget(welcome_text)
            
            self.setLayout(layout)
    
    class ComponentsPage(QWizardPage):
        def __init__(self):
            super().__init__()
            self.setTitle("Select Components to Install")
            
            layout = QVBoxLayout()
            
            # Components selection
            components_label = QLabel("Select the components you want to install:")
            layout.addWidget(components_label)
            
            self.download_models_cb = QCheckBox("Download detection models")
            self.download_models_cb.setChecked(True)
            layout.addWidget(self.download_models_cb)
            
            self.create_shortcut_cb = QCheckBox("Create desktop shortcut")
            self.create_shortcut_cb.setChecked(True)
            layout.addWidget(self.create_shortcut_cb)
            
            if is_raspberry_pi():
                self.setup_raspberry_pi_cb = QCheckBox("Configure Raspberry Pi (camera, audio)")
                self.setup_raspberry_pi_cb.setChecked(True)
                layout.addWidget(self.setup_raspberry_pi_cb)
            
            self.registerField("download_models", self.download_models_cb)
            self.registerField("create_shortcut", self.create_shortcut_cb)
            if is_raspberry_pi():
                self.registerField("setup_raspberry_pi", self.setup_raspberry_pi_cb)
            
            self.setLayout(layout)
    
    class ConfigurationPage(QWizardPage):
        def __init__(self):
            super().__init__()
            self.setTitle("System Configuration")
            
            layout = QVBoxLayout()
            
            # Camera configuration
            camera_group = QGroupBox("Camera Configuration")
            camera_layout = QVBoxLayout()
            
            camera_id_layout = QHBoxLayout()
            camera_id_label = QLabel("Camera ID:")
            self.camera_id_combo = QComboBox()
            self.camera_id_combo.addItems(["0", "1", "2", "3"])
            camera_id_layout.addWidget(camera_id_label)
            camera_id_layout.addWidget(self.camera_id_combo)
            camera_layout.addLayout(camera_id_layout)
            
            resolution_layout = QHBoxLayout()
            resolution_label = QLabel("Resolution:")
            self.resolution_combo = QComboBox()
            self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
            resolution_layout.addWidget(resolution_label)
            resolution_layout.addWidget(self.resolution_combo)
            camera_layout.addLayout(resolution_layout)
            
            camera_group.setLayout(camera_layout)
            layout.addWidget(camera_group)
            
            # Web interface configuration
            web_group = QGroupBox("Web Interface Configuration")
            web_layout = QVBoxLayout()
            
            port_layout = QHBoxLayout()
            port_label = QLabel("Web Port:")
            self.port_edit = QLineEdit("5000")
            port_layout.addWidget(port_label)
            port_layout.addWidget(self.port_edit)
            web_layout.addLayout(port_layout)
            
            web_group.setLayout(web_layout)
            layout.addWidget(web_group)
            
            # Detection configuration
            detection_group = QGroupBox("Detection Configuration")
            detection_layout = QVBoxLayout()
            
            threshold_layout = QHBoxLayout()
            threshold_label = QLabel("Detection Threshold:")
            self.threshold_combo = QComboBox()
            self.threshold_combo.addItems(["0.3 (More sensitive)", "0.5 (Balanced)", "0.7 (Less sensitive)"])
            self.threshold_combo.setCurrentIndex(1)
            threshold_layout.addWidget(threshold_label)
            threshold_layout.addWidget(self.threshold_combo)
            detection_layout.addLayout(threshold_layout)
            
            detection_group.setLayout(detection_layout)
            layout.addWidget(detection_group)
            
            self.registerField("camera_id", self.camera_id_combo, "currentText")
            self.registerField("resolution", self.resolution_combo, "currentText")
            self.registerField("web_port", self.port_edit)
            self.registerField("detection_threshold", self.threshold_combo, "currentText")
            
            self.setLayout(layout)
    
    class InstallationPage(QWizardPage):
        def __init__(self):
            super().__init__()
            self.setTitle("Installing Baby Monitor System")
            
            layout = QVBoxLayout()
            
            self.status_label = QLabel("Preparing installation...")
            layout.addWidget(self.status_label)
            
            self.progress_bar = QProgressBar()
            layout.addWidget(self.progress_bar)
            
            self.details_label = QLabel("")
            self.details_label.setWordWrap(True)
            layout.addWidget(self.details_label)
            
            self.setLayout(layout)
            
            # Flag to track if installation is complete
            self.installation_complete = False
        
        def isComplete(self):
            # Only allow proceeding to the next page when installation is complete
            return self.installation_complete
        
        def initializePage(self):
            # Reset the completion flag
            self.installation_complete = False
            
            # Emit the completeChanged signal to update the Next button state
            self.completeChanged.emit()
            
            # Get options from previous pages
            options = {
                'download_models': self.field("download_models"),
                'create_shortcut': self.field("create_shortcut"),
                'camera_id': self.field("camera_id"),
                'resolution': self.field("resolution"),
                'web_port': self.field("web_port"),
                'detection_threshold': self.field("detection_threshold")
            }
            
            if is_raspberry_pi():
                options['setup_raspberry_pi'] = self.field("setup_raspberry_pi")
            
            # Update .env file with selected options
            threshold_map = {
                "0.3 (More sensitive)": "0.3",
                "0.5 (Balanced)": "0.5",
                "0.7 (Less sensitive)": "0.7"
            }
            
            # Start installation thread
            self.thread = SetupThread(options)
            self.thread.progress_update.connect(self.update_progress)
            self.thread.finished_signal.connect(self.installation_finished)
            self.thread.start()
        
        def update_progress(self, progress, message):
            self.progress_bar.setValue(progress)
            self.status_label.setText(message)
            self.details_label.setText(f"Current task: {message}")
        
        def installation_finished(self):
            self.progress_bar.setValue(100)
            self.status_label.setText("Installation completed successfully!")
            self.details_label.setText("You can now start the Baby Monitor System.")
            
            # Set the completion flag and update the Next button
            self.installation_complete = True
            self.completeChanged.emit()
    
    class CompletionPage(QWizardPage):
        def __init__(self):
            super().__init__()
            self.setTitle("Installation Complete")
            
            layout = QVBoxLayout()
            
            completion_text = QLabel(
                "The Baby Monitor System has been successfully installed!\n\n"
                "You can now start the system using one of the following methods:\n\n"
                "• Desktop shortcut (if created)\n"
                "• Command line: python main.py\n\n"
                "For more information, please refer to the README.md file."
            )
            completion_text.setWordWrap(True)
            layout.addWidget(completion_text)
            
            self.start_now_cb = QCheckBox("Start Baby Monitor System now")
            self.start_now_cb.setChecked(True)
            layout.addWidget(self.start_now_cb)
            
            self.registerField("start_now", self.start_now_cb)
            
            self.setLayout(layout)
    
    class InstallerWizard(QWizard):
        def __init__(self):
            super().__init__()
            
            self.setWindowTitle("Baby Monitor System Setup")
            self.setWizardStyle(QWizard.ModernStyle)
            
            # Set window size
            self.setMinimumSize(600, 500)
            
            # Add pages
            self.addPage(WelcomePage())
            self.addPage(ComponentsPage())
            self.addPage(ConfigurationPage())
            self.addPage(InstallationPage())
            self.addPage(CompletionPage())
            
            # Connect signals
            self.finished.connect(self.on_finished)
        
        def on_finished(self, result):
            if result == QWizard.Accepted and self.field("start_now"):
                # Start the Baby Monitor System
                if platform.system() == "Windows":
                    python_path = "venv\\Scripts\\python"
                else:
                    python_path = "venv/bin/python"
                
                subprocess.Popen([python_path, "main.py"])
    
    app = QApplication(sys.argv)
    wizard = InstallerWizard()
    wizard.show()
    sys.exit(app.exec_())

def main():
    """Main entry point for setup."""
    # Check Python version first
    check_python_version()
    
    parser = argparse.ArgumentParser(description="Baby Monitor System Setup")
    parser.add_argument("--no-gui", action="store_true", help="Run setup without GUI")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading models")
    parser.add_argument("--skip-shortcut", action="store_true", help="Skip creating desktop shortcut")
    
    args = parser.parse_args()
    
    # Create necessary directories
    for directory in [MODELS_DIR, LOGS_DIR, CONFIG_DIR, DATA_DIR, SCRIPTS_DIR]:
        directory.mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    try:
        # Setup environment
        python_path = setup_environment()
        
        # Setup models
        if not args.skip_models:
            setup_models()
        
        # Create utility scripts
        create_scripts()
        
        logger.info("Setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] != "install":
        # Run the setup script
        main()
    else:
        # Run setuptools setup when 'install' argument is provided
        setup(
            name="babymonitor",
            version="1.0.0",
            packages=find_packages(where="src"),
            package_dir={"": "src"},
            install_requires=[f"{pkg}=={ver}" for pkg, ver in REQUIRED_PACKAGES.items()],
            python_requires=f">={'.'.join(map(str, MIN_PYTHON))},<={'.'.join(map(str, MAX_PYTHON))}",
            entry_points={
                "console_scripts": [
                    "babymonitor=babymonitor.core.main:main",
                ],
            },
            include_package_data=True,
            package_data={
                "babymonitor": [
                    "detectors/models/*.pb",
                    "detectors/models/*.pbtxt",
                    "detectors/models/emotion/*.pt",
                    "web/templates/*",
                    "web/static/*"
                ]
            },
            author="Your Name",
            author_email="your.email@example.com",
            description="A baby monitoring system with person detection and emotion recognition",
            long_description=long_description,
            long_description_content_type="text/markdown",
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
        ) 