from setuptools import setup, find_packages
import os
from pathlib import Path
import urllib.request
import sys
import subprocess
import platform

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
        # Note: emotion_model.pt and audio_model.pt need to be downloaded separately
        # as they are custom trained models
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

def main():
    """Main setup function."""
    print("Starting Baby Monitor System setup...")
    
    # Create necessary directories
    for dir_name in ["models", "logs", "data", "config"]:
        Path(dir_name).mkdir(exist_ok=True)

    # Setup environment
    setup_environment()
    
    # Download and setup models
    setup_models()

    # Continue with package setup
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
            "ultralytics>=8.0.0",  # Added for YOLOv8
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