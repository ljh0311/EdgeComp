from setuptools import setup, find_packages
import os
from pathlib import Path

# Ensure models directory exists
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# List of required model files
REQUIRED_MODELS = [
    "yolov8n.pt",
    "emotion_model.pt",
    "audio_model.pt"
]

# Check for missing models
missing_models = [model for model in REQUIRED_MODELS if not (models_dir / model).exists()]
if missing_models:
    print("Warning: The following model files are missing and need to be downloaded:")
    for model in missing_models:
        print(f"  - {model}")

setup(
    name="edge_comp",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "transformers>=4.5.0",
        "librosa>=0.8.1",
        "scipy>=1.7.0",
        "PyAudio>=0.2.11",
        "sounddevice>=0.4.4",
        "opencv-python>=4.5.3",
        "python-socketio>=5.4.0",
        "Flask>=2.0.1",
        "Flask-SocketIO>=5.1.1",
        "eventlet>=0.33.0",
        "pyzmq>=22.3.0",
        "ultralytics>=8.0.0",  # For YOLOv8
        "scikit-learn>=0.24.2",  # For audio processing
        "matplotlib>=3.4.3",  # For visualization
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "edge_comp": ["models/*.pt", "models/*.pth"],
    },
    # Add entry points if needed
    entry_points={
        "console_scripts": [
            "edge-comp=edge_comp.main:main",
        ],
    },
    # Add metadata
    author="Your Name",
    author_email="your.email@example.com",
    description="Edge computing application for baby monitoring",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="edge-computing, baby-monitor, yolo, emotion-detection",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 