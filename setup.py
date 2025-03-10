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
    name="babymonitor",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python",
        "numpy",
        "torch",
        "matplotlib",
        "pillow",
        "flask",
        "flask-socketio",
        "transformers",
        "sounddevice",
        "scipy",
        "librosa"
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