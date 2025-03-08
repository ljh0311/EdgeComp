from setuptools import setup, find_packages

setup(
    name="edge_comp",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19.0,<2.1.0",
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "transformers>=4.5.0,<5.0.0",
        "librosa>=0.8.1",
        "scipy>=1.7.0",
        "PyAudio>=0.2.11",
        "sounddevice>=0.4.4",
        "opencv-python>=4.5.3,<4.11.0",
        "python-socketio>=5.4.0",
        "Flask>=2.0.1",
        "Flask-SocketIO>=5.1.1",
        "eventlet>=0.33.0",
        "pyzmq>=22.3.0",
        "ultralytics>=8.0.0",
        "huggingface-hub>=0.26.0,<1.0.0",
    ],
    python_requires=">=3.8",
) 