"""
Baby Monitor System
==================
Main setup script that redirects to the comprehensive installer.
"""

import sys
import os
from pathlib import Path

# Redirect to comprehensive installer
if __name__ == "__main__":
    print("Baby Monitor System - Installation")
    print("===================================")
    print("Redirecting to comprehensive installer...")
    
    # Check if the install script exists
    install_script = Path("scripts/install/setup.py")
    if install_script.exists():
        # Execute the comprehensive installer
        print("Found comprehensive installer at scripts/install/setup.py")
        print("Executing comprehensive installer...\n")
        # Get the same arguments that were passed to this script
        args = sys.argv[1:]
        arg_str = " ".join(args)
        os.system(f"python {install_script} {arg_str}")
    else:
        # Fall back to simplified setup
        print("Comprehensive installer not found, falling back to basic setup...")
        from setuptools import setup, find_packages

        setup(
            name="babymonitor",
            version="1.0.0",
            packages=find_packages(where="src"),
            package_dir={"": "src"},
            include_package_data=True,
            install_requires=[
                "flask>=2.3.3",
                "flask-socketio>=5.3.6",
                "opencv-python-headless>=4.8.1.78",
                "numpy>=1.24.3",
                "torch>=1.13.1",
                "sounddevice>=0.4.6",
                "python-dotenv>=1.0.0",
                "psutil>=5.9.5",
                "eventlet>=0.33.3",
                "werkzeug>=2.3.7",
                "jsonschema>=4.17.3"
            ],
            entry_points={
                "console_scripts": [
                    "babymonitor=src.babymonitor.core.main:main",
                ],
            },
            author="Your Name",
            author_email="your.email@example.com",
            description="A baby monitoring system with person detection and emotion recognition",
            python_requires=">=3.8.0,<3.12.0",
        )
        
        print("\nBasic setup completed. For a more comprehensive setup with GUI installer,")
        print("models downloading, and configuration, please run:")
        print("python scripts/install/install.py")
