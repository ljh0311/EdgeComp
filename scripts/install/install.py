#!/usr/bin/env python3
"""
Baby Monitor System - Installation Script
=========================================

This script serves as the main entry point for installing the Baby Monitor System.
It detects the platform and launches the appropriate installation method.

Usage:
    python install.py [options]

Options:
    --no-gui         Run installation without GUI
    --skip-models    Skip downloading models
    --skip-shortcut  Skip creating desktop shortcut
    --help          Show this help message
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path

def show_banner():
    """Display a welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║             BABY MONITOR SYSTEM INSTALLER                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    
    A comprehensive baby monitoring solution with real-time
    person detection, emotion recognition, and web interface.
    """
    print(banner)

def check_python_version():
    """Check if the current Python version is supported."""
    current_version = sys.version_info[:3]
    if current_version > (3, 11, 5):
        print("ERROR: Python version not supported!")
        print(f"Current version: Python {'.'.join(map(str, current_version))}")
        print("Please install Python 3.11.5 or earlier.")
        print("Download Python 3.11.5 from: https://www.python.org/downloads/release/python-3115/")
        sys.exit(1)
    elif current_version < (3, 8):
        print("ERROR: Python version too old!")
        print(f"Current version: Python {'.'.join(map(str, current_version))}")
        print("Minimum required version is Python 3.8")
        sys.exit(1)
    
    print(f"Python version check passed: {'.'.join(map(str, current_version))}")

def is_raspberry_pi():
    """Check if the system is a Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return any("Raspberry Pi" in line for line in f)
    except:
        return False

def check_dependencies():
    """Check if required system dependencies are installed."""
    system = platform.system()
    
    if system == "Windows":
        # Check for Visual C++ Redistributable
        if not os.path.exists("C:\\Windows\\System32\\vcruntime140.dll"):
            print("WARNING: Visual C++ Redistributable might be missing.")
            print("Please download and install from: https://aka.ms/vs/16/release/vc_redist.x64.exe")
    
    elif system == "Linux":
        # Check for common Linux dependencies
        dependencies = [
            "python3-dev",
            "python3-pip",
            "python3-venv",
            "libportaudio2",
            "portaudio19-dev",
            "libsndfile1"
        ]
        
        if is_raspberry_pi():
            dependencies.extend([
                "python3-picamera2",
                "libopencv-dev",
                "python3-opencv"
            ])
        
        missing = []
        for dep in dependencies:
            if subprocess.call(["dpkg", "-s", dep], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
                missing.append(dep)
        
        if missing:
            print("Missing system dependencies:", ", ".join(missing))
            print("Please install them using:")
            print(f"sudo apt-get install {' '.join(missing)}")
            return False
    
    return True

def main():
    """Main entry point for the installer."""
    show_banner()
    
    # Check Python version first
    check_python_version()
    
    parser = argparse.ArgumentParser(description="Baby Monitor System Installer")
    parser.add_argument("--no-gui", action="store_true", help="Run installation without GUI")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading models")
    parser.add_argument("--skip-shortcut", action="store_true", help="Skip creating desktop shortcut")
    
    args = parser.parse_args()
    
    # Determine the platform
    system = platform.system()
    print(f"Detected platform: {system}")
    
    # Check system dependencies
    if not check_dependencies():
        print("Please install the required dependencies and try again.")
        return 1
    
    if is_raspberry_pi():
        print("Detected Raspberry Pi hardware")
        
        # For Raspberry Pi, use the specialized script if GUI is not requested
        if args.no_gui:
            print("Running Raspberry Pi installation script...")
            try:
                subprocess.run(["bash", "scripts/install_pi.sh"], check=True)
                print("Raspberry Pi installation completed successfully!")
                return 0
            except subprocess.CalledProcessError as e:
                print(f"Error during Raspberry Pi installation: {e}")
                return 1
    
    # For all other cases, use the setup.py script
    cmd = [sys.executable, "setup.py"]
    
    if args.no_gui:
        cmd.append("--no-gui")
    
    if args.skip_models:
        cmd.append("--skip-models")
    
    if args.skip_shortcut:
        cmd.append("--skip-shortcut")
    
    print(f"Running setup: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Installation completed successfully!")
        
        # Print next steps
        print("\nNext steps:")
        if system == "Windows":
            print("1. Run 'scripts\\fix_windows.bat' to start the application")
        else:
            print("1. Run 'python run_monitor.py' to start the application")
        print("2. Open http://localhost:5000 in your web browser")
        print("3. Check the README.md file for more information")
        
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nInstallation cancelled by user.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 