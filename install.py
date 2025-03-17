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
    --help           Show this help message
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

def is_raspberry_pi():
    """Check if the system is a Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return any("Raspberry Pi" in line for line in f)
    except:
        return False

def main():
    """Main entry point for the installer."""
    show_banner()
    
    parser = argparse.ArgumentParser(description="Baby Monitor System Installer")
    parser.add_argument("--no-gui", action="store_true", help="Run installation without GUI")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading models")
    parser.add_argument("--skip-shortcut", action="store_true", help="Skip creating desktop shortcut")
    
    args = parser.parse_args()
    
    # Determine the platform
    system = platform.system()
    print(f"Detected platform: {system}")
    
    if is_raspberry_pi():
        print("Detected Raspberry Pi hardware")
        
        # For Raspberry Pi, use the specialized script if GUI is not requested
        if args.no_gui:
            print("Running Raspberry Pi installation script...")
            try:
                subprocess.run(["bash", "install_pi.sh"], check=True)
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
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nInstallation cancelled by user.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 