#!/usr/bin/env python3
"""
Baby Monitor System Manager
A unified script for installation, repair, and system management.
"""

import os
import sys
import platform
import subprocess
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('system_manager')

class SystemManager:
    def __init__(self):
        self.is_windows = platform.system().lower() == 'windows'
        self.is_raspberry_pi = os.path.exists('/proc/device-tree/model') and 'raspberry pi' in open('/proc/device-tree/model').read().lower()
        self.root_dir = Path(__file__).parent.parent
        self.venv_dir = self.root_dir / 'venv'

    def setup_environment(self):
        """Set up virtual environment and install base dependencies."""
        logger.info("Setting up environment...")
        if not self.venv_dir.exists():
            self._create_virtual_environment()
        self._install_base_requirements()

    def install_system(self):
        """Install the complete Baby Monitor system."""
        logger.info("Installing Baby Monitor system...")
        self.setup_environment()
        
        if self.is_raspberry_pi:
            self._install_raspberry_pi_dependencies()
        else:
            self._install_pc_dependencies()
        
        self._install_emotion_models()
        self._configure_system()

    def repair_system(self):
        """Repair common system issues."""
        logger.info("Repairing system...")
        self._fix_emotion_models()
        self._fix_metrics()
        self._verify_installation()

    def start_gui(self):
        """Launch the GUI manager."""
        logger.info("Starting GUI manager...")
        try:
            from scripts_manager_gui import main
            main()
        except ImportError:
            logger.error("GUI manager not available. Please install PyQt5.")
            sys.exit(1)

    def _create_virtual_environment(self):
        """Create a virtual environment."""
        logger.info("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', str(self.venv_dir)], check=True)

    def _install_base_requirements(self):
        """Install base requirements."""
        pip_cmd = str(self.venv_dir / ('Scripts' if self.is_windows else 'bin') / 'pip')
        subprocess.run([pip_cmd, 'install', '-r', str(self.root_dir / 'requirements.txt')], check=True)

    def _install_raspberry_pi_dependencies(self):
        """Install Raspberry Pi specific dependencies."""
        logger.info("Installing Raspberry Pi dependencies...")
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        subprocess.run(['sudo', 'apt-get', 'install', '-y',
                      'libatlas-base-dev',
                      'libjasper-dev',
                      'libqtgui4',
                      'libqt4-test',
                      'libhdf5-dev',
                      'libhdf5-serial-dev',
                      'libharfbuzz0b',
                      'libwebp6',
                      'libtiff5',
                      'libjasper1',
                      'libilmbase23',
                      'libopenexr23',
                      'libgstreamer1.0-0',
                      'libavcodec58',
                      'libavformat58',
                      'libswscale5'], check=True)

    def _install_pc_dependencies(self):
        """Install PC specific dependencies."""
        logger.info("Installing PC dependencies...")
        # Add any PC-specific dependencies here
        pass

    def _install_emotion_models(self):
        """Install and verify emotion detection models."""
        logger.info("Installing emotion models...")
        models_dir = self.root_dir / 'models'
        if not models_dir.exists():
            models_dir.mkdir(parents=True)
        # Add model installation logic here

    def _configure_system(self):
        """Configure system settings."""
        logger.info("Configuring system...")
        # Add system configuration logic here

    def _fix_emotion_models(self):
        """Fix emotion model issues."""
        logger.info("Fixing emotion models...")
        # Add emotion model repair logic here

    def _fix_metrics(self):
        """Fix metrics database issues."""
        logger.info("Fixing metrics...")
        # Add metrics repair logic here

    def _verify_installation(self):
        """Verify system installation."""
        logger.info("Verifying installation...")
        # Add verification logic here

def main():
    parser = argparse.ArgumentParser(description='Baby Monitor System Manager')
    parser.add_argument('action', choices=['install', 'repair', 'gui'],
                      help='Action to perform')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    manager = SystemManager()

    try:
        if args.action == 'install':
            manager.install_system()
        elif args.action == 'repair':
            manager.repair_system()
        elif args.action == 'gui':
            manager.start_gui()
    except Exception as e:
        logger.error(f"Error during {args.action}: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 