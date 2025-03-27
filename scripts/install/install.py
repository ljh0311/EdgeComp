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
    --mode [normal|dev]  Set the operation mode (normal for regular users, dev for developers)
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
    if current_version > (3, 12, 99):
        print("ERROR: Python version not supported!")
        print(f"Current version: Python {'.'.join(map(str, current_version))}")
        print("Please install Python 3.8 through 3.12")
        print(
            "Download Python 3.11.5 from: https://www.python.org/downloads/release/python-3115/"
        )
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
        with open("/proc/cpuinfo", "r") as f:
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
            print(
                "Please download and install from: https://aka.ms/vs/16/release/vc_redist.x64.exe"
            )

    elif system == "Linux":
        # Check for common Linux dependencies
        dependencies = [
            "python3-dev",
            "python3-pip",
            "python3-venv",
            "libportaudio2",
            "portaudio19-dev",
            "libsndfile1",
        ]

        if is_raspberry_pi():
            dependencies.extend(
                ["python3-picamera2", "libopencv-dev", "python3-opencv"]
            )

        missing = []
        for dep in dependencies:
            if (
                subprocess.call(
                    ["dpkg", "-s", dep],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                != 0
            ):
                missing.append(dep)

        if missing:
            print("Missing system dependencies:", ", ".join(missing))
            print("Please install them using:")
            print(f"sudo apt-get install {' '.join(missing)}")
            return False

    return True


def run_gui_installer():
    """Run the GUI installer."""
    try:
        from PyQt5.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QVBoxLayout,
            QLabel,
            QPushButton,
            QCheckBox,
            QComboBox,
            QGroupBox,
            QHBoxLayout,
            QMessageBox,
            QProgressBar,
        )
        from PyQt5.QtGui import QFont, QIcon, QPixmap
        from PyQt5.QtCore import Qt, QThread, pyqtSignal
    except ImportError:
        print("PyQt5 not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "PyQt5"])
        from PyQt5.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QVBoxLayout,
            QLabel,
            QPushButton,
            QCheckBox,
            QComboBox,
            QGroupBox,
            QHBoxLayout,
            QMessageBox,
            QProgressBar,
        )
        from PyQt5.QtGui import QFont, QIcon, QPixmap
        from PyQt5.QtCore import Qt, QThread, pyqtSignal

    class InstallationThread(QThread):
        """Thread for running the installation process."""

        update_signal = pyqtSignal(str)
        progress_signal = pyqtSignal(int)
        finished_signal = pyqtSignal(int)

        def __init__(self, options):
            super().__init__()
            self.options = options

        def run(self):
            """Run the installation process."""
            self.update_signal.emit("Starting installation...")
            self.progress_signal.emit(10)

            # Build setup.py command
            cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "setup.py")]

            # Add options
            if self.options.get("skip_gui", False):
                cmd.append("--no-gui")

            if self.options.get("skip_models", False):
                cmd.append("--skip-models")

            if self.options.get("skip_shortcut", False):
                cmd.append("--skip-shortcut")

            if "mode" in self.options:
                cmd.extend(["--mode", self.options["mode"]])

            self.update_signal.emit(f"Running command: {' '.join(cmd)}")
            self.progress_signal.emit(30)

            # Execute the setup.py command
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                )

                # Capture and emit output
                progress = 30
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break

                    self.update_signal.emit(line.strip())

                    # Update progress based on key messages in the output
                    if "Creating directories" in line:
                        progress = 40
                    elif "Setting up environment" in line:
                        progress = 60
                    elif "Downloading models" in line:
                        progress = 80
                    elif "Setup completed" in line:
                        progress = 95

                    self.progress_signal.emit(progress)

                # Wait for process to finish
                process.wait()

                if process.returncode == 0:
                    self.update_signal.emit("Installation completed successfully!")
                    self.progress_signal.emit(100)
                    self.finished_signal.emit(0)
                else:
                    self.update_signal.emit(
                        f"Installation failed with code {process.returncode}"
                    )
                    self.finished_signal.emit(process.returncode)

            except Exception as e:
                self.update_signal.emit(f"Error during installation: {e}")
                self.finished_signal.emit(1)

    class InstallerWindow(QMainWindow):
        """Main window for the installer GUI."""

        def __init__(self):
            super().__init__()

            self.setWindowTitle("Baby Monitor System Installer")
            self.setMinimumSize(700, 500)

            # Set up the central widget and layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)

            main_layout = QVBoxLayout(central_widget)

            # Header section
            header_layout = QVBoxLayout()

            # Logo (if available)
            logo_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "src",
                "babymonitor",
                "web",
                "static",
                "img",
                "logo.png",
            )
            if os.path.exists(logo_path):
                logo_label = QLabel()
                pixmap = QPixmap(logo_path)
                scaled_pixmap = pixmap.scaled(
                    200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                logo_label.setPixmap(scaled_pixmap)
                logo_label.setAlignment(Qt.AlignCenter)
                header_layout.addWidget(logo_label)

            # Title
            title_label = QLabel("Baby Monitor System Installer")
            title_font = QFont()
            title_font.setPointSize(16)
            title_font.setBold(True)
            title_label.setFont(title_font)
            title_label.setAlignment(Qt.AlignCenter)
            header_layout.addWidget(title_label)

            # Subtitle
            subtitle_label = QLabel(
                "A comprehensive baby monitoring solution with AI features"
            )
            subtitle_label.setAlignment(Qt.AlignCenter)
            header_layout.addWidget(subtitle_label)

            main_layout.addLayout(header_layout)

            # System information section
            info_group = QGroupBox("System Information")
            info_layout = QVBoxLayout(info_group)

            platform_text = f"Platform: {platform.system()} {platform.release()}"
            platform_label = QLabel(platform_text)
            info_layout.addWidget(platform_label)

            python_text = f"Python: {platform.python_version()}"
            python_label = QLabel(python_text)
            info_layout.addWidget(python_label)

            if is_raspberry_pi():
                pi_label = QLabel("Hardware: Raspberry Pi detected")
                info_layout.addWidget(pi_label)

            main_layout.addWidget(info_group)

            # Installation options section
            options_group = QGroupBox("Installation Options")
            options_layout = QVBoxLayout(options_group)

            # Installation type
            type_layout = QHBoxLayout()
            type_label = QLabel("Installation Type:")
            type_layout.addWidget(type_label)

            self.install_type_combo = QComboBox()
            self.install_type_combo.addItems(
                ["Full Installation", "Minimal Installation"]
            )
            type_layout.addWidget(self.install_type_combo)
            options_layout.addLayout(type_layout)

            # Operation mode
            mode_layout = QHBoxLayout()
            mode_label = QLabel("Operation Mode:")
            mode_layout.addWidget(mode_label)

            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Normal Mode", "Developer Mode"])
            mode_layout.addWidget(self.mode_combo)
            options_layout.addLayout(mode_layout)

            # Options
            self.skip_models_checkbox = QCheckBox(
                "Skip downloading detection models (faster installation)"
            )
            options_layout.addWidget(self.skip_models_checkbox)

            self.skip_shortcut_checkbox = QCheckBox("Skip creating desktop shortcut")
            options_layout.addWidget(self.skip_shortcut_checkbox)

            main_layout.addWidget(options_group)

            # Progress section (initially hidden)
            self.progress_group = QGroupBox("Installation Progress")
            progress_layout = QVBoxLayout(self.progress_group)

            self.progress_bar = QProgressBar()
            progress_layout.addWidget(self.progress_bar)

            self.status_label = QLabel("Ready to install")
            self.status_label.setWordWrap(True)
            progress_layout.addWidget(self.status_label)

            self.progress_group.hide()  # Initially hidden
            main_layout.addWidget(self.progress_group)

            # Buttons section
            buttons_layout = QHBoxLayout()

            help_button = QPushButton("Help")
            help_button.clicked.connect(self.show_help)
            buttons_layout.addWidget(help_button)

            buttons_layout.addStretch()

            exit_button = QPushButton("Exit")
            exit_button.clicked.connect(self.close)
            buttons_layout.addWidget(exit_button)

            self.install_button = QPushButton("Install")
            self.install_button.clicked.connect(self.start_installation)
            buttons_layout.addWidget(self.install_button)

            main_layout.addLayout(buttons_layout)

            # Set window style
            self.setStyleSheet(
                """
                QMainWindow {
                    background-color: #f5f5f5;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #cccccc;
                    border-radius: 5px;
                    margin-top: 1ex;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                    padding: 0 3px;
                }
                QPushButton {
                    background-color: #0078d7;
                    color: white;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #0063b1;
                }
                QPushButton:pressed {
                    background-color: #004c8c;
                }
                QProgressBar {
                    border: 1px solid #cccccc;
                    border-radius: 5px;
                    text-align: center;
                    height: 20px;
                }
                QProgressBar::chunk {
                    background-color: #0078d7;
                    width: 10px;
                    margin: 0.5px;
                }
            """
            )

        def show_help(self):
            """Show help message."""
            help_text = """
<h3>Baby Monitor System Installer Help</h3>

<p><b>Installation Types:</b></p>
<ul>
<li><b>Full Installation:</b> Installs all components including models and desktop shortcuts.</li>
<li><b>Minimal Installation:</b> Installs only the essential components.</li>
</ul>

<p><b>Operation Modes:</b></p>
<ul>
<li><b>Normal Mode:</b> Standard interface for regular users.</li>
<li><b>Developer Mode:</b> Advanced features and debugging options for developers.</li>
</ul>

<p><b>Options:</b></p>
<ul>
<li><b>Skip downloading models:</b> Don't download the detection models (faster installation but requires manual download later).</li>
<li><b>Skip creating desktop shortcut:</b> Don't create a desktop shortcut for the application.</li>
</ul>

<p><b>System Requirements:</b></p>
<ul>
<li>Python 3.8 - 3.11</li>
<li>Internet connection (for model downloads)</li>
<li>Camera device (webcam or Raspberry Pi camera)</li>
</ul>

<p>For more information, please see the README.md file in the project root directory.</p>
"""

            QMessageBox.information(self, "Installation Help", help_text)

        def update_status(self, message):
            """Update the status label."""
            self.status_label.setText(message)

        def update_progress(self, value):
            """Update the progress bar."""
            self.progress_bar.setValue(value)

        def installation_finished(self, exit_code):
            """Handle installation completion."""
            self.install_button.setEnabled(True)
            self.install_button.setText("Install")

            if exit_code == 0:
                QMessageBox.information(
                    self,
                    "Installation Complete",
                    "Installation completed successfully!\n\n"
                    "You can now start the Baby Monitor System using:\n"
                    "• Desktop shortcut (if created)\n"
                    "• Start script in the project root directory",
                )
            else:
                QMessageBox.critical(
                    self,
                    "Installation Failed",
                    f"Installation failed with exit code {exit_code}.\n"
                    "Please check the logs for more information.",
                )

        def start_installation(self):
            """Start the installation process."""
            # Get options
            mode = (
                "dev" if self.mode_combo.currentText() == "Developer Mode" else "normal"
            )

            options = {
                "skip_gui": False,  # Always use GUI when launching from GUI
                "skip_models": self.skip_models_checkbox.isChecked(),
                "skip_shortcut": self.skip_shortcut_checkbox.isChecked(),
                "mode": mode,
            }

            # Adjust options based on installation type
            if self.install_type_combo.currentText() == "Minimal Installation":
                options["skip_models"] = True
                options["skip_shortcut"] = True

            # Check Python version first
            try:
                check_python_version()
            except SystemExit:
                QMessageBox.critical(
                    self,
                    "Python Version Error",
                    "Unsupported Python version. Please use Python 3.8 - 3.11.",
                )
                return

            # Check dependencies
            if not check_dependencies():
                result = QMessageBox.warning(
                    self,
                    "Missing Dependencies",
                    "Some system dependencies are missing. "
                    "The installation may fail.\n\n"
                    "Do you want to continue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                )

                if result == QMessageBox.No:
                    return

            # Show progress group
            self.progress_group.show()
            self.progress_bar.setValue(0)
            self.update_status("Preparing installation...")

            # Disable install button
            self.install_button.setEnabled(False)
            self.install_button.setText("Installing...")

            # Create and start thread
            self.install_thread = InstallationThread(options)
            self.install_thread.update_signal.connect(self.update_status)
            self.install_thread.progress_signal.connect(self.update_progress)
            self.install_thread.finished_signal.connect(self.installation_finished)
            self.install_thread.start()

    def launch_gui():
        """Launch the GUI installer."""
        app = QApplication(sys.argv)
        window = InstallerWindow()
        window.show()
        return app.exec_()


def main():
    """Main entry point for the installer."""
    show_banner()

    # Check Python version first
    check_python_version()

    parser = argparse.ArgumentParser(description="Baby Monitor System Installer")
    parser.add_argument(
        "--no-gui", action="store_true", help="Run installation without GUI"
    )
    parser.add_argument(
        "--skip-models", action="store_true", help="Skip downloading models"
    )
    parser.add_argument(
        "--skip-shortcut", action="store_true", help="Skip creating desktop shortcut"
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "dev"],
        default="normal",
        help="Operation mode (normal or dev)",
    )

    args = parser.parse_args()

    # Determine the platform
    system = platform.system()
    print(f"Detected platform: {system}")

    # Check system dependencies
    if not check_dependencies():
        print("Please install the required dependencies and try again.")
        return 1

    # If --no-gui is specified or we're in a non-interactive environment, use command line
    if args.no_gui or not sys.stdout.isatty():
        # For all platforms, use the setup.py script directly
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "setup.py")]

        if args.no_gui:
            cmd.append("--no-gui")

        if args.skip_models:
            cmd.append("--skip-models")

        if args.skip_shortcut:
            cmd.append("--skip-shortcut")

        if args.mode:
            cmd.extend(["--mode", args.mode])

        print(f"Running setup: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            print("Installation completed successfully!")

            # Print next steps
            print("\nNext steps:")
            if system == "Windows":
                print(
                    f"1. Run 'venv\\Scripts\\python -m src.run_server --mode {args.mode}' to start the application"
                )
            else:
                print(
                    f"1. Run 'venv/bin/python -m src.run_server --mode {args.mode}' to start the application"
                )
            print("2. Open http://localhost:5000 in your web browser")
            print("3. Check the README.md file for more information")

            return 0
        except subprocess.CalledProcessError as e:
            print(f"Error during installation: {e}")
            return 1
        except KeyboardInterrupt:
            print("\nInstallation cancelled by user.")
            return 1
    else:
        # Launch the GUI installer
        return launch_gui()


if __name__ == "__main__":
    sys.exit(main())
