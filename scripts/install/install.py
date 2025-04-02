#!/usr/bin/env python
"""
Baby Monitor System - Installation Wizard
========================================
This script provides a user-friendly interface for installing the Baby Monitor System,
including options for model training and downloading.

Version 2.1.0 - Added person state detection functionality
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path
import time
import shutil

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Version information
VERSION = "2.1.0"
VERSION_NAME = "State Detection"

# Team information
TEAM_INFO = [
    ("JunHong", "Backend Processing & Client Logic"),
    ("Darrel", "Dashboard Frontend"),
    ("Ashraf", "Datasets & Model Architecture"),
    ("Xuan Yu", "Specialized Datasets & Training"),
    ("Javin", "Camera Detection System")
]

def check_models():
    """Check if required models are available."""
    model_paths = {
        "emotion_model": "src/babymonitor/models/emotion_model.pt",
        "person_detection": "src/babymonitor/models/yolov8n.pt",
        "wav2vec2_model": "src/babymonitor/models/wav2vec2_emotion.pt",
        "person_state_classifier": "src/babymonitor/models/person_state_classifier.pkl"
    }
    
    missing_models = []
    existing_models = []
    
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
            existing_models.append((model_name, f"{model_size:.1f} MB"))
        else:
            missing_models.append(model_name)
    
    return existing_models, missing_models

class InstallerApp:
    """GUI installer for the Baby Monitor System."""
    
    def __init__(self, root):
        """Initialize the installer app."""
        self.root = root
        self.root.title(f"Baby Monitor System - Installation Wizard v{VERSION}")
        
        # Set window size
        window_width = 700
        window_height = 500
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.welcome_tab = ttk.Frame(self.notebook)
        self.models_tab = ttk.Frame(self.notebook)
        self.install_tab = ttk.Frame(self.notebook)
        self.config_tab = ttk.Frame(self.notebook)
        self.team_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.welcome_tab, text="Welcome")
        self.notebook.add(self.models_tab, text="Models")
        self.notebook.add(self.install_tab, text="Installation")
        self.notebook.add(self.config_tab, text="Configuration")
        self.notebook.add(self.team_tab, text="Team")
        
        # Setup each tab
        self.setup_welcome_tab()
        self.setup_models_tab()
        self.setup_install_tab()
        self.setup_config_tab()
        self.setup_team_tab()
        
        # Start on the appropriate tab based on command line arguments
        if "--models" in sys.argv:
            self.notebook.select(self.models_tab)
        elif "--install" in sys.argv:
            self.notebook.select(self.install_tab)
        elif "--config" in sys.argv:
            self.notebook.select(self.config_tab)
    
    def setup_welcome_tab(self):
        """Set up the welcome tab."""
        welcome_frame = ttk.Frame(self.welcome_tab, padding=20)
        welcome_frame.pack(fill='both', expand=True)
        
        # Logo or title
        ttk.Label(
            welcome_frame, 
            text="Baby Monitor System",
            font=("Arial", 24, "bold")
        ).pack(pady=10)
        
        ttk.Label(
            welcome_frame, 
            text=f"Version {VERSION} - {VERSION_NAME}",
            font=("Arial", 12)
        ).pack(pady=5)
        
        # Welcome message
        ttk.Label(
            welcome_frame,
            text="Welcome to the Baby Monitor System installation wizard.\n\n"
                 "This wizard will guide you through the process of setting up the Baby Monitor System, "
                 "including managing models and configuring your system.\n\n"
                 "New in this version: Person state detection capability, allowing the system to "
                 "identify whether a person is seated, lying, moving, or standing.",
            wraplength=600,
            justify="center"
        ).pack(pady=10)
        
        # System information
        sys_info = f"System Information:\n"
        sys_info += f"• Operating System: {platform.system()} {platform.version()}\n"
        sys_info += f"• Python Version: {platform.python_version()}\n"
        sys_info += f"• Processor: {platform.processor()}"
        
        sys_info_frame = ttk.LabelFrame(welcome_frame, text="System Information")
        sys_info_frame.pack(fill="x", pady=20)
        ttk.Label(sys_info_frame, text=sys_info, justify="left", padding=10).pack(fill="x")
        
        # Navigation buttons
        button_frame = ttk.Frame(welcome_frame)
        button_frame.pack(fill="x", pady=20)
        
        ttk.Button(
            button_frame, 
            text="Next: Check Models",
            command=lambda: self.notebook.select(self.models_tab)
        ).pack(side="right")
    
    def setup_models_tab(self):
        """Set up the models tab."""
        models_frame = ttk.Frame(self.models_tab, padding=20)
        models_frame.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(
            models_frame, 
            text="Model Management",
            font=("Arial", 18, "bold")
        ).pack(pady=10)
        
        # Model status frame
        status_frame = ttk.LabelFrame(models_frame, text="Model Status")
        status_frame.pack(fill="x", pady=10)
        
        # Check model status
        existing_models, missing_models = check_models()
        
        if existing_models:
            ttk.Label(
                status_frame, 
                text="Available Models:",
                font=("Arial", 11, "bold")
            ).pack(anchor="w", padx=10, pady=5)
            
            for model_name, model_size in existing_models:
                ttk.Label(
                    status_frame,
                    text=f"✓ {model_name} ({model_size})",
                    foreground="green"
                ).pack(anchor="w", padx=20, pady=2)
        
        if missing_models:
            ttk.Label(
                status_frame, 
                text="Missing Models:",
                font=("Arial", 11, "bold")
            ).pack(anchor="w", padx=10, pady=5)
            
            for model_name in missing_models:
                ttk.Label(
                    status_frame,
                    text=f"✗ {model_name}",
                    foreground="red"
                ).pack(anchor="w", padx=20, pady=2)
        
        # Model management frame
        manage_frame = ttk.LabelFrame(models_frame, text="Model Actions")
        manage_frame.pack(fill="x", pady=10)
        
        # Action buttons
        ttk.Button(
            manage_frame,
            text="Download Pretrained Models",
            command=self.download_models
        ).pack(fill="x", padx=10, pady=5)
        
        ttk.Button(
            manage_frame,
            text="Train Models",
            command=self.train_models
        ).pack(fill="x", padx=10, pady=5)
        
        ttk.Button(
            manage_frame,
            text="Refresh Model Status",
            command=self.refresh_models_tab
        ).pack(fill="x", padx=10, pady=5)
        
        # Navigation buttons
        button_frame = ttk.Frame(models_frame)
        button_frame.pack(fill="x", pady=20)
        
        ttk.Button(
            button_frame, 
            text="Back",
            command=lambda: self.notebook.select(self.welcome_tab)
        ).pack(side="left")
        
        ttk.Button(
            button_frame, 
            text="Next: Install System",
            command=lambda: self.notebook.select(self.install_tab)
        ).pack(side="right")
    
    def setup_install_tab(self):
        """Set up the installation tab."""
        install_frame = ttk.Frame(self.install_tab, padding=20)
        install_frame.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(
            install_frame, 
            text="Installation",
            font=("Arial", 18, "bold")
        ).pack(pady=10)
        
        # Installation options
        options_frame = ttk.LabelFrame(install_frame, text="Installation Options")
        options_frame.pack(fill="x", pady=10)
        
        # Installation mode
        ttk.Label(
            options_frame,
            text="Installation Mode:"
        ).pack(anchor="w", padx=10, pady=5)
        
        self.install_mode = tk.StringVar(value="normal")
        ttk.Radiobutton(
            options_frame,
            text="Normal Mode",
            variable=self.install_mode,
            value="normal"
        ).pack(anchor="w", padx=20, pady=2)
        
        ttk.Radiobutton(
            options_frame,
            text="Developer Mode (includes additional testing tools)",
            variable=self.install_mode,
            value="dev"
        ).pack(anchor="w", padx=20, pady=2)
        
        # State detection option
        ttk.Label(
            options_frame,
            text="Person State Detection:"
        ).pack(anchor="w", padx=10, pady=5)
        
        self.state_detection = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Enable person state detection (seated, lying, moving, standing)",
            variable=self.state_detection
        ).pack(anchor="w", padx=20, pady=2)
        
        # Installation location
        location_frame = ttk.Frame(options_frame)
        location_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(
            location_frame,
            text="Installation Location:"
        ).pack(side="left", padx=5)
        
        self.install_dir = tk.StringVar(value=os.getcwd())
        ttk.Entry(
            location_frame,
            textvariable=self.install_dir,
            width=40
        ).pack(side="left", padx=5)
        
        ttk.Button(
            location_frame,
            text="Browse...",
            command=self.browse_install_dir
        ).pack(side="left", padx=5)
        
        # Install button
        ttk.Button(
            install_frame,
            text="Install Baby Monitor System",
            command=self.install_system
        ).pack(fill="x", padx=10, pady=20)
        
        # Installation status
        self.install_status = ttk.Label(
            install_frame,
            text="Ready to install.",
            wraplength=600,
            justify="left"
        )
        self.install_status.pack(fill="x", pady=10)
        
        # Navigation buttons
        button_frame = ttk.Frame(install_frame)
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(
            button_frame, 
            text="Back",
            command=lambda: self.notebook.select(self.models_tab)
        ).pack(side="left")
        
        ttk.Button(
            button_frame, 
            text="Next: Configuration",
            command=lambda: self.notebook.select(self.config_tab)
        ).pack(side="right")
    
    def setup_config_tab(self):
        """Set up the configuration tab."""
        config_frame = ttk.Frame(self.config_tab, padding=20)
        config_frame.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(
            config_frame, 
            text="Configuration",
            font=("Arial", 18, "bold")
        ).pack(pady=10)
        
        # Configuration options
        options_frame = ttk.LabelFrame(config_frame, text="System Configuration")
        options_frame.pack(fill="x", pady=10)
        
        # Launch options
        self.auto_start = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Launch Baby Monitor automatically at startup",
            variable=self.auto_start
        ).pack(anchor="w", padx=10, pady=5)
        
        self.create_shortcut = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Create desktop shortcut",
            variable=self.create_shortcut
        ).pack(anchor="w", padx=10, pady=5)
        
        # Camera options
        camera_frame = ttk.LabelFrame(config_frame, text="Camera Configuration")
        camera_frame.pack(fill="x", pady=10)
        
        # Resolution options
        ttk.Label(
            camera_frame,
            text="Camera Resolution:"
        ).pack(anchor="w", padx=10, pady=5)
        
        self.camera_resolution = tk.StringVar(value="640x480")
        resolution_frame = ttk.Frame(camera_frame)
        resolution_frame.pack(fill="x", padx=20, pady=2)
        
        resolutions = ["320x240", "640x480", "800x600", "1280x720"]
        for resolution in resolutions:
            ttk.Radiobutton(
                resolution_frame,
                text=resolution,
                variable=self.camera_resolution,
                value=resolution
            ).pack(side="left", padx=10)
        
        # FPS setting
        fps_frame = ttk.Frame(camera_frame)
        fps_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(
            fps_frame,
            text="Camera FPS:"
        ).pack(side="left", padx=5)
        
        self.camera_fps = tk.StringVar(value="15")
        ttk.Spinbox(
            fps_frame,
            from_=5,
            to=30,
            textvariable=self.camera_fps,
            width=5
        ).pack(side="left", padx=5)
        
        # System optimization
        if platform.system() == "Linux":
            optimization_frame = ttk.LabelFrame(config_frame, text="System Optimization")
            optimization_frame.pack(fill="x", pady=10)
            
            ttk.Button(
                optimization_frame,
                text="Optimize System Performance",
                command=self.optimize_system
            ).pack(fill="x", padx=10, pady=5)
            
            ttk.Label(
                optimization_frame,
                text="Note: System optimization requires root privileges and may require a restart.",
                wraplength=600,
                foreground="gray"
            ).pack(anchor="w", padx=10, pady=5)
        
        # Apply configuration
        ttk.Button(
            config_frame,
            text="Apply Configuration",
            command=self.apply_config
        ).pack(fill="x", padx=10, pady=20)
        
        # Configuration status
        self.config_status = ttk.Label(
            config_frame,
            text="Ready to configure.",
            wraplength=600,
            justify="left"
        )
        self.config_status.pack(fill="x", pady=10)
        
        # Navigation buttons
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(
            button_frame, 
            text="Back",
            command=lambda: self.notebook.select(self.install_tab)
        ).pack(side="left")
        
        ttk.Button(
            button_frame, 
            text="Next: Team Info",
            command=lambda: self.notebook.select(self.team_tab)
        ).pack(side="right")
    
    def setup_team_tab(self):
        """Set up the team information tab."""
        team_frame = ttk.Frame(self.team_tab, padding=20)
        team_frame.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(
            team_frame, 
            text="Development Team",
            font=("Arial", 18, "bold")
        ).pack(pady=10)
        
        # Description
        ttk.Label(
            team_frame,
            text="This Baby Monitor System was developed by the following team members:",
            wraplength=600,
            justify="center"
        ).pack(pady=10)
        
        # Team info frame
        info_frame = ttk.Frame(team_frame)
        info_frame.pack(fill="x", pady=10)
        
        # Create a borderless table for team info
        for i, (name, role) in enumerate(TEAM_INFO):
            row_frame = ttk.Frame(info_frame)
            row_frame.pack(fill="x", pady=5)
            
            # Name column
            name_label = ttk.Label(
                row_frame,
                text=name,
                font=("Arial", 11, "bold"),
                width=15,
                anchor="w"
            )
            name_label.pack(side="left", padx=10)
            
            # Role column
            role_label = ttk.Label(
                row_frame,
                text=role,
                wraplength=500,
                justify="left"
            )
            role_label.pack(side="left", padx=10, fill="x", expand=True)
        
        # Closing note
        ttk.Label(
            team_frame,
            text=f"\nBaby Monitor System v{VERSION} - {VERSION_NAME}\n"
                "Thank you for using our software!",
            wraplength=600,
            justify="center",
            font=("Arial", 10, "italic")
        ).pack(pady=20)
        
        # Navigation buttons
        button_frame = ttk.Frame(team_frame)
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(
            button_frame, 
            text="Back",
            command=lambda: self.notebook.select(self.config_tab)
        ).pack(side="left")
        
        ttk.Button(
            button_frame, 
            text="Finish",
            command=self.finish_installation
        ).pack(side="right")
    
    def refresh_models_tab(self):
        """Refresh the models tab."""
        # Recreate the models tab
        self.models_tab.destroy()
        self.models_tab = ttk.Frame(self.notebook)
        self.notebook.forget(1)
        self.notebook.insert(1, self.models_tab, text="Models")
        self.setup_models_tab()
        self.notebook.select(self.models_tab)
    
    def browse_install_dir(self):
        """Browse for installation directory."""
        directory = filedialog.askdirectory(initialdir=self.install_dir.get())
        if directory:
            self.install_dir.set(directory)
    
    def download_models(self):
        """Download pretrained models."""
        self.run_with_progress(
            ["python", "setup.py", "--download", "--no-gui"],
            "Downloading models...",
            "Models downloaded successfully!",
            self.refresh_models_tab
        )
    
    def train_models(self):
        """Train models from scratch."""
        result = messagebox.askquestion(
            "Train Models",
            "Training models from scratch can take a long time and requires "
            "significant computational resources.\n\n"
            "Are you sure you want to proceed?",
            icon="warning"
        )
        
        if result == "yes":
            self.run_with_progress(
                ["python", "setup.py", "--train", "--no-gui"],
                "Training models... This may take a while.",
                "Models trained successfully!",
                self.refresh_models_tab
            )
    
    def install_system(self):
        """Install the Baby Monitor System."""
        # Check if models are available first
        _, missing_models = check_models()
        
        if missing_models:
            result = messagebox.askquestion(
                "Missing Models",
                f"The following models are missing: {', '.join(missing_models)}\n\n"
                "Would you like to download them before installation?",
                icon="warning"
            )
            
            if result == "yes":
                self.download_models()
        
        # Build command with options
        cmd = ["python", "setup.py", "--install", "--no-gui", f"--mode={self.install_mode.get()}"]
        
        # Add state detection option
        if not self.state_detection.get():
            cmd.append("--no-state-detection")
        
        # Proceed with installation
        self.run_with_progress(
            cmd,
            "Installing Baby Monitor System...",
            "Installation completed successfully!",
            None
        )
    
    def optimize_system(self):
        """Optimize the system for better performance."""
        if platform.system() == "Linux":
            result = messagebox.askquestion(
                "System Optimization",
                "System optimization requires root privileges and may require a restart.\n\n"
                "Do you want to proceed?",
                icon="warning"
            )
            
            if result == "yes":
                if "raspberry" in platform.release().lower():
                    command = ["sudo", "bash", "tools/system/optimize_raspberry_pi.sh"]
                else:
                    command = ["sudo", "bash", "tools/system/optimize_linux.sh"]
                
                self.run_with_progress(
                    command,
                    "Optimizing system...",
                    "System optimization completed successfully!",
                    None
                )
    
    def apply_config(self):
        """Apply system configuration."""
        try:
            # Update .env file with configuration settings
            self.update_env_file()
            
            # Create desktop shortcut if requested
            if self.create_shortcut.get():
                self.create_desktop_shortcut()
            
            # Configure autostart if requested
            if self.auto_start.get():
                self.configure_autostart()
            
            self.config_status.config(
                text="Configuration applied successfully!",
                foreground="green"
            )
        except Exception as e:
            self.config_status.config(
                text=f"Error applying configuration: {str(e)}",
                foreground="red"
            )
    
    def update_env_file(self):
        """Update or create the .env file with configuration settings."""
        env_path = os.path.join(os.getcwd(), ".env")
        env_data = {}
        
        # Read existing .env file if it exists
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env_data[key.strip()] = value.strip()
        
        # Update with new settings
        env_data["CAMERA_RESOLUTION"] = self.camera_resolution.get()
        env_data["CAMERA_FPS"] = self.camera_fps.get()
        env_data["STATE_DETECTION_ENABLED"] = str(self.state_detection.get()).lower()
        env_data["MODE"] = self.install_mode.get()
        
        # Write updated .env file
        with open(env_path, "w") as f:
            f.write("# Baby Monitor System Environment Configuration\n")
            f.write(f"# Updated by installer v{VERSION} on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in sorted(env_data.items()):
                f.write(f"{key}={value}\n")
    
    def finish_installation(self):
        """Finish the installation process."""
        messagebox.showinfo(
            "Installation Complete",
            "Baby Monitor System has been successfully installed!\n\n"
            "You can now launch the application using:\n"
            "- Windows: run_babymonitor.bat\n"
            "- Linux/macOS: ./run_babymonitor.sh\n\n"
            "Thank you for using our Baby Monitor System!"
        )
        self.root.destroy()
    
    def create_desktop_shortcut(self):
        """Create a desktop shortcut for the application."""
        if platform.system() == "Windows":
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            shortcut_path = os.path.join(desktop_path, "Baby Monitor.lnk")
            
            # Use PowerShell to create shortcut
            powershell_command = f"""
            $WshShell = New-Object -comObject WScript.Shell
            $Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
            $Shortcut.TargetPath = "{os.path.join(os.getcwd(), 'run_babymonitor.bat')}"
            $Shortcut.WorkingDirectory = "{os.getcwd()}"
            $Shortcut.Save()
            """
            
            with open("create_shortcut.ps1", "w") as f:
                f.write(powershell_command)
            
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "create_shortcut.ps1"])
            os.remove("create_shortcut.ps1")
        elif platform.system() == "Linux":
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            shortcut_path = os.path.join(desktop_path, "BabyMonitor.desktop")
            
            with open(shortcut_path, "w") as f:
                f.write(f"""[Desktop Entry]
Type=Application
Name=Baby Monitor
Comment=Baby Monitor System v{VERSION}
Exec={os.path.join(os.getcwd(), 'run_babymonitor.sh')}
Path={os.getcwd()}
Terminal=false
Categories=Utility;
""")
            
            os.chmod(shortcut_path, 0o755)
    
    def configure_autostart(self):
        """Configure the application to start automatically at system startup."""
        if platform.system() == "Windows":
            startup_path = os.path.join(
                os.path.expanduser("~"), 
                "AppData", "Roaming", "Microsoft", "Windows", "Start Menu", "Programs", "Startup"
            )
            shortcut_path = os.path.join(startup_path, "Baby Monitor.lnk")
            
            # Use PowerShell to create shortcut
            powershell_command = f"""
            $WshShell = New-Object -comObject WScript.Shell
            $Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
            $Shortcut.TargetPath = "{os.path.join(os.getcwd(), 'run_babymonitor.bat')}"
            $Shortcut.WorkingDirectory = "{os.getcwd()}"
            $Shortcut.Save()
            """
            
            with open("create_autostart.ps1", "w") as f:
                f.write(powershell_command)
            
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "create_autostart.ps1"])
            os.remove("create_autostart.ps1")
        elif platform.system() == "Linux":
            autostart_path = os.path.join(
                os.path.expanduser("~"), 
                ".config", "autostart"
            )
            os.makedirs(autostart_path, exist_ok=True)
            
            shortcut_path = os.path.join(autostart_path, "BabyMonitor.desktop")
            
            with open(shortcut_path, "w") as f:
                f.write(f"""[Desktop Entry]
Type=Application
Name=Baby Monitor
Comment=Baby Monitor System v{VERSION}
Exec={os.path.join(os.getcwd(), 'run_babymonitor.sh')}
Path={os.getcwd()}
Terminal=false
Categories=Utility;
X-GNOME-Autostart-enabled=true
""")
            
            os.chmod(shortcut_path, 0o755)
    
    def run_with_progress(self, command, start_message, success_message, callback=None):
        """Run a command with a progress dialog."""
        # Create progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Progress")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Set window size and position
        window_width = 400
        window_height = 150
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        progress_window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Add progress message
        message_label = ttk.Label(
            progress_window,
            text=start_message,
            wraplength=380,
            justify="center"
        )
        message_label.pack(pady=10)
        
        # Add progress bar
        progress = ttk.Progressbar(progress_window, mode='indeterminate', length=350)
        progress.pack(pady=10)
        progress.start()
        
        # Add cancel button
        cancel_button = ttk.Button(
            progress_window,
            text="Cancel",
            command=lambda: self.cancel_process(progress_window, process)
        )
        cancel_button.pack(pady=10)
        
        # Run command in a separate thread
        def run_command():
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    message_label.config(text=success_message)
                    progress.stop()
                    cancel_button.config(text="Close")
                    cancel_button.config(command=progress_window.destroy)
                    
                    if callback:
                        self.root.after(1000, callback)
                else:
                    message_label.config(
                        text=f"Error: {stderr}",
                        foreground="red"
                    )
                    progress.stop()
                    cancel_button.config(text="Close")
                    cancel_button.config(command=progress_window.destroy)
            except Exception as e:
                message_label.config(
                    text=f"Error: {str(e)}",
                    foreground="red"
                )
                progress.stop()
                cancel_button.config(text="Close")
                cancel_button.config(command=progress_window.destroy)
        
        # Start the command in a separate thread
        self.root.after(100, run_command)
    
    def cancel_process(self, window, process):
        """Cancel a running process."""
        if process and process.poll() is None:
            process.terminate()
        window.destroy()

def command_line_mode():
    """Run the installer in command line mode."""
    parser = argparse.ArgumentParser(description="Baby Monitor System Installation")
    parser.add_argument("--install", action="store_true", help="Install the system")
    parser.add_argument("--models", action="store_true", help="Check and manage models")
    parser.add_argument("--download", action="store_true", help="Download pretrained models")
    parser.add_argument("--train", action="store_true", help="Train models from scratch")
    parser.add_argument("--mode", choices=["normal", "dev"], default="normal", help="Installation mode")
    parser.add_argument("--no-state-detection", action="store_true", help="Disable person state detection")
    args = parser.parse_args()
    
    print(f"Baby Monitor System - Installation Wizard v{VERSION} (Command Line Mode)")
    print("=" * 60)
    print(f"Version: {VERSION} - {VERSION_NAME}")
    print("=" * 60)
    
    # Check for models
    existing_models, missing_models = check_models()
    
    print("\nModel Status:")
    if existing_models:
        print("Available Models:")
        for model_name, model_size in existing_models:
            print(f"  ✓ {model_name} ({model_size})")
    
    if missing_models:
        print("Missing Models:")
        for model_name in missing_models:
            print(f"  ✗ {model_name}")
    
    # Handle model management
    if args.download:
        print("\nDownloading pretrained models...")
        subprocess.run(["python", "setup.py", "--download", "--no-gui"])
    
    if args.train:
        print("\nTraining models from scratch...")
        subprocess.run(["python", "setup.py", "--train", "--no-gui"])
    
    # Handle installation
    if args.install:
        cmd = ["python", "setup.py", "--install", "--no-gui", f"--mode={args.mode}"]
        if args.no_state_detection:
            cmd.append("--no-state-detection")
        
        print(f"\nInstalling Baby Monitor System in {args.mode} mode...")
        if args.no_state_detection:
            print("Person state detection will be disabled.")
            
        subprocess.run(cmd)
    
    # Check if any action was taken
    if not any([args.install, args.models, args.download, args.train]):
        print("\nNo action specified. Use --help for available options.")
    
    print("\nInstallation wizard completed.")
    
    # Display team information
    print("\nBaby Monitor System Developed By:")
    print("=" * 40)
    for name, role in TEAM_INFO:
        print(f"• {name}: {role}")
    print("=" * 40)

def main():
    """Main entry point."""
    # Check if we should use GUI or command line mode
    if "--no-gui" in sys.argv or not TKINTER_AVAILABLE:
        command_line_mode()
    else:
        # Use GUI mode
        root = tk.Tk()
        app = InstallerApp(root)
        root.mainloop()

if __name__ == "__main__":
    main()
