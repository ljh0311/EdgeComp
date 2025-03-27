#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Baby Monitor System - Unified Installer and Repair Tool
A comprehensive graphical interface for installing, managing, and repairing the Baby Monitor System.
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import platform
import webbrowser
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INSTALL_DIR = SCRIPTS_DIR / "install"
TRAINING_DIR = PROJECT_ROOT / "training"
SRC_DIR = PROJECT_ROOT / "src"

class ScriptRunner:
    """Class to run scripts and capture their output"""
    def __init__(self, output_widget, status_label):
        self.output_widget = output_widget
        self.status_label = status_label
        self.process = None
        self.running = False
    
    def run_script(self, script_path, args=None):
        """Run a script and capture its output"""
        if self.running:
            messagebox.showwarning("Process Running", 
                               "Another process is already running. Please wait for it to complete.")
            return
        
        self.running = True
        self.output_widget.delete(1.0, tk.END)
        self.status_label.config(text=f"Running: {Path(script_path).name}")
        
        # Start the script in a new thread
        threading.Thread(target=self._run_script_thread, args=(script_path, args), daemon=True).start()
    
    def _run_script_thread(self, script_path, args=None):
        """Run a script in a separate thread"""
        script_path = Path(script_path)
        
        try:
            # Determine how to run the script based on its extension
            if script_path.suffix == ".bat":
                self._run_batch_script(script_path, args)
            elif script_path.suffix == ".sh":
                self._run_shell_script(script_path, args)
            elif script_path.suffix == ".py":
                self._run_python_script(script_path, args)
            elif script_path.suffix == ".ipynb":
                self._run_jupyter_notebook(script_path)
            else:
                self._append_output(f"Unsupported script type: {script_path.suffix}")
                self.status_label.config(text="Error: Unsupported script type")
        except Exception as e:
            self._append_output(f"Error running script: {e}")
            self.status_label.config(text="Error running script")
        finally:
            self.running = False
    
    def _run_batch_script(self, script_path, args=None):
        """Run a Windows batch script"""
        if platform.system() != "Windows":
            self._append_output("Error: Cannot run .bat scripts on non-Windows systems.")
            self.status_label.config(text="Error: Cannot run .bat scripts on non-Windows systems.")
            return
        
        self._append_output(f"Running batch script: {script_path.name}\n")
        
        # Build the command
        cmd = [str(script_path)]
        if args:
            cmd.extend(args)
        
        # Run the batch script
        self.process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=str(PROJECT_ROOT)
        )
        
        # Capture and display output
        self._capture_output()
    
    def _run_shell_script(self, script_path, args=None):
        """Run a shell script"""
        if platform.system() == "Windows":
            self._append_output("Error: Cannot run .sh scripts on Windows without WSL or Git Bash.")
            self.status_label.config(text="Error: Cannot run .sh scripts on Windows.")
            return
        
        self._append_output(f"Running shell script: {script_path.name}\n")
        
        # Build the command
        cmd = ["bash", str(script_path)]
        if args:
            cmd.extend(args)
        
        # Run the shell script
        self.process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=str(PROJECT_ROOT)
        )
        
        # Capture and display output
        self._capture_output()
    
    def _run_python_script(self, script_path, args=None):
        """Run a Python script"""
        self._append_output(f"Running Python script: {script_path.name}\n")
        
        # Build the command
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        # Run the Python script
        self.process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=str(PROJECT_ROOT)
        )
        
        # Capture and display output
        self._capture_output()
    
    def _run_jupyter_notebook(self, notebook_path):
        """Run a Jupyter notebook"""
        self._append_output(f"Opening Jupyter notebook: {notebook_path.name}\n")
        
        # First check if jupyter is available
        try:
            subprocess.run([sys.executable, "-m", "pip", "show", "jupyter"], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError:
            self._append_output("Error: Jupyter is not installed. Installing...\n")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "jupyter"], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                self._append_output("Jupyter installed successfully.\n")
            except subprocess.CalledProcessError as e:
                self._append_output(f"Error installing Jupyter: {e}\n")
                self.status_label.config(text="Error: Could not install Jupyter")
                return
        
        # Now open the notebook
        try:
            self._append_output("Starting Jupyter notebook server...\n")
            self._append_output("(This will open in your default browser)\n")
            
            # Run jupyter notebook in a separate process
            notebook_dir = notebook_path.parent
            subprocess.Popen([sys.executable, "-m", "jupyter", "notebook", 
                          str(notebook_path)], cwd=str(notebook_dir))
            
            self._append_output("Jupyter notebook server started.\n")
            self.status_label.config(text="Jupyter notebook server running")
        except Exception as e:
            self._append_output(f"Error starting Jupyter notebook: {e}\n")
            self.status_label.config(text="Error: Could not start Jupyter")
    
    def _capture_output(self):
        """Capture and display output from the process"""
        while self.process.poll() is None:
            line = self.process.stdout.readline()
            if line:
                self._append_output(line)
        
        # Read any remaining output
        remaining_output = self.process.stdout.read()
        if remaining_output:
            self._append_output(remaining_output)
        
        # Update status with exit code
        exit_code = self.process.returncode
        self.status_label.config(text=f"Completed with exit code: {exit_code}")
        self._append_output(f"\nProcess completed with exit code: {exit_code}")
    
    def _append_output(self, text):
        """Append text to the output widget"""
        self.output_widget.insert(tk.END, text)
        self.output_widget.see(tk.END)


class UnifiedManagerGUI(tk.Tk):
    """Main GUI class for the Unified Baby Monitor Manager"""
    def __init__(self):
        super().__init__()
        
        self.title("Baby Monitor System - Unified Installer and Repair Tool")
        self.geometry("900x650")
        self.minsize(800, 600)
        
        # Set application icon if available
        icon_path = Path(PROJECT_ROOT) / "assets" / "icon.ico"
        if icon_path.exists() and platform.system() == "Windows":
            try:
                self.iconbitmap(str(icon_path))
            except:
                pass  # Ignore icon errors
        
        # Detect platform
        self.system = platform.system()
        self.is_raspberry_pi = self._is_raspberry_pi()
        
        # Configure the grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        # Create the main frame
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure the main frame grid
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=0)  # Header
        main_frame.rowconfigure(1, weight=0)  # Welcome section
        main_frame.rowconfigure(2, weight=1)  # Content
        main_frame.rowconfigure(3, weight=0)  # Status bar
        
        # Create the header
        self.create_header(main_frame)
        
        # Create welcome section with buttons for install or repair
        self.create_welcome_section(main_frame)
        
        # Create the content pane (initially hidden)
        self.content_pane = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        self.content_pane.grid(row=2, column=0, sticky="nsew")
        self.content_pane.grid_remove()  # Hide initially
        
        # Create the frames for the content pane
        self.create_content_frames()
        
        # Create the status bar
        self.create_status_bar(main_frame)
        
        # Initialize the script runner
        self.script_runner = ScriptRunner(self.output_console, self.status_label)
        
        # Apply theme
        self.apply_theme()
        
        # Scan for available scripts
        self.scan_available_scripts()
    
    def _is_raspberry_pi(self):
        """Check if the system is a Raspberry Pi."""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                return any("Raspberry Pi" in line for line in f)
        except:
            return False
    
    def apply_theme(self):
        """Apply a custom theme to the application"""
        style = ttk.Style()
        
        # Configure colors
        bg_color = "#f5f5f5"
        accent_color = "#0078d7"
        
        # Configure styles
        style.configure("TFrame", background=bg_color)
        style.configure("Header.TLabel", font=("Arial", 16, "bold"))
        style.configure("Title.TLabel", font=("Arial", 14, "bold"))
        style.configure("Subtitle.TLabel", font=("Arial", 12))
        
        # Configure button styles
        style.configure("Action.TButton", font=("Arial", 12, "bold"))
        style.configure("Normal.TButton", font=("Arial", 10))
        
        # Configure platform tabs
        if self.system == "Windows":
            self.platform_color = "#007bff"  # Windows blue
        elif self.is_raspberry_pi:
            self.platform_color = "#c51a4a"  # Raspberry Pi red
        else:
            self.platform_color = "#ff7700"  # Linux orange
        
        style.configure("Platform.TLabel", foreground=self.platform_color, font=("Arial", 12, "bold"))
    
    def create_header(self, parent):
        """Create the header section"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # Title label
        title_label = ttk.Label(header_frame, text="Baby Monitor System - Unified Manager", 
                               style="Header.TLabel")
        title_label.pack(side=tk.LEFT, pady=5)
        
        # System info
        system_info = f"{self.system}"
        if self.is_raspberry_pi:
            system_info += " (Raspberry Pi)"
        
        system_label = ttk.Label(header_frame, text=f"Detected: {system_info}", style="Platform.TLabel")
        system_label.pack(side=tk.RIGHT, pady=5)
    
    def create_welcome_section(self, parent):
        """Create the welcome section with options to install or repair"""
        welcome_frame = ttk.Frame(parent)
        welcome_frame.grid(row=1, column=0, sticky="ew", pady=10)
        
        # Configure the welcome frame
        welcome_frame.columnconfigure(0, weight=1)
        welcome_frame.rowconfigure(0, weight=0)
        welcome_frame.rowconfigure(1, weight=0)
        welcome_frame.rowconfigure(2, weight=0)
        
        # Welcome message
        welcome_msg = ttk.Label(welcome_frame, 
                            text="Welcome to the Baby Monitor System Manager",
                            style="Title.TLabel")
        welcome_msg.grid(row=0, column=0, pady=5)
        
        # Description
        description = ttk.Label(welcome_frame, 
                             text="Choose an action to perform:",
                             style="Subtitle.TLabel")
        description.grid(row=1, column=0, pady=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(welcome_frame)
        buttons_frame.grid(row=2, column=0, pady=10)
        
        # Install button
        install_button = ttk.Button(buttons_frame, text="Install / Reinstall System", 
                                 command=self.show_install_section,
                                 style="Action.TButton")
        install_button.pack(side=tk.LEFT, padx=10)
        
        # Repair button
        repair_button = ttk.Button(buttons_frame, text="Repair / Fix System",
                               command=self.show_repair_section,
                               style="Action.TButton")
        repair_button.pack(side=tk.LEFT, padx=10)
        
        # Training button
        training_button = ttk.Button(buttons_frame, text="Training & Models",
                                 command=self.show_training_section,
                                 style="Action.TButton")
        training_button.pack(side=tk.LEFT, padx=10)
        
        # Configuration button
        config_button = ttk.Button(buttons_frame, text="Configuration",
                               command=self.show_config_section,
                               style="Action.TButton")
        config_button.pack(side=tk.LEFT, padx=10)
        
        # Open web button (only if service is likely running)
        web_repair_button = ttk.Button(buttons_frame, text="Web Repair Tools",
                                    command=self.open_web_repair,
                                    style="Action.TButton")
        web_repair_button.pack(side=tk.LEFT, padx=10)
        
        # Save reference
        self.welcome_frame = welcome_frame
    
    def create_content_frames(self):
        """Create frames that will be shown in the content pane"""
        # Create the various section frames
        self.install_frame = ttk.LabelFrame(self.content_pane, text="Installation")
        self.repair_frame = ttk.LabelFrame(self.content_pane, text="Repair Tools")
        self.training_frame = ttk.LabelFrame(self.content_pane, text="Training & Models")
        self.config_frame = ttk.LabelFrame(self.content_pane, text="Configuration")
        self.output_frame = ttk.LabelFrame(self.content_pane, text="Output")
        
        # Configure frames with grids
        for frame in [self.install_frame, self.repair_frame, self.training_frame, self.config_frame]:
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
        
        # Install section notebooks
        self.install_notebook = ttk.Notebook(self.install_frame)
        self.install_notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create tabs for different platform installs
        self.windows_install_tab = ttk.Frame(self.install_notebook)
        self.linux_install_tab = ttk.Frame(self.install_notebook)
        self.pi_install_tab = ttk.Frame(self.install_notebook)
        
        # Add relevant tabs based on platform
        if self.system == "Windows":
            self.install_notebook.add(self.windows_install_tab, text="Windows Install")
        elif self.is_raspberry_pi:
            self.install_notebook.add(self.pi_install_tab, text="Raspberry Pi Install")
        else:
            self.install_notebook.add(self.linux_install_tab, text="Linux Install")
        
        # Configure the install tabs
        for tab in [self.windows_install_tab, self.linux_install_tab, self.pi_install_tab]:
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
        
        # Create frames for install tabs
        self.windows_install_frame = ttk.Frame(self.windows_install_tab)
        self.windows_install_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.linux_install_frame = ttk.Frame(self.linux_install_tab)
        self.linux_install_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.pi_install_frame = ttk.Frame(self.pi_install_tab)
        self.pi_install_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Repair section notebooks
        self.repair_notebook = ttk.Notebook(self.repair_frame)
        self.repair_notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create tabs for repair options
        self.camera_repair_tab = ttk.Frame(self.repair_notebook)
        self.audio_repair_tab = ttk.Frame(self.repair_notebook)
        self.system_repair_tab = ttk.Frame(self.repair_notebook)
        
        # Add repair tabs
        self.repair_notebook.add(self.camera_repair_tab, text="Camera")
        self.repair_notebook.add(self.audio_repair_tab, text="Audio")
        self.repair_notebook.add(self.system_repair_tab, text="System")
        
        # Configure repair tabs
        for tab in [self.camera_repair_tab, self.audio_repair_tab, self.system_repair_tab]:
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
        
        # Create frames for repair tabs
        self.camera_repair_frame = ttk.Frame(self.camera_repair_tab)
        self.camera_repair_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.audio_repair_frame = ttk.Frame(self.audio_repair_tab)
        self.audio_repair_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.system_repair_frame = ttk.Frame(self.system_repair_tab)
        self.system_repair_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Training section notebooks
        self.training_notebook = ttk.Notebook(self.training_frame)
        self.training_notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create tabs for training options
        self.model_download_tab = ttk.Frame(self.training_notebook)
        self.emotion_train_tab = ttk.Frame(self.training_notebook)
        self.benchmark_tab = ttk.Frame(self.training_notebook)
        
        # Add training tabs
        self.training_notebook.add(self.model_download_tab, text="Download Models")
        self.training_notebook.add(self.emotion_train_tab, text="Train Emotion Models")
        self.training_notebook.add(self.benchmark_tab, text="Benchmark Models")
        
        # Configure training tabs
        for tab in [self.model_download_tab, self.emotion_train_tab, self.benchmark_tab]:
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
        
        # Create frames for training tabs
        self.model_download_frame = ttk.Frame(self.model_download_tab)
        self.model_download_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.emotion_train_frame = ttk.Frame(self.emotion_train_tab)
        self.emotion_train_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.benchmark_frame = ttk.Frame(self.benchmark_tab)
        self.benchmark_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configuration section notebooks
        self.config_notebook = ttk.Notebook(self.config_frame)
        self.config_notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create tabs for configuration options
        self.system_config_tab = ttk.Frame(self.config_notebook)
        self.camera_config_tab = ttk.Frame(self.config_notebook)
        self.audio_config_tab = ttk.Frame(self.config_notebook)
        self.network_config_tab = ttk.Frame(self.config_notebook)
        
        # Add configuration tabs
        self.config_notebook.add(self.system_config_tab, text="System")
        self.config_notebook.add(self.camera_config_tab, text="Camera")
        self.config_notebook.add(self.audio_config_tab, text="Audio")
        self.config_notebook.add(self.network_config_tab, text="Network")
        
        # Configure configuration tabs
        for tab in [self.system_config_tab, self.camera_config_tab, self.audio_config_tab, self.network_config_tab]:
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
        
        # Create frames for configuration tabs
        self.system_config_frame = ttk.Frame(self.system_config_tab)
        self.system_config_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.camera_config_frame = ttk.Frame(self.camera_config_tab)
        self.camera_config_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.audio_config_frame = ttk.Frame(self.audio_config_tab)
        self.audio_config_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.network_config_frame = ttk.Frame(self.network_config_tab)
        self.network_config_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure output frame
        self.output_frame.columnconfigure(0, weight=1)
        self.output_frame.rowconfigure(0, weight=1)
        self.output_frame.rowconfigure(1, weight=0)
        
        # Output console
        self.output_console = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD,
                                                    font=("Consolas", 9))
        self.output_console.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Output buttons frame
        buttons_frame = ttk.Frame(self.output_frame)
        buttons_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Clear button
        clear_button = ttk.Button(buttons_frame, text="Clear Output", 
                                command=lambda: self.output_console.delete(1.0, tk.END))
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Copy button
        copy_button = ttk.Button(buttons_frame, text="Copy to Clipboard", 
                               command=self.copy_to_clipboard)
        copy_button.pack(side=tk.LEFT, padx=5)
        
        # Back button
        back_button = ttk.Button(buttons_frame, text="Back to Main Menu", 
                              command=self.show_welcome_screen)
        back_button.pack(side=tk.RIGHT, padx=5)
    
    def create_status_bar(self, parent):
        """Create the status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        
        # Status label
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Version label
        version_label = ttk.Label(status_frame, text="v1.1.0")
        version_label.pack(side=tk.RIGHT, padx=5)
    
    def show_welcome_screen(self):
        """Show the welcome screen"""
        # Remove content pane
        self.content_pane.grid_remove()
        
        # Show welcome frame
        self.welcome_frame.grid()
        
        # Update status
        self.status_label.config(text="Ready")
    
    def show_install_section(self):
        """Show the installation section"""
        # Hide welcome frame
        self.welcome_frame.grid_remove()
        
        # Remove any existing panes
        for pane in self.content_pane.panes():
            self.content_pane.forget(pane)
        
        # Add install and output frames
        self.content_pane.add(self.install_frame, weight=1)
        self.content_pane.add(self.output_frame, weight=1)
        
        # Show content pane
        self.content_pane.grid()
        
        # Update status
        self.status_label.config(text="Installation Section")
    
    def show_repair_section(self):
        """Show the repair section"""
        # Hide welcome frame
        self.welcome_frame.grid_remove()
        
        # Remove any existing panes
        for pane in self.content_pane.panes():
            self.content_pane.forget(pane)
        
        # Add repair and output frames
        self.content_pane.add(self.repair_frame, weight=1)
        self.content_pane.add(self.output_frame, weight=1)
        
        # Show content pane
        self.content_pane.grid()
        
        # Update status
        self.status_label.config(text="Repair Section")
    
    def show_training_section(self):
        """Show the training section"""
        # Hide welcome frame
        self.welcome_frame.grid_remove()
        
        # Remove any existing panes
        for pane in self.content_pane.panes():
            self.content_pane.forget(pane)
        
        # Add training and output frames
        self.content_pane.add(self.training_frame, weight=1)
        self.content_pane.add(self.output_frame, weight=1)
        
        # Show content pane
        self.content_pane.grid()
        
        # Update status
        self.status_label.config(text="Training & Models Section")
    
    def show_config_section(self):
        """Show the configuration section"""
        # Hide welcome frame
        self.welcome_frame.grid_remove()
        
        # Remove any existing panes
        for pane in self.content_pane.panes():
            self.content_pane.forget(pane)
        
        # Add config and output frames
        self.content_pane.add(self.config_frame, weight=1)
        self.content_pane.add(self.output_frame, weight=1)
        
        # Show content pane
        self.content_pane.grid()
        
        # Update status
        self.status_label.config(text="Configuration Section")
    
    def open_web_repair(self):
        """Open the web repair tools page"""
        # Construct the URL (either localhost if running, or from file)
        web_url = "http://localhost:5000/repair"
        
        # Try to open in browser
        try:
            webbrowser.open(web_url)
            self.status_label.config(text="Opening web repair tools...")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open web repair tools: {e}")
            self.status_label.config(text="Failed to open web repair tools")
    
    def scan_available_scripts(self):
        """Scan for available scripts and create buttons for them"""
        # Clear existing frames
        for frame_list in [
            [self.windows_install_frame, self.linux_install_frame, self.pi_install_frame],
            [self.camera_repair_frame, self.audio_repair_frame, self.system_repair_frame],
            [self.model_download_frame, self.emotion_train_frame, self.benchmark_frame],
            [self.system_config_frame, self.camera_config_frame, self.audio_config_frame, self.network_config_frame]
        ]:
            for frame in frame_list:
                for widget in frame.winfo_children():
                    widget.destroy()
        
        # Initialize script lists
        windows_install_scripts = []
        linux_install_scripts = []
        pi_install_scripts = []
        
        camera_repair_scripts = []
        audio_repair_scripts = []
        system_repair_scripts = []
        
        model_download_scripts = []
        emotion_train_scripts = []
        benchmark_scripts = []
        
        system_config_scripts = []
        camera_config_scripts = []
        audio_config_scripts = []
        network_config_scripts = []
        
        # Scan install directory for platform-specific scripts
        if INSTALL_DIR.exists():
            for script in INSTALL_DIR.glob("*"):
                if script.is_file() and script.name != "README.md":
                    script_name = script.name.lower()
                    
                    # Filter and categorize by platform
                    if script.suffix == ".bat" or "windows" in script_name:
                        windows_install_scripts.append(script)
                    elif "pi" in script_name or "raspberry" in script_name:
                        pi_install_scripts.append(script)
                    elif script.suffix == ".sh" or "linux" in script_name or "install.py" in script_name:
                        # Python scripts are generally cross-platform
                        if script.suffix == ".py":
                            windows_install_scripts.append(script)
                            linux_install_scripts.append(script)
                            pi_install_scripts.append(script)
                        else:
                            linux_install_scripts.append(script)
        
        # Scan main scripts directory for fix scripts
        for script in SCRIPTS_DIR.glob("*"):
            if script.is_file() and (script.name.startswith("fix_") or script.name.startswith("restart_")):
                script_name = script.name.lower()
                script_compatible = True
                
                # Check platform compatibility
                if script.suffix == ".bat" and self.system != "Windows":
                    script_compatible = False
                elif script.suffix == ".sh" and self.system == "Windows":
                    script_compatible = False
                
                if script_compatible:
                    if "camera" in script_name or "video" in script_name:
                        camera_repair_scripts.append(script)
                    elif "audio" in script_name or "sound" in script_name:
                        audio_repair_scripts.append(script)
                    else:
                        system_repair_scripts.append(script)
        
        # Scan training directory for training scripts
        if TRAINING_DIR.exists():
            for script in TRAINING_DIR.glob("*"):
                if script.is_file():
                    script_name = script.name.lower()
                    if script.suffix in [".py", ".ipynb"]:  # Python scripts and notebooks
                        if "download" in script_name or "setup" in script_name:
                            model_download_scripts.append(script)
                        elif "train" in script_name or "finetune" in script_name or "emotion" in script_name:
                            emotion_train_scripts.append(script)
                        elif "benchmark" in script_name or "test" in script_name:
                            benchmark_scripts.append(script)
        
        # Scan SRC directory for utility and configuration scripts
        if SRC_DIR.exists():
            # Find configuration and utility scripts in babymonitor/utils
            utils_dir = SRC_DIR / "babymonitor" / "utils"
            if utils_dir.exists():
                for script in utils_dir.glob("*.py"):
                    script_name = script.name.lower()
                    if "config" in script_name:
                        system_config_scripts.append(script)
                    elif "camera" in script_name:
                        camera_config_scripts.append(script)
                    elif "audio" in script_name:
                        audio_config_scripts.append(script)
                    elif "network" in script_name or "web" in script_name:
                        network_config_scripts.append(script)
                    elif "setup" in script_name or "model" in script_name:
                        model_download_scripts.append(script)
            
            # Check for camera-specific utility scripts
            camera_dir = SRC_DIR / "babymonitor" / "camera"
            if camera_dir.exists():
                for script in camera_dir.glob("*.py"):
                    if script.name.lower() != "__init__.py":
                        camera_config_scripts.append(script)
            
            # Check for audio-specific utility scripts
            audio_dir = SRC_DIR / "babymonitor" / "audio"
            if audio_dir.exists():
                for script in audio_dir.glob("*.py"):
                    if script.name.lower() != "__init__.py":
                        audio_config_scripts.append(script)
        
        # Add platform-specific utility scripts
        setup_scripts_dir = SRC_DIR / "babymonitor" / "scripts"
        if setup_scripts_dir.exists():
            for script in setup_scripts_dir.glob("*"):
                if script.is_file():
                    script_name = script.name.lower()
                    
                    # Check platform compatibility
                    script_compatible = True
                    if script.suffix == ".bat" and self.system != "Windows":
                        script_compatible = False
                    elif script.suffix == ".sh" and self.system == "Windows":
                        script_compatible = False
                    
                    if script_compatible:
                        if "setup" in script_name or "install" in script_name:
                            if "windows" in script_name:
                                windows_install_scripts.append(script)
                            elif "linux" in script_name:
                                linux_install_scripts.append(script)
                            elif "pi" in script_name or "raspberry" in script_name:
                                pi_install_scripts.append(script)
                            else:
                                # Generic setup scripts work on all platforms
                                windows_install_scripts.append(script)
                                linux_install_scripts.append(script)
                                pi_install_scripts.append(script)
        
        # Add the installer.py to all install sections
        installer_path = INSTALL_DIR / "install.py"
        if installer_path.exists():
            windows_install_scripts.append(installer_path)
            linux_install_scripts.append(installer_path)
            pi_install_scripts.append(installer_path)
        
        # Add platform-specific GUI elements
        if self.system == "Windows":
            self._add_section_heading(self.windows_install_frame, "Windows Installation Scripts")
            self._add_script_buttons(self.windows_install_frame, windows_install_scripts)
        else:
            if self.is_raspberry_pi:
                self._add_section_heading(self.pi_install_frame, "Raspberry Pi Installation Scripts")
                self._add_script_buttons(self.pi_install_frame, pi_install_scripts)
            else:
                self._add_section_heading(self.linux_install_frame, "Linux Installation Scripts")
                self._add_script_buttons(self.linux_install_frame, linux_install_scripts)
        
        # Add repair script buttons if compatible
        self._add_section_heading(self.camera_repair_frame, "Camera Repair Tools")
        self._add_script_buttons(self.camera_repair_frame, camera_repair_scripts)
        
        self._add_section_heading(self.audio_repair_frame, "Audio Repair Tools")
        self._add_script_buttons(self.audio_repair_frame, audio_repair_scripts)
        
        self._add_section_heading(self.system_repair_frame, "System Repair Tools")
        self._add_script_buttons(self.system_repair_frame, system_repair_scripts)
        
        # Add training script buttons
        self._add_section_heading(self.model_download_frame, "Model Download Tools")
        self._add_script_buttons(self.model_download_frame, model_download_scripts)
        
        self._add_section_heading(self.emotion_train_frame, "Emotion Model Training")
        self._add_script_buttons(self.emotion_train_frame, emotion_train_scripts)
        
        self._add_section_heading(self.benchmark_frame, "Model Benchmarking")
        self._add_script_buttons(self.benchmark_frame, benchmark_scripts)
        
        # Add configuration script buttons
        self._add_section_heading(self.system_config_frame, "System Configuration")
        self._add_script_buttons(self.system_config_frame, system_config_scripts)
        
        self._add_section_heading(self.camera_config_frame, "Camera Configuration")
        self._add_script_buttons(self.camera_config_frame, camera_config_scripts)
        
        self._add_section_heading(self.audio_config_frame, "Audio Configuration")
        self._add_script_buttons(self.audio_config_frame, audio_config_scripts)
        
        self._add_section_heading(self.network_config_frame, "Network Configuration")
        self._add_script_buttons(self.network_config_frame, network_config_scripts)
        
        # Add standalone direct buttons for training models
        self._add_training_buttons()
        
        # Add configuration tools
        self._add_config_tools()
        
        # Add unified installer to the relevant install section
        if self.system == "Windows":
            self._add_unified_installer_button(self.windows_install_frame)
        elif self.is_raspberry_pi:
            self._add_unified_installer_button(self.pi_install_frame)
        else:
            self._add_unified_installer_button(self.linux_install_frame)
        
        # Add direct repair buttons
        self._add_direct_repair_buttons()
    
    def _add_section_heading(self, parent, heading_text):
        """Add a section heading to a frame"""
        heading = ttk.Label(parent, text=heading_text, style="Title.TLabel")
        heading.pack(fill=tk.X, padx=5, pady=5)
    
    def _add_script_buttons(self, parent, scripts):
        """Add script buttons to a frame"""
        if not scripts:
            no_scripts_label = ttk.Label(parent, text="No scripts available")
            no_scripts_label.pack(fill=tk.X, padx=5, pady=5)
            return
        
        for script in sorted(scripts, key=lambda x: x.name):
            # Create frame for the script
            script_frame = ttk.Frame(parent)
            script_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Get script text and tooltip
            button_text = script.name
            tooltip_text = f"Run {script.name}"
            
            if "install" in str(script.parent):
                tooltip_text = f"Run installation script: {script.name}"
            
            # Create button
            script_button = ttk.Button(script_frame, text=button_text,
                                    command=lambda s=script: self.script_runner.run_script(s))
            script_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Add tooltip
            self.create_tooltip(script_button, tooltip_text)
    
    def _add_unified_installer_button(self, parent):
        """Add a unified installer button to a frame"""
        # Create a separator
        separator = ttk.Separator(parent, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, padx=10, pady=10)
        
        # Create a label
        label = ttk.Label(parent, text="Recommended Installation Method:")
        label.pack(fill=tk.X, padx=5, pady=5)
        
        # Create frame for the button
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create button
        installer_button = ttk.Button(button_frame, text="Start Unified GUI Installer",
                                   command=self.run_unified_installer,
                                   style="Action.TButton")
        installer_button.pack(fill=tk.X, padx=20, pady=5)
    
    def _add_direct_repair_buttons(self):
        """Add direct repair buttons to the repair tabs"""
        # Camera repair
        camera_repair_frame = ttk.Frame(self.camera_repair_frame)
        camera_repair_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Create camera repair buttons
        restart_camera_btn = ttk.Button(camera_repair_frame, text="Restart Camera",
                                     command=lambda: self.run_repair_command("restart_camera"))
        restart_camera_btn.pack(side=tk.LEFT, padx=5)
        
        test_camera_btn = ttk.Button(camera_repair_frame, text="Test Camera",
                                   command=lambda: self.run_repair_command("test_camera"))
        test_camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Audio repair
        audio_repair_frame = ttk.Frame(self.audio_repair_frame)
        audio_repair_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Create audio repair buttons
        restart_audio_btn = ttk.Button(audio_repair_frame, text="Restart Audio",
                                    command=lambda: self.run_repair_command("restart_audio"))
        restart_audio_btn.pack(side=tk.LEFT, padx=5)
        
        test_audio_btn = ttk.Button(audio_repair_frame, text="Test Audio",
                                  command=lambda: self.run_repair_command("test_audio"))
        test_audio_btn.pack(side=tk.LEFT, padx=5)
        
        # System repair
        system_repair_frame = ttk.Frame(self.system_repair_frame)
        system_repair_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Create system repair buttons
        restart_system_btn = ttk.Button(system_repair_frame, text="Restart System",
                                     command=lambda: self.run_repair_command("restart_system"))
        restart_system_btn.pack(side=tk.LEFT, padx=5)
        
        check_system_btn = ttk.Button(system_repair_frame, text="Check System",
                                    command=lambda: self.run_repair_command("check_system"))
        check_system_btn.pack(side=tk.LEFT, padx=5)
    
    def run_unified_installer(self):
        """Run the unified installer GUI"""
        installer_path = INSTALL_DIR / "install.py"
        
        if installer_path.exists():
            self.script_runner.run_script(installer_path)
        else:
            messagebox.showerror("Error", "Unified installer not found in the install directory.")
    
    def run_repair_command(self, command):
        """Run a repair command"""
        # Here we'd typically launch a script or command based on the repair action
        # For this implementation, we'll just show what would happen
        self.output_console.delete(1.0, tk.END)
        self.output_console.insert(tk.END, f"Running repair command: {command}\n")
        self.output_console.insert(tk.END, "This would execute the appropriate repair action.\n")
        
        # Look for matching fix script
        fix_scripts = list(SCRIPTS_DIR.glob(f"fix_*.bat")) + list(SCRIPTS_DIR.glob(f"fix_*.sh"))
        matching_scripts = [s for s in fix_scripts if command.split('_')[1] in s.name.lower()]
        
        # If found, run it
        if matching_scripts:
            self.script_runner.run_script(matching_scripts[0])
        else:
            self.output_console.insert(tk.END, f"No matching fix script found for command: {command}\n")
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # Create a toplevel window
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(self.tooltip, text=text, background="#FFFFDD",
                            relief=tk.SOLID, borderwidth=1)
            label.pack(ipadx=5, ipady=2)
        
        def leave(event):
            if hasattr(self, "tooltip"):
                self.tooltip.destroy()
        
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
    
    def copy_to_clipboard(self):
        """Copy the output console content to clipboard"""
        self.clipboard_clear()
        self.clipboard_append(self.output_console.get(1.0, tk.END))
        messagebox.showinfo("Clipboard", "Output copied to clipboard!")
    
    def _add_training_buttons(self):
        """Add standalone buttons for training-related functions"""
        # Model download frame
        download_btn_frame = ttk.Frame(self.model_download_frame)
        download_btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Create download model buttons
        download_all_btn = ttk.Button(download_btn_frame, text="Download All Models",
                                   command=self.download_all_models)
        download_all_btn.pack(side=tk.LEFT, padx=5)
        
        download_emotion_btn = ttk.Button(download_btn_frame, text="Download Emotion Models",
                                      command=self.download_emotion_models)
        download_emotion_btn.pack(side=tk.LEFT, padx=5)
        
        verify_models_btn = ttk.Button(download_btn_frame, text="Verify Model Files",
                                    command=self.verify_models)
        verify_models_btn.pack(side=tk.LEFT, padx=5)
        
        # Emotion training frame
        train_btn_frame = ttk.Frame(self.emotion_train_frame)
        train_btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Create training buttons
        speechbrain_btn = ttk.Button(train_btn_frame, text="Open SpeechBrain Training",
                                  command=self.open_speechbrain_training)
        speechbrain_btn.pack(side=tk.LEFT, padx=5)
        
        custom_model_btn = ttk.Button(train_btn_frame, text="Create Custom Model",
                                    command=self.run_create_emotion_model)
        custom_model_btn.pack(side=tk.LEFT, padx=5)
        
        # Benchmark frame
        benchmark_btn_frame = ttk.Frame(self.benchmark_frame)
        benchmark_btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Create benchmark buttons
        benchmark_btn = ttk.Button(benchmark_btn_frame, text="Run Model Benchmark",
                                command=self.run_model_benchmark)
        benchmark_btn.pack(side=tk.LEFT, padx=5)
        
        test_models_btn = ttk.Button(benchmark_btn_frame, text="Test Emotion Models",
                                  command=self.test_emotion_models)
        test_models_btn.pack(side=tk.LEFT, padx=5)
    
    def _add_config_tools(self):
        """Add configuration tools to the config section"""
        # System config frame
        system_config_btn_frame = ttk.Frame(self.system_config_frame)
        system_config_btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Create system config buttons
        edit_config_btn = ttk.Button(system_config_btn_frame, text="Edit .env File",
                                  command=self.edit_env_file)
        edit_config_btn.pack(side=tk.LEFT, padx=5)
        
        system_mode_btn = ttk.Button(system_config_btn_frame, text="Set System Mode",
                                  command=self.set_system_mode)
        system_mode_btn.pack(side=tk.LEFT, padx=5)
        
        # Camera config frame
        camera_config_btn_frame = ttk.Frame(self.camera_config_frame)
        camera_config_btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Create camera config buttons
        detect_cameras_btn = ttk.Button(camera_config_btn_frame, text="Detect Cameras",
                                     command=self.detect_cameras)
        detect_cameras_btn.pack(side=tk.LEFT, padx=5)
        
        test_camera_btn = ttk.Button(camera_config_btn_frame, text="Test Camera",
                                  command=self.test_camera)
        test_camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Audio config frame
        audio_config_btn_frame = ttk.Frame(self.audio_config_frame)
        audio_config_btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Create audio config buttons
        detect_audio_btn = ttk.Button(audio_config_btn_frame, text="Detect Audio Devices",
                                   command=self.detect_audio_devices)
        detect_audio_btn.pack(side=tk.LEFT, padx=5)
        
        test_audio_btn = ttk.Button(audio_config_btn_frame, text="Test Audio",
                                 command=self.test_audio)
        test_audio_btn.pack(side=tk.LEFT, padx=5)
        
        # Network config frame
        network_config_btn_frame = ttk.Frame(self.network_config_frame)
        network_config_btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Create network config buttons
        set_port_btn = ttk.Button(network_config_btn_frame, text="Configure Web Port",
                               command=self.configure_web_port)
        set_port_btn.pack(side=tk.LEFT, padx=5)
        
        test_server_btn = ttk.Button(network_config_btn_frame, text="Test Web Server",
                                  command=self.test_web_server)
        test_server_btn.pack(side=tk.LEFT, padx=5)
    
    # Training utility methods
    def download_all_models(self):
        """Download all required models"""
        # First look for dedicated download script
        download_script = TRAINING_DIR / "download_models.py"
        if download_script.exists():
            self.script_runner.run_script(download_script)
        else:
            # Try the setup_models.py utility
            setup_models_script = SRC_DIR / "babymonitor" / "utils" / "setup_models.py"
            if setup_models_script.exists():
                self.script_runner.run_script(setup_models_script)
            else:
                # Fallback to running the setup.py with --download-models flag
                setup_script = INSTALL_DIR / "setup.py"
                if setup_script.exists():
                    self.script_runner.run_script(setup_script, ["--download-models"])
                else:
                    messagebox.showerror("Error", "Could not find any model download scripts")
    
    def download_emotion_models(self):
        """Download emotion detection models"""
        # Run setup.py with emotion-only flag if available
        setup_script = INSTALL_DIR / "setup.py"
        if setup_script.exists():
            self.script_runner.run_script(setup_script, ["--download-models", "--emotion-only"])
        else:
            messagebox.showerror("Error", "Could not find setup.py script")
    
    def verify_models(self):
        """Verify that model files are correctly installed"""
        # Run setup.py with fix-models flag if available
        setup_script = INSTALL_DIR / "setup.py"
        if setup_script.exists():
            self.script_runner.run_script(setup_script, ["--fix-models"])
        else:
            messagebox.showerror("Error", "Could not find setup.py script")
    
    def open_speechbrain_training(self):
        """Open SpeechBrain training notebook"""
        # Look for the speechbrain notebook
        notebook_path = TRAINING_DIR / "speechbrain-finetune.ipynb"
        if notebook_path.exists():
            self.script_runner.run_script(notebook_path)
        else:
            messagebox.showerror("Error", "Could not find SpeechBrain training notebook")
    
    def run_create_emotion_model(self):
        """Run script to create a custom emotion model"""
        # Look for the create_emotion_model script
        script_path = SRC_DIR / "babymonitor" / "utils" / "create_emotion_model.py"
        if script_path.exists():
            self.script_runner.run_script(script_path)
        else:
            messagebox.showerror("Error", "Could not find custom emotion model creation script")
    
    def run_model_benchmark(self):
        """Run the model benchmark script"""
        # Look for the benchmark script
        benchmark_script = TRAINING_DIR / "model_benchmark.py"
        if benchmark_script.exists():
            self.script_runner.run_script(benchmark_script)
        else:
            messagebox.showerror("Error", "Could not find model benchmark script")
    
    def test_emotion_models(self):
        """Test emotion detection models"""
        # Look for test scripts
        test_script = SRC_DIR / "babymonitor" / "utils" / "test_model_loader.py"
        if test_script.exists():
            self.script_runner.run_script(test_script)
        else:
            messagebox.showerror("Error", "Could not find emotion model test script")
    
    # Configuration utility methods
    def edit_env_file(self):
        """Open .env file for editing"""
        env_file = PROJECT_ROOT / ".env"
        
        if not env_file.exists():
            # Create a default .env file if it doesn't exist
            result = messagebox.askyesno("File Not Found", 
                                     ".env file not found. Create a default one?")
            if result:
                # Try to run setup.py to create default .env
                setup_script = INSTALL_DIR / "setup.py"
                if setup_script.exists():
                    self.script_runner.run_script(setup_script, ["--configure-only"])
                    return
                else:
                    # Create basic .env manually
                    with open(env_file, "w") as f:
                        f.write("# Baby Monitor System Environment Configuration\n")
                        f.write("MODE=normal\n")
                        f.write("CAMERA_INDEX=0\n")
                        f.write("WEB_PORT=5000\n")
            else:
                return
        
        # Open the file in a basic text editor
        if platform.system() == "Windows":
            os.startfile(env_file)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(env_file)])
        else:  # Linux
            subprocess.run(["xdg-open", str(env_file)])
        
        messagebox.showinfo("Edit Configuration", 
                         "The .env file has been opened for editing.\n\n" +
                         "After saving your changes, restart the application for them to take effect.")
    
    def set_system_mode(self):
        """Set the system mode (normal or dev)"""
        # Create a simple dialog for choosing mode
        dialog = tk.Toplevel(self)
        dialog.title("Set System Mode")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Mode selection
        ttk.Label(frame, text="Select System Mode:", style="Title.TLabel").pack(pady=(0, 10))
        
        mode_var = tk.StringVar(value="normal")
        
        ttk.Radiobutton(frame, text="Normal Mode - For regular users",
                      variable=mode_var, value="normal").pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(frame, text="Developer Mode - For advanced features and debugging",
                      variable=mode_var, value="dev").pack(anchor=tk.W, pady=2)
        
        # Button frame
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        def set_mode():
            mode = mode_var.get()
            # Look for .env file
            env_file = PROJECT_ROOT / ".env"
            
            if env_file.exists():
                # Read existing content
                with open(env_file, "r") as f:
                    lines = f.readlines()
                
                # Update the MODE line
                mode_updated = False
                for i, line in enumerate(lines):
                    if line.startswith("MODE="):
                        lines[i] = f"MODE={mode}\n"
                        mode_updated = True
                        break
                
                if not mode_updated:
                    lines.append(f"MODE={mode}\n")
                
                # Write back
                with open(env_file, "w") as f:
                    f.writelines(lines)
                
                messagebox.showinfo("Success", f"System mode set to: {mode}")
            else:
                # Create new .env with mode
                with open(env_file, "w") as f:
                    f.write("# Baby Monitor System Environment Configuration\n")
                    f.write(f"MODE={mode}\n")
                    f.write("CAMERA_INDEX=0\n")
                    f.write("WEB_PORT=5000\n")
                
                messagebox.showinfo("Success", 
                                 f"Created new .env file with mode: {mode}")
            
            dialog.destroy()
        
        ttk.Button(btn_frame, text="Apply", command=set_mode).pack(side=tk.RIGHT, padx=5)
    
    def detect_cameras(self):
        """Detect available cameras and update configuration"""
        # Look for the camera utility script
        camera_script = SRC_DIR / "babymonitor" / "utils" / "camera.py"
        if camera_script.exists():
            self.script_runner.run_script(camera_script)
        else:
            # Run setup.py with camera repair flag
            setup_script = INSTALL_DIR / "setup.py"
            if setup_script.exists():
                self.script_runner.run_script(setup_script, ["--repair-camera"])
            else:
                messagebox.showerror("Error", "Could not find camera utility script")
    
    def test_camera(self):
        """Test camera functionality"""
        # Look for camera test script
        camera_test = SRC_DIR / "babymonitor" / "camera" / "test_camera.py"
        if camera_test.exists():
            self.script_runner.run_script(camera_test)
        else:
            # Try main camera.py
            camera_script = SRC_DIR / "babymonitor" / "camera.py"
            if camera_script.exists():
                self.script_runner.run_script(camera_script)
            else:
                messagebox.showerror("Error", "Could not find camera test script")
    
    def detect_audio_devices(self):
        """Detect available audio devices and update configuration"""
        # Look for audio utility
        audio_script = SRC_DIR / "babymonitor" / "audio.py"
        if audio_script.exists():
            self.script_runner.run_script(audio_script)
        else:
            messagebox.showerror("Error", "Could not find audio utility script")
    
    def test_audio(self):
        """Test audio functionality"""
        # Look for audio test script
        audio_test = SRC_DIR / "babymonitor" / "audio" / "test_audio.py"
        if audio_test.exists():
            self.script_runner.run_script(audio_test)
        else:
            # Try main audio.py
            audio_script = SRC_DIR / "babymonitor" / "audio.py"
            if audio_script.exists():
                self.script_runner.run_script(audio_script)
            else:
                messagebox.showerror("Error", "Could not find audio test script")
    
    def configure_web_port(self):
        """Configure web server port"""
        # Create a simple dialog for setting web port
        dialog = tk.Toplevel(self)
        dialog.title("Configure Web Port")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Port input
        ttk.Label(frame, text="Set Web Server Port:", style="Title.TLabel").pack(pady=(0, 10))
        
        port_var = tk.StringVar(value="5000")
        port_entry = ttk.Entry(frame, textvariable=port_var, width=10)
        port_entry.pack(pady=5)
        
        # Default ports
        ttk.Label(frame, text="Common ports:").pack(anchor=tk.W, pady=(10, 0))
        
        def set_default_port(port):
            port_var.set(str(port))
        
        ports_frame = ttk.Frame(frame)
        ports_frame.pack(fill=tk.X, pady=5)
        
        for port in [5000, 8000, 8080, 3000]:
            ttk.Button(ports_frame, text=str(port), 
                    command=lambda p=port: set_default_port(p)).pack(side=tk.LEFT, padx=5)
        
        # Button frame
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        def set_port():
            try:
                port = int(port_var.get())
                if port < 1 or port > 65535:
                    raise ValueError("Port must be between 1 and 65535")
                
                # Look for .env file
                env_file = PROJECT_ROOT / ".env"
                
                if env_file.exists():
                    # Read existing content
                    with open(env_file, "r") as f:
                        lines = f.readlines()
                    
                    # Update the WEB_PORT line
                    port_updated = False
                    for i, line in enumerate(lines):
                        if line.startswith("WEB_PORT="):
                            lines[i] = f"WEB_PORT={port}\n"
                            port_updated = True
                            break
                    
                    if not port_updated:
                        lines.append(f"WEB_PORT={port}\n")
                    
                    # Write back
                    with open(env_file, "w") as f:
                        f.writelines(lines)
                else:
                    # Create new .env with port
                    with open(env_file, "w") as f:
                        f.write("# Baby Monitor System Environment Configuration\n")
                        f.write("MODE=normal\n")
                        f.write("CAMERA_INDEX=0\n")
                        f.write(f"WEB_PORT={port}\n")
                
                messagebox.showinfo("Success", 
                                 f"Web port set to: {port}\n\n" +
                                 "Restart the application for changes to take effect.")
                dialog.destroy()
            
            except ValueError as e:
                messagebox.showerror("Invalid Port", str(e))
        
        ttk.Button(btn_frame, text="Apply", command=set_port).pack(side=tk.RIGHT, padx=5)
    
    def test_web_server(self):
        """Test the web server functionality"""
        # Look for test_server.py
        test_server = SRC_DIR / "test_server.py"
        if test_server.exists():
            self.script_runner.run_script(test_server)
        else:
            # Try to run run_server.py
            run_server = SRC_DIR / "run_server.py"
            if run_server.exists():
                self.script_runner.run_script(run_server)
            else:
                messagebox.showerror("Error", "Could not find web server test script")


if __name__ == "__main__":
    app = UnifiedManagerGUI()
    
    # Check command line arguments to go directly to a specific section
    if len(sys.argv) > 1:
        if "--install" in sys.argv:
            app.show_install_section()
        elif "--repair" in sys.argv:
            app.show_repair_section()
        elif "--models" in sys.argv:
            app.show_training_section()
        elif "--config" in sys.argv:
            app.show_config_section()
    
    app.mainloop() 