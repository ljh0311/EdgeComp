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
        
        system_label = ttk.Label(header_frame, text=f"Detected: {system_info}")
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
        
        # Open web button
        web_repair_button = ttk.Button(buttons_frame, text="Web Repair Tools",
                                    command=self.open_web_repair,
                                    style="Action.TButton")
        web_repair_button.pack(side=tk.LEFT, padx=10)
        
        # Save reference
        self.welcome_frame = welcome_frame
    
    def create_content_frames(self):
        """Create frames that will be shown in the content pane"""
        # Create the install section
        self.install_frame = ttk.LabelFrame(self.content_pane, text="Installation")
        
        # Create the repair section
        self.repair_frame = ttk.LabelFrame(self.content_pane, text="Repair Tools")
        
        # Create the output frame
        self.output_frame = ttk.LabelFrame(self.content_pane, text="Output")
        
        # Configure install frame
        self.install_frame.columnconfigure(0, weight=1)
        self.install_frame.rowconfigure(0, weight=1)
        
        # Create notebook for install tabs
        self.install_notebook = ttk.Notebook(self.install_frame)
        self.install_notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create tabs for different platform installs
        self.windows_install_tab = ttk.Frame(self.install_notebook)
        self.linux_install_tab = ttk.Frame(self.install_notebook)
        self.pi_install_tab = ttk.Frame(self.install_notebook)
        
        # Add tabs based on platform
        self.install_notebook.add(self.windows_install_tab, text="Windows Install")
        if self.system != "Windows":
            self.install_notebook.add(self.linux_install_tab, text="Linux Install")
        if self.is_raspberry_pi:
            self.install_notebook.add(self.pi_install_tab, text="Raspberry Pi Install")
        
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
        
        # Configure repair frame
        self.repair_frame.columnconfigure(0, weight=1)
        self.repair_frame.rowconfigure(0, weight=1)
        
        # Create notebook for repair tabs
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
        version_label = ttk.Label(status_frame, text="v1.0.0")
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
        
        # Select appropriate install tab based on platform
        if self.system == "Windows":
            self.install_notebook.select(self.windows_install_tab)
        elif self.is_raspberry_pi:
            self.install_notebook.select(self.pi_install_tab)
        else:
            self.install_notebook.select(self.linux_install_tab)
        
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
        for widget in self.windows_install_frame.winfo_children():
            widget.destroy()
        for widget in self.linux_install_frame.winfo_children():
            widget.destroy()
        for widget in self.pi_install_frame.winfo_children():
            widget.destroy()
        for widget in self.camera_repair_frame.winfo_children():
            widget.destroy()
        for widget in self.audio_repair_frame.winfo_children():
            widget.destroy()
        for widget in self.system_repair_frame.winfo_children():
            widget.destroy()
        
        # Install scripts
        windows_install_scripts = []
        linux_install_scripts = []
        pi_install_scripts = []
        
        # Repair scripts
        camera_repair_scripts = []
        audio_repair_scripts = []
        system_repair_scripts = []
        
        # Scan install directory
        if INSTALL_DIR.exists():
            for script in INSTALL_DIR.glob("*"):
                if script.is_file() and script.name != "README.md":
                    script_name = script.name.lower()
                    if script.suffix == ".bat" or "windows" in script_name:
                        windows_install_scripts.append(script)
                    elif "pi" in script_name or "raspberry" in script_name:
                        pi_install_scripts.append(script)
                    elif script.suffix == ".sh" or "linux" in script_name or "install.py" in script_name:
                        linux_install_scripts.append(script)
        
        # Scan main scripts directory for fix scripts
        for script in SCRIPTS_DIR.glob("*"):
            if script.is_file() and (script.name.startswith("fix_") or script.name.startswith("restart_")):
                script_name = script.name.lower()
                if "camera" in script_name or "video" in script_name:
                    camera_repair_scripts.append(script)
                elif "audio" in script_name or "sound" in script_name:
                    audio_repair_scripts.append(script)
                else:
                    system_repair_scripts.append(script)
                    
        # Add the installer.py to all install sections
        installer_path = INSTALL_DIR / "install.py"
        if installer_path.exists():
            windows_install_scripts.append(installer_path)
            linux_install_scripts.append(installer_path)
            pi_install_scripts.append(installer_path)
        
        # Add Windows install script buttons
        self._add_section_heading(self.windows_install_frame, "Windows Installation Scripts")
        self._add_script_buttons(self.windows_install_frame, windows_install_scripts)
        
        # Add Linux install script buttons
        self._add_section_heading(self.linux_install_frame, "Linux Installation Scripts")
        self._add_script_buttons(self.linux_install_frame, linux_install_scripts)
        
        # Add Pi install script buttons
        self._add_section_heading(self.pi_install_frame, "Raspberry Pi Installation Scripts")
        self._add_script_buttons(self.pi_install_frame, pi_install_scripts)
        
        # Add Camera repair script buttons
        self._add_section_heading(self.camera_repair_frame, "Camera Repair Tools")
        self._add_script_buttons(self.camera_repair_frame, camera_repair_scripts)
        
        # Add Audio repair script buttons
        self._add_section_heading(self.audio_repair_frame, "Audio Repair Tools")
        self._add_script_buttons(self.audio_repair_frame, audio_repair_scripts)
        
        # Add System repair script buttons
        self._add_section_heading(self.system_repair_frame, "System Repair Tools")
        self._add_script_buttons(self.system_repair_frame, system_repair_scripts)
        
        # Add unified installer to all install sections
        self._add_unified_installer_button(self.windows_install_frame)
        self._add_unified_installer_button(self.linux_install_frame)
        self._add_unified_installer_button(self.pi_install_frame)
        
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


if __name__ == "__main__":
    app = UnifiedManagerGUI()
    app.mainloop() 