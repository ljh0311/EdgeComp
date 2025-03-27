"""
Baby Monitor System
==================
Main setup script that launches the GUI installer with options for:
- Install/Reinstall the system
- Build/Download models
- Fix common issues
- Configure system
"""

import sys
import os
import platform
from pathlib import Path
import importlib.util
import subprocess

def is_raspberry_pi():
    """Check if the system is a Raspberry Pi."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            return any("Raspberry Pi" in line for line in f)
    except:
        return False

def import_setup_module(path):
    """Import setup.py as a module to directly access its functions."""
    spec = importlib.util.spec_from_file_location("setup_module", path)
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)
    return setup_module

def main():
    """Main entry point that launches the GUI installer."""
    print("Baby Monitor System - Installation")
    print("===================================")
    
    install_dir = Path("scripts/install")
    scripts_dir = Path("scripts")
    
    # Check if installation scripts exist
    if not install_dir.exists():
        print("Error: Installation scripts not found.")
        print("Please make sure you are running this script from the project root directory.")
        sys.exit(1)
    
    # Get command line arguments
    args = sys.argv[1:]
    force_no_gui = "--no-gui" in args
    
    # Process specific mode requests
    install_mode = False
    model_mode = False
    fix_mode = False
    config_mode = False
    
    if "--install" in args:
        install_mode = True
    elif "--models" in args:
        model_mode = True
    elif "--fix" in args:
        fix_mode = True
    elif "--config" in args:
        config_mode = True
    
    # If scripts_manager_gui.py exists, use it as default (comprehensive GUI)
    scripts_manager_gui = scripts_dir / "scripts_manager_gui.py"
    if scripts_manager_gui.exists() and not force_no_gui:
        print("Launching comprehensive GUI installer...")
        try:
            # Use subprocess to run with proper Python interpreter
            cmd = [sys.executable, str(scripts_manager_gui)]
            
            # Add appropriate argument to go directly to the right section
            if install_mode:
                cmd.append("--install")
            elif model_mode:
                cmd.append("--models")
            elif fix_mode:
                cmd.append("--repair")
            elif config_mode:
                cmd.append("--config")
            
            # Add any other arguments
            for arg in args:
                if arg not in ["--install", "--models", "--fix", "--config", "--no-gui"]:
                    cmd.append(arg)
            
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            print(f"Error launching GUI: {e}")
            print("Falling back to alternative installer...")
        except FileNotFoundError:
            print(f"Error: Could not find Python executable: {sys.executable}")
            print("Falling back to alternative installer...")
    
    # Otherwise use install.py with GUI as fallback
    install_py = install_dir / "install.py"
    if install_py.exists():
        print("Launching installation wizard...")
        try:
            # Pass through any arguments except --no-gui which would have been handled above
            filtered_args = [arg for arg in args if arg != "--no-gui"]
            subprocess.run([sys.executable, str(install_py)] + filtered_args, check=True)
            return
        except subprocess.CalledProcessError as e:
            print(f"Error launching installer: {e}")
            print("Trying platform-specific installers...")
        except FileNotFoundError:
            print(f"Error: Could not find Python executable: {sys.executable}")
            print("Trying platform-specific installers...")
    
    # If all else fails, use platform-specific installers
    system = platform.system()
    print(f"Detected platform: {system}")
    args_str = " ".join(args)
    
    if system == "Windows":
        # Use Windows batch file
        installer = install_dir / "install.bat"
        if installer.exists():
            print("Running Windows installer...")
            os.system(f"{installer} {args_str}")
        else:
            print("Error: No suitable installer found for Windows.")
            sys.exit(1)
    
    elif system == "Linux":
        # Check if it's a Raspberry Pi
        if is_raspberry_pi():
            # Use Raspberry Pi installer
            installer = install_dir / "install_pi.sh"
            if installer.exists():
                print("Detected Raspberry Pi, running specialized installer...")
                os.system(f"bash {installer} {args_str}")
            else:
                installer = install_dir / "install.sh"
                if installer.exists():
                    print("Running Linux installer...")
                    os.system(f"bash {installer} {args_str}")
                else:
                    print("Error: No suitable installer found for Raspberry Pi.")
                    sys.exit(1)
        else:
            # Use Linux installer
            installer = install_dir / "install.sh"
            if installer.exists():
                print("Running Linux installer...")
                os.system(f"bash {installer} {args_str}")
            else:
                print("Error: No suitable installer found for Linux.")
                sys.exit(1)
    
    elif system == "Darwin":  # macOS
        # Use Linux/Mac installer
        installer = install_dir / "install.sh"
        if installer.exists():
            print("Running macOS installer...")
            os.system(f"bash {installer} {args_str}")
        else:
            print("Error: No suitable installer found for macOS.")
            sys.exit(1)
    
    else:
        print(f"Error: Unsupported platform: {system}")
        sys.exit(1)

if __name__ == "__main__":
    main()
