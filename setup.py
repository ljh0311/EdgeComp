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
import subprocess

def main():
    """Main entry point that launches the GUI installer."""
    print("Baby Monitor System - Installation")
    print("===================================")
    
    install_dir = Path("scripts/install")
    scripts_dir = Path("scripts")
    
    if not install_dir.exists():
        print("Error: Installation scripts not found.")
        print("Please make sure you are running this script from the project root directory.")
        sys.exit(1)
    
    args = sys.argv[1:]
    force_no_gui = "--no-gui" in args
    
    # Try GUI installer first
    scripts_manager_gui = scripts_dir / "scripts_manager_gui.py"
    if scripts_manager_gui.exists() and not force_no_gui:
        try:
            cmd = [sys.executable, str(scripts_manager_gui)] + args
            subprocess.run(cmd, check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error launching GUI: {e}")
            print("Falling back to alternative installer...")
    
    # Try standard installer
    install_py = install_dir / "install.py"
    if install_py.exists():
        try:
            filtered_args = [arg for arg in args if arg != "--no-gui"]
            subprocess.run([sys.executable, str(install_py)] + filtered_args, check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error launching installer: {e}")
            print("Trying platform-specific installers...")
    
    # Fall back to platform-specific installers
    system = platform.system()
    args_str = " ".join(args)
    
    if system == "Windows":
        installer = install_dir / "install.bat"
        if installer.exists():
            os.system(f"{installer} {args_str}")
        else:
            print("Error: No suitable installer found for Windows.")
            sys.exit(1)
            
    elif system in ("Linux", "Darwin"):
        installer = install_dir / "install.sh"
        if installer.exists():
            os.system(f"bash {installer} {args_str}")
        else:
            print(f"Error: No suitable installer found for {system}.")
            sys.exit(1)
            
    else:
        print(f"Error: Unsupported platform: {system}")
        sys.exit(1)

if __name__ == "__main__":
    main()
