"""
Baby Monitor System
==================
Main setup script that detects the platform and runs the appropriate installer.
"""

import sys
import os
import platform
from pathlib import Path

def is_raspberry_pi():
    """Check if the system is a Raspberry Pi."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            return any("Raspberry Pi" in line for line in f)
    except:
        return False

def main():
    """Main entry point that detects platform and runs appropriate installer."""
    print("Baby Monitor System - Installation")
    print("===================================")
    
    install_dir = Path("scripts/install")
    
    # Check if installation scripts exist
    if not install_dir.exists():
        print("Error: Installation scripts not found.")
        print("Please make sure you are running this script from the project root directory.")
        sys.exit(1)
    
    # Get command line arguments to pass to the installer
    args = sys.argv[1:]
    args_str = " ".join(args)
    
    # Detect platform and run appropriate installer
    system = platform.system()
    print(f"Detected platform: {system}")
    
    if system == "Windows":
        # Use Windows batch file
        installer = install_dir / "install.bat"
        if installer.exists():
            print("Running Windows installer...")
            os.system(f"{installer} {args_str}")
        else:
            # Fallback to Python installer
            print("Windows-specific installer not found, using generic installer...")
            python_installer = install_dir / "install.py"
            os.system(f"python {python_installer} {args_str}")
    
    elif system == "Linux":
        # Check if it's a Raspberry Pi
        if is_raspberry_pi():
            # Use Raspberry Pi installer
            installer = install_dir / "install_pi.sh"
            if installer.exists():
                print("Detected Raspberry Pi, running specialized installer...")
                os.system(f"bash {installer} {args_str}")
            else:
                # Fallback to Linux installer
                installer = install_dir / "install.sh"
                if installer.exists():
                    print("Running Linux installer...")
                    os.system(f"bash {installer} {args_str}")
                else:
                    # Fallback to Python installer
                    print("Linux-specific installer not found, using generic installer...")
                    python_installer = install_dir / "install.py"
                    os.system(f"python {python_installer} {args_str}")
        else:
            # Use Linux installer
            installer = install_dir / "install.sh"
            if installer.exists():
                print("Running Linux installer...")
                os.system(f"bash {installer} {args_str}")
            else:
                # Fallback to Python installer
                print("Linux-specific installer not found, using generic installer...")
                python_installer = install_dir / "install.py"
                os.system(f"python {python_installer} {args_str}")
    
    elif system == "Darwin":  # macOS
        # Use Linux/Mac installer
        installer = install_dir / "install.sh"
        if installer.exists():
            print("Running macOS installer...")
            os.system(f"bash {installer} {args_str}")
        else:
            # Fallback to Python installer
            print("macOS-specific installer not found, using generic installer...")
            python_installer = install_dir / "install.py"
            os.system(f"python {python_installer} {args_str}")
    
    else:
        # Unknown platform, use Python installer
        print(f"Unknown platform: {system}, using generic installer...")
        python_installer = install_dir / "install.py"
        os.system(f"python {python_installer} {args_str}")

if __name__ == "__main__":
    main()
