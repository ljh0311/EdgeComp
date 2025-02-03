import platform
import subprocess
import sys
import os

def is_raspberry_pi():
    """Check if the system is a Raspberry Pi"""
    return platform.machine() in ('armv7l', 'aarch64')

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n📍 {description}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        return False

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_name = "venv"
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists(venv_name):
        print("\n🔧 Creating virtual environment...")
        run_command(f"python{' -m' if not is_raspberry_pi() else '3 -m'} venv {venv_name}", 
                   "Virtual environment creation")

    # Activation command varies by platform
    if platform.system() == "Windows":
        activate_script = os.path.join(venv_name, "Scripts", "activate")
    else:
        activate_script = os.path.join(venv_name, "bin", "activate")
    
    print(f"\n⚠️ Please activate the virtual environment manually:")
    if platform.system() == "Windows":
        print(f".\\{venv_name}\\Scripts\\activate")
    else:
        print(f"source {venv_name}/bin/activate")

def install_raspberry_pi_dependencies():
    """Install Raspberry Pi specific dependencies"""
    commands = [
        ("sudo apt update", "Updating package list"),
        ("sudo apt install -y python3-picamera2", "Installing picamera2"),
        ("sudo apt install -y python3-pyaudio portaudio19-dev", "Installing audio dependencies"),
        ("pip3 install ultralytics opencv-python numpy pyaudio", "Installing Python packages")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def install_windows_dependencies():
    """Install Windows specific dependencies"""
    commands = [
        ("pip install ultralytics opencv-python numpy", "Installing core packages"),
        ("pip install pipwin", "Installing pipwin"),
        ("pipwin install pyaudio", "Installing PyAudio")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def main():
    print("🔍 Detecting system configuration...")
    is_pi = is_raspberry_pi()
    system = "Raspberry Pi" if is_pi else "Windows"
    print(f"📌 Detected system: {system}")

    # Setup virtual environment
    setup_virtual_environment()

    # Wait for user to activate virtual environment
    input("\n⚠️ Please activate the virtual environment and press Enter to continue...")

    print("\n🚀 Starting installation process...")
    
    success = install_raspberry_pi_dependencies() if is_pi else install_windows_dependencies()
    
    if success:
        print("\n✅ Installation completed successfully!")
        print("\nℹ️ You can now run the baby monitor with:")
        print("python main.py")
    else:
        print("\n❌ Installation failed. Please check the errors above and try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        sys.exit(1) 