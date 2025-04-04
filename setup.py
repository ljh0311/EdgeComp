"""
Baby Monitor System
==================
Main setup script that launches the GUI installer with options for:
- Install/Reinstall the system
- Build/Download models
- Fix common issues
- Configure system
- Check and train models
"""

import sys
import os
import platform
import shutil
from pathlib import Path
import importlib.util
import subprocess
import argparse
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BabyMonitorSetup")

# Constants
MODELS_DIR = Path("src/babymonitor/models")

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

def check_models():
    """Check if required models are available."""
    model_paths = {
        "emotion_model": "src/babymonitor/models/emotion_model.pt",
        "person_detection": "src/babymonitor/models/yolov8n.pt",
        "wav2vec2_model": "src/babymonitor/models/wav2vec2_emotion.pt"
    }
    
    missing_models = []
    existing_models = []
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
            existing_models.append((model_name, f"{model_size:.1f}"))
        else:
            missing_models.append(model_name)
    
    return existing_models, missing_models

def train_models(models_to_train):
    """Train specified models."""
    results = []
    train_script_map = {
        "emotion_model": "src/babymonitor/emotion/train_emotion_model.py",
        "wav2vec2_model": "models/training/custom_speechbrain.py"
    }
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for model in models_to_train:
        print(f"\n{'='*50}")
        print(f"TRAINING: {model}")
        print(f"{'='*50}")
        
        if model in train_script_map:
            script_path = train_script_map[model]
            
            if os.path.exists(script_path):
                print(f"Using training script: {script_path}")
                try:
                    cmd = [sys.executable, script_path]
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    results.append((model, "SUCCESS", None))
                    
                    # Move trained model to correct location
                    if model == "emotion_model" and os.path.exists("models/best_emotion_model.pt"):
                        os.makedirs("src/babymonitor/models", exist_ok=True)
                        shutil.copy("models/best_emotion_model.pt", "src/babymonitor/models/emotion_model.pt")
                        results.append((model, "MOVED", "src/babymonitor/models/emotion_model.pt"))
                    
                    elif model == "wav2vec2_model" and os.path.exists("results"):
                        # Find the latest model in results directory
                        result_dirs = [d for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))]
                        if result_dirs:
                            latest_dir = sorted(result_dirs)[-1]
                            model_path = os.path.join("results", latest_dir, "save/ckpt/model.ckpt")
                            if os.path.exists(model_path):
                                os.makedirs("src/babymonitor/models", exist_ok=True)
                                shutil.copy(model_path, "src/babymonitor/models/wav2vec2_emotion.pt")
                                results.append((model, "MOVED", "src/babymonitor/models/wav2vec2_emotion.pt"))
                
                except subprocess.CalledProcessError as e:
                            results.append((model, "FAILED", f"Error code: {e.returncode}, {e.stderr}"))
            else:
                results.append((model, "SKIPPED", f"Training script not found: {script_path}"))
        else:
            results.append((model, "SKIPPED", f"No training script available for {model}"))
    
    return results

def download_pretrained_models():
    """Download pretrained models if needed."""
    model_urls = {
        "person_detection": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "emotion_model": "https://github.com/speechbrain/speechbrain/releases/download/0.5.12/emotion-recognition-wav2vec2-IEMOCAP.ckpt",
        "wav2vec2_model": "https://github.com/speechbrain/speechbrain/releases/download/0.5.12/wav2vec2-base-emotion-recognition.ckpt"
    }
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for model_name, url in model_urls.items():
        target_dir = "src/babymonitor/models"
        os.makedirs(target_dir, exist_ok=True)
        
        if model_name == "person_detection":
            target_path = os.path.join(target_dir, "yolov8n.pt")
        elif model_name == "emotion_model":
            target_path = os.path.join(target_dir, "emotion_model.pt")
        elif model_name == "wav2vec2_model":
            target_path = os.path.join(target_dir, "wav2vec2_emotion.pt")
        
        if not os.path.exists(target_path):
            print(f"\nDownloading {model_name} from {url}...")
            try:
                # Use appropriate download method based on platform
                if platform.system() == "Windows":
                    try:
                        # First try with PowerShell
                        subprocess.run([
                            "powershell", "-Command",
                            f"(New-Object System.Net.WebClient).DownloadFile('{url}', '{target_path}')"
                        ], check=True)
                    except subprocess.CalledProcessError:
                        # If PowerShell fails, try with curl which is available in newer Windows versions
                        subprocess.run(["curl", "-L", "-o", target_path, url], check=True)
                else:
                    # For Linux/macOS, try wget first, then curl
                    try:
                        subprocess.run(["wget", "-O", target_path, url], check=True)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        subprocess.run(["curl", "-L", "-o", target_path, url], check=True)
                
                results.append((model_name, "SUCCESS", target_path))
            except Exception as e:
                results.append((model_name, "FAILED", str(e)))
        else:
            results.append((model_name, "SKIPPED", f"Model already exists at {target_path}"))
    
    return results

def print_model_status():
    """Print the status of all required models."""
    existing_models, missing_models = check_models()
    
    print("\nMODEL STATUS:")
    print("=" * 50)
    
    if existing_models:
        print("Available models:")
        for model_name, model_size in existing_models:
            print(f"  ✓ {model_name} ({model_size} MB)")
    
    if missing_models:
        print("\nMissing models:")
        for model in missing_models:
            print(f"  ✗ {model}")
    
    print("=" * 50)
    return missing_models

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Baby Monitor System Setup")
    
    # Main modes
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--install", action="store_true", help="Install the system")
    parser.add_argument("--models", action="store_true", help="Check and manage models")
    parser.add_argument("--fix", action="store_true", help="Fix common issues")
    parser.add_argument("--config", action="store_true", help="Configure the system")
    
    # Model-specific options
    parser.add_argument("--train", action="store_true", help="Train missing models")
    parser.add_argument("--download", action="store_true", help="Download pretrained models")
    parser.add_argument("--specific-model", choices=["emotion_model", "person_detection", "wav2vec2_model"], 
                        help="Specify a particular model to train or download")
    parser.add_argument("--train-all", action="store_true", help="Train all models even if they exist")
    
    # Installation mode
    parser.add_argument("--mode", choices=["normal", "dev"], default="normal", 
                        help="Installation mode: normal for standard users, dev for developers")
    
    # Other options
    parser.add_argument("--repair-env", action="store_true", help="Repair Python environment")
    parser.add_argument("--reset", action="store_true", help="Reset all configurations")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment checks")
    
    return parser.parse_args()

def run_platform_specific_installer(args_str=""):
    """Run the appropriate platform-specific installer."""
    system = platform.system()
    install_dir = Path("scripts/install")
    
    print(f"Detected platform: {system}")
    
    if system == "Windows":
        # Use Windows batch file
        installer = install_dir / "install.bat"
        if installer.exists():
            print("Running Windows installer...")
            return os.system(f"{installer} {args_str}")
        else:
            print("Error: No suitable installer found for Windows.")
            return 1
    
    elif system == "Linux":
        # Check if it's a Raspberry Pi
        if is_raspberry_pi():
            # Use Raspberry Pi installer
            installer = install_dir / "install_pi.sh"
            if installer.exists():
                print("Detected Raspberry Pi, running specialized installer...")
                return os.system(f"bash {installer} {args_str}")
            else:
                installer = install_dir / "install.sh"
                if installer.exists():
                    print("Running Linux installer...")
                    return os.system(f"bash {installer} {args_str}")
                else:
                    print("Error: No suitable installer found for Raspberry Pi.")
                    return 1
        else:
            # Use Linux installer
            installer = install_dir / "install.sh"
            if installer.exists():
                print("Running Linux installer...")
                return os.system(f"bash {installer} {args_str}")
            else:
                print("Error: No suitable installer found for Linux.")
                return 1
    
    elif system == "Darwin":  # macOS
        # Use Linux/Mac installer
        installer = install_dir / "install.sh"
        if installer.exists():
            print("Running macOS installer...")
            return os.system(f"bash {installer} {args_str}")
        else:
            print("Error: No suitable installer found for macOS.")
            return 1
    
    else:
        print(f"Error: Unsupported platform: {system}")
        return 1

def main():
    """Main entry point that handles installation and model management."""
    args = parse_args()
    
    print("\n" + "="*50)
    print(" Baby Monitor System - Installation & Setup Utility ")
    print("="*50 + "\n")
    
    install_dir = Path("scripts/install")
    scripts_dir = Path("scripts")
    
    # Check if installation scripts exist
    if not install_dir.exists():
        print("Error: Installation scripts not found.")
        print("Please make sure you are running this script from the project root directory.")
        sys.exit(1)
    
    # Create args_str for platform-specific installers
    args_list = []
    if args.install:
        args_list.append("--install")
    if args.models:
        args_list.append("--models")
    if args.fix:
        args_list.append("--fix")
    if args.config:
        args_list.append("--config")
    if args.no_gui:
        args_list.append("--no-gui")
    if args.train:
        args_list.append("--train")
    if args.download:
        args_list.append("--download")
    if args.mode:
        args_list.append(f"--mode={args.mode}")
    if args.repair_env:
        args_list.append("--repair-env")
    if args.reset:
        args_list.append("--reset")
    
    args_str = " ".join(args_list)
    
    # Handle model management first if requested
    handled_model_actions = False
    
    if args.models or args.train or args.download:
        missing_models = print_model_status()
        handled_model_actions = True
        
        # Handle specific model actions
        if args.specific_model:
            if args.download:
                print(f"\nDownloading specific model: {args.specific_model}...")
                model_urls = {
                    "person_detection": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                    "emotion_model": "https://github.com/speechbrain/speechbrain/releases/download/0.5.12/emotion-recognition-wav2vec2-IEMOCAP.ckpt",
                    "wav2vec2_model": "https://github.com/speechbrain/speechbrain/releases/download/0.5.12/wav2vec2-base-emotion-recognition.ckpt"
                }
                
                if args.specific_model in model_urls:
                    target_dir = "src/babymonitor/models"
                    os.makedirs(target_dir, exist_ok=True)
                    
                    if args.specific_model == "person_detection":
                        target_path = os.path.join(target_dir, "yolov8n.pt")
                    elif args.specific_model == "emotion_model":
                        target_path = os.path.join(target_dir, "emotion_model.pt")
                    elif args.specific_model == "wav2vec2_model":
                        target_path = os.path.join(target_dir, "wav2vec2_emotion.pt")
                    
                    url = model_urls[args.specific_model]
                    
                    try:
                        # Use appropriate download method based on platform
                        if platform.system() == "Windows":
                            try:
                                subprocess.run([
                                    "powershell", "-Command",
                                    f"(New-Object System.Net.WebClient).DownloadFile('{url}', '{target_path}')"
                                ], check=True)
                            except subprocess.CalledProcessError:
                                subprocess.run(["curl", "-L", "-o", target_path, url], check=True)
                        else:
                            try:
                                subprocess.run(["wget", "-O", target_path, url], check=True)
                            except (subprocess.CalledProcessError, FileNotFoundError):
                                subprocess.run(["curl", "-L", "-o", target_path, url], check=True)
                        
                        print(f"Successfully downloaded {args.specific_model} to {target_path}")
                    except Exception as e:
                        print(f"Error downloading {args.specific_model}: {e}")
            
            if args.train:
                print(f"\nTraining specific model: {args.specific_model}...")
                train_results = train_models([args.specific_model])
                
                print("\nTraining results:")
                for model, status, message in train_results:
                    status_symbol = "✓" if status in ["SUCCESS", "MOVED"] else "✗"
                    print(f"  {status_symbol} {model}: {status}")
                    if message:
                        print(f"     {message}")
        
        elif args.download:
            print("\nDownloading all pretrained models...")
            results = download_pretrained_models()
            
            print("\nDownload results:")
            for model, status, message in results:
                status_symbol = "✓" if status == "SUCCESS" else "✗"
                print(f"  {status_symbol} {model}: {status}")
                if message and status != "SUCCESS":
                    print(f"     {message}")
            
            # Refresh model status
            missing_models = print_model_status()
        
        elif args.train:
            models_to_train = missing_models
            
            if args.train_all:
                # Train all models even if they exist
                models_to_train = ["emotion_model", "wav2vec2_model"]
            
            if models_to_train:
                print("\nTraining models...")
                results = train_models(models_to_train)
                
                print("\nTraining results:")
                for model, status, message in results:
                    status_symbol = "✓" if status in ["SUCCESS", "MOVED"] else "✗"
                    print(f"  {status_symbol} {model}: {status}")
                    if message:
                        print(f"     {message}")
                
                # Refresh model status
                print_model_status()
            else:
                print("\nNo models need to be trained.")
    
    # If only model actions were requested, exit
    if handled_model_actions and not args.install and not args.fix and not args.config:
        print("\nModel management complete. Exiting.")
        return
    
    # Handle repair environment action
    if args.repair_env:
        install_py = install_dir / "install.py"
        if install_py.exists():
            print("\nRepairing Python environment...")
            try:
                subprocess.run([sys.executable, str(install_py), "--repair-env", "--no-gui"], check=True)
                print("Environment repair completed.")
                if not (args.install or args.models or args.fix or args.config or args.reset):
                    return
            except subprocess.CalledProcessError as e:
                print(f"Error repairing environment: {e}")
                print("Trying platform-specific installer...")
                exit_code = run_platform_specific_installer(f"--repair-env {args_str}")
                if not (args.install or args.models or args.fix or args.config or args.reset):
                    sys.exit(exit_code)
    
    # Handle reset action
    if args.reset:
        install_py = install_dir / "install.py"
        if install_py.exists():
            print("\nResetting system configuration...")
            try:
                subprocess.run([sys.executable, str(install_py), "--reset", "--no-gui"], check=True)
                print("System reset completed.")
                if not (args.install or args.fix or args.config):
                    return
            except subprocess.CalledProcessError as e:
                print(f"Error resetting system: {e}")
                print("Trying platform-specific installer...")
                exit_code = run_platform_specific_installer(f"--reset {args_str}")
                if not (args.install or args.fix or args.config):
                    sys.exit(exit_code)
    
    # For other actions, use the GUI installer if possible
    if not args.no_gui:
        # Check if scripts_manager_gui.py exists (comprehensive GUI)
        scripts_manager_gui = scripts_dir / "scripts_manager_gui.py"
        if scripts_manager_gui.exists():
            print("Launching comprehensive GUI installer...")
            try:
                # Use subprocess to run with proper Python interpreter
                cmd = [sys.executable, str(scripts_manager_gui)]
                
                # Add appropriate argument to go directly to the right section
                if args.install:
                    cmd.append("--install")
                elif args.models:
                    cmd.append("--models")
                elif args.fix:
                    cmd.append("--repair")
                elif args.config:
                    cmd.append("--config")
                
                # Add any other arguments except those we already processed
                for arg in sys.argv[1:]:
                    if arg not in ["--no-gui", "--train", "--download", "--specific-model", "--train-all", 
                                   "--repair-env", "--reset", "--skip-checks"]:
                        cmd.append(arg)
                
                subprocess.run(cmd, check=True)
                return
            except subprocess.CalledProcessError as e:
                print(f"Error launching GUI: {e}")
                print("Falling back to alternative installer...")
            except FileNotFoundError:
                print(f"Error: Could not find Python executable: {sys.executable}")
                print("Falling back to alternative installer...")
        
        # Try install.py with GUI if available
        install_py = install_dir / "install.py"
        if install_py.exists():
            print("Launching installation wizard...")
            try:
                # Pass through any arguments except those we already processed
                filtered_args = [arg for arg in sys.argv[1:] if arg not in ["--no-gui", "--train", 
                                                                          "--download", "--specific-model", 
                                                                          "--train-all", "--repair-env", 
                                                                          "--reset", "--skip-checks"]]
                subprocess.run([sys.executable, str(install_py)] + filtered_args, check=True)
                return
            except subprocess.CalledProcessError as e:
                print(f"Error launching installer: {e}")
                print("Trying platform-specific installers...")
            except FileNotFoundError:
                print(f"Error: Could not find Python executable: {sys.executable}")
                print("Trying platform-specific installers...")
    
    # Fall back to platform-specific installers
    exit_code = run_platform_specific_installer(args_str)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
