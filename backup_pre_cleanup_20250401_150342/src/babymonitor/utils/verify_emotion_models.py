"""
Verify Emotion Models
====================
This script checks if all emotion models are properly installed and provides a report.
It examines both the global models directory and the application models directory.

Usage:
    python -m src.babymonitor.utils.verify_emotion_models
"""

import os
from pathlib import Path
import argparse
import sys

def check_models_directory(directory_path, indent=""):
    """Check a models directory and return a dictionary of found models."""
    print(f"{indent}Checking directory: {directory_path}")
    
    if not directory_path.exists():
        print(f"{indent}Directory does not exist!")
        return {}
    
    found_models = {}
    total_files = 0
    
    # Check all files in the directory
    for item in directory_path.iterdir():
        if item.is_file():
            total_files += 1
            file_size = item.stat().st_size / (1024 * 1024)  # Convert to MB
            found_models[item.name] = {
                "path": str(item),
                "size": f"{file_size:.2f} MB",
                "type": item.suffix
            }
            print(f"{indent}- {item.name} ({file_size:.2f} MB)")
        elif item.is_dir():
            # Recursively check subdirectories
            print(f"{indent}+ {item.name}/")
            subdir_models = check_models_directory(item, indent + "  ")
            for model_name, model_info in subdir_models.items():
                found_models[f"{item.name}/{model_name}"] = model_info
                total_files += 1
    
    if total_files == 0:
        print(f"{indent}No files found in this directory.")
    else:
        print(f"{indent}Total files: {total_files}")
    
    return found_models

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Verify emotion models installation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show more detailed information")
    args = parser.parse_args()
    
    # Get current file location and calculate project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    
    # Project paths
    global_models_dir = project_root / "models"
    emotion_models_dir = global_models_dir / "emotion"
    
    app_models_dir = current_dir.parent / "models"
    app_emotion_dir = app_models_dir / "emotion"
    
    # Print header
    print("=================================================")
    print("Emotion Models Verification Report")
    print("=================================================")
    print(f"Project Root: {project_root}")
    print()
    
    # Check global models directory
    print("1. Checking Global Models Directory")
    print("-------------------------------------------------")
    if global_models_dir.exists():
        global_models = check_models_directory(emotion_models_dir)
        global_model_count = len(global_models)
    else:
        print("Global models directory not found!")
        global_model_count = 0
    print()
    
    # Check application models directory
    print("2. Checking Application Models Directory")
    print("-------------------------------------------------")
    if app_models_dir.exists() and app_emotion_dir.exists():
        app_models = check_models_directory(app_emotion_dir)
        app_model_count = len(app_models)
    else:
        if not app_models_dir.exists():
            print("Application models directory not found!")
        elif not app_emotion_dir.exists():
            print("Application emotion models directory not found!")
        app_model_count = 0
    print()
    
    # Print summary
    print("=================================================")
    print("Summary")
    print("=================================================")
    print(f"Global emotion models found: {global_model_count}")
    print(f"Application emotion models found: {app_model_count}")
    
    # Check if models are missing
    if global_model_count > app_model_count:
        print("\nWARNING: Some emotion models are not properly installed!")
        print("The following models were found in the global directory but not in the application directory:")
        
        for model_name, model_info in global_models.items():
            model_key = model_name.split('/')[-1]  # Get just the filename
            found = False
            
            for app_model_name in app_models.keys():
                if model_key in app_model_name:
                    found = True
                    break
            
            if not found:
                print(f"- {model_name} ({model_info['size']})")
        
        print("\nRecommendation:")
        print("Run the setup.py script with the --download-models option to fix this issue:")
        print("  python setup.py --download-models")
        print("\nOr run the setup_models.py utility directly:")
        print("  python -m src.babymonitor.utils.setup_models")
    else:
        print("\nAll emotion models appear to be properly installed!")
    
if __name__ == "__main__":
    main() 