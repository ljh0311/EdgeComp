#!/usr/bin/env python
"""
Create Backup Utility

This script creates a new backup of the baby monitor system,
saving critical files and configurations.
"""

import os
import sys
import shutil
import argparse
import time
from datetime import datetime

def create_backup(custom_name=None):
    """Create a new backup of the system."""
    # Define paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # Create timestamp for backup folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if custom_name:
        backup_name = f"backup_{custom_name}_{timestamp}"
    else:
        backup_name = f"backup_{timestamp}"
    
    backup_dir = os.path.join(parent_dir, backup_name)
    
    # Ensure the backup directory doesn't already exist
    if os.path.exists(backup_dir):
        print(f"Error: Backup directory '{backup_name}' already exists.")
        return False
    
    # Create the backup directory
    os.makedirs(backup_dir)
    
    # Define directories and files to backup
    backup_items = [
        "src",
        "tools",
        "tests",
        "requirements.txt",
        "README.md",
        "run_babymonitor.bat",
        "run_babymonitor.sh"
    ]
    
    # Define items to exclude from backup
    exclude_patterns = [
        "__pycache__",
        "*.pyc",
        "*.log",
        "venv",
        ".git",
        "backup_*"
    ]
    
    print(f"Creating backup '{backup_name}'...")
    
    # Copy each item to the backup directory
    for item in backup_items:
        src_path = os.path.join(parent_dir, item)
        dst_path = os.path.join(backup_dir, item)
        
        if not os.path.exists(src_path):
            print(f"Warning: {item} not found, skipping...")
            continue
        
        try:
            if os.path.isdir(src_path):
                shutil.copytree(
                    src_path, 
                    dst_path,
                    ignore=shutil.ignore_patterns(*exclude_patterns)
                )
                print(f"Copied directory: {item}")
            else:
                # Create parent directory if it doesn't exist
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"Copied file: {item}")
        except Exception as e:
            print(f"Error copying {item}: {str(e)}")
    
    # Create a backup info file
    info_file = os.path.join(backup_dir, "backup_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"Backup created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Backup name: {backup_name}\n")
        f.write("\nBackup contents:\n")
        for item in backup_items:
            if os.path.exists(os.path.join(backup_dir, item)):
                f.write(f"- {item}\n")
    
    print(f"\nBackup '{backup_name}' created successfully!")
    print(f"Location: {backup_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create a new backup of the baby monitor system.")
    parser.add_argument('-n', '--name', help="Custom name to add to the backup folder")
    args = parser.parse_args()
    
    create_backup(args.name)

if __name__ == "__main__":
    main() 