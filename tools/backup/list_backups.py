#!/usr/bin/env python
"""
List Backups Utility

This script lists all available backup folders in the baby monitor system
with detailed information about each backup.
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime

def get_directory_size(path):
    """Calculate the total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def format_size(size_bytes):
    """Format a size in bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def list_backups(verbose=False):
    """List all backup folders and their details."""
    # Define the backup directory, relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    backup_dir = os.path.join(parent_dir, "backup_*")
    
    # Find all backup folders
    backup_folders = glob.glob(backup_dir)
    
    if not backup_folders:
        print("No backup folders found.")
        return
    
    # Sort backups by date (newest first)
    backup_folders.sort(reverse=True)
    
    # Print header
    if verbose:
        print(f"{'Backup ID':<25} {'Date':<20} {'Size':<10} {'Files':<10}")
        print("-" * 65)
    else:
        print(f"{'Backup ID':<25} {'Date':<20}")
        print("-" * 45)
    
    # Print each backup
    for folder in backup_folders:
        folder_name = os.path.basename(folder)
        
        # Parse the date from the folder name
        try:
            date_part = folder_name.split('_', 1)[1]
            date_obj = datetime.strptime(date_part, "%Y%m%d_%H%M%S")
            date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            date_str = "Unknown date"
        
        if verbose:
            # Calculate folder size and file count
            size = get_directory_size(folder)
            file_count = sum(len(files) for _, _, files in os.walk(folder))
            
            print(f"{folder_name:<25} {date_str:<20} {format_size(size):<10} {file_count:<10}")
        else:
            print(f"{folder_name:<25} {date_str:<20}")

def main():
    parser = argparse.ArgumentParser(description="List available backup folders.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show detailed information about each backup")
    args = parser.parse_args()
    
    list_backups(args.verbose)

if __name__ == "__main__":
    main() 