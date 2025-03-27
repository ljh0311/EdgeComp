#!/usr/bin/env python
"""
Backup Cleanup Utility

This script manages backup folders by removing older backups while keeping
a specified number of the most recent ones.
"""

import os
import re
import shutil
import argparse
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cleanup_backups')

def get_backup_folders(base_dir='.'):
    """Find all backup folders in the given directory."""
    backup_pattern = re.compile(r'backup_(\d{8})_(\d{6})')
    backup_folders = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and backup_pattern.match(item):
            # Extract timestamp from folder name
            match = backup_pattern.match(item)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                timestamp_str = f"{date_str}_{time_str}"
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    backup_folders.append((item_path, timestamp))
                except ValueError:
                    logger.warning(f"Could not parse timestamp from folder: {item}")
    
    # Sort by timestamp (newest first)
    backup_folders.sort(key=lambda x: x[1], reverse=True)
    return backup_folders

def find_recursive_backups(path):
    """Identify backup folders that contain other backup folders."""
    backup_pattern = re.compile(r'backup_\d{8}_\d{6}')
    recursive_backups = []
    
    if not os.path.exists(path):
        return recursive_backups
        
    for root, dirs, _ in os.walk(path):
        for dir_name in dirs:
            if backup_pattern.match(dir_name):
                recursive_backup_path = os.path.join(root, dir_name)
                if recursive_backup_path != path:  # Don't count the folder itself
                    recursive_backups.append(recursive_backup_path)
    
    return recursive_backups

def cleanup_backups(keep=3, src_dir='.', dry_run=False):
    """
    Clean up backup folders, keeping the specified number of recent backups.
    
    Args:
        keep (int): Number of recent backups to keep
        src_dir (str): Directory containing the backups
        dry_run (bool): If True, only simulate the cleanup without removing files
    """
    backup_folders = get_backup_folders(src_dir)
    
    if not backup_folders:
        logger.info("No backup folders found.")
        return
    
    logger.info(f"Found {len(backup_folders)} backup folders")
    
    # Keep the most recent 'keep' backups
    to_keep = backup_folders[:keep]
    to_remove = backup_folders[keep:]
    
    logger.info(f"Keeping {len(to_keep)} recent backups:")
    for folder, timestamp in to_keep:
        logger.info(f"  - {os.path.basename(folder)} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')})")
    
    if not to_remove:
        logger.info("No backups to remove.")
        return
    
    logger.info(f"Removing {len(to_remove)} older backups:")
    for folder, timestamp in to_remove:
        # Check for recursive backups
        recursive_backups = find_recursive_backups(folder)
        if recursive_backups:
            logger.warning(f"Found {len(recursive_backups)} recursive backups in {os.path.basename(folder)}")
        
        logger.info(f"  - {os.path.basename(folder)} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')})")
        
        if not dry_run:
            try:
                shutil.rmtree(folder)
                logger.info(f"    Removed: {os.path.basename(folder)}")
            except Exception as e:
                logger.error(f"    Failed to remove {folder}: {str(e)}")
        else:
            logger.info(f"    Would remove: {os.path.basename(folder)} (dry run)")

def main():
    parser = argparse.ArgumentParser(description='Clean up backup folders.')
    parser.add_argument('--keep', type=int, default=3,
                        help='Number of recent backups to keep (default: 3)')
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory containing backup folders (default: current directory)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate cleanup without removing files')
    
    args = parser.parse_args()
    
    logger.info(f"Starting backup cleanup (keep={args.keep}, dir={args.dir}, dry_run={args.dry_run})")
    cleanup_backups(args.keep, args.dir, args.dry_run)
    logger.info("Backup cleanup completed.")

if __name__ == "__main__":
    main() 