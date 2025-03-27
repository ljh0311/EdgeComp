#!/usr/bin/env python
"""
Backup Restoration Utility

This script restores the system from a specified backup folder.
"""

import os
import re
import shutil
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('restore_backup')

def validate_backup_folder(backup_folder):
    """
    Validate if the specified folder is a valid backup.
    
    Args:
        backup_folder (str): Path to the backup folder
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if folder exists
    if not os.path.exists(backup_folder):
        logger.error(f"Backup folder does not exist: {backup_folder}")
        return False
    
    # Check if it's a directory
    if not os.path.isdir(backup_folder):
        logger.error(f"Specified path is not a directory: {backup_folder}")
        return False
    
    # Check if it matches backup naming pattern
    backup_pattern = re.compile(r'backup_\d{8}_\d{6}')
    folder_name = os.path.basename(backup_folder)
    if not backup_pattern.match(folder_name):
        logger.error(f"Folder name does not match backup pattern: {folder_name}")
        return False
    
    return True

def create_current_backup(app_root, exclude_folders=None):
    """
    Create a backup of the current state before restoration.
    
    Args:
        app_root (str): Root directory of the application
        exclude_folders (list): List of folder names to exclude
        
    Returns:
        str: Path to the created backup folder
    """
    if exclude_folders is None:
        exclude_folders = []
    
    # Create backup folder name with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = os.path.join(app_root, f"backup_{timestamp}")
    
    logger.info(f"Creating backup of current state: {os.path.basename(backup_folder)}")
    
    try:
        # Create backup folder
        os.makedirs(backup_folder, exist_ok=True)
        
        # Copy files and folders (excluding backup folders)
        for item in os.listdir(app_root):
            item_path = os.path.join(app_root, item)
            
            # Skip backups and excluded folders
            if item.startswith("backup_") or item in exclude_folders:
                continue
                
            # Copy file or directory to backup
            if os.path.isdir(item_path):
                shutil.copytree(item_path, os.path.join(backup_folder, item))
            else:
                shutil.copy2(item_path, os.path.join(backup_folder, item))
                
        logger.info(f"Backup created successfully: {os.path.basename(backup_folder)}")
        return backup_folder
        
    except Exception as e:
        logger.error(f"Failed to create backup: {str(e)}")
        if os.path.exists(backup_folder):
            shutil.rmtree(backup_folder)
        return None

def restore_from_backup(backup_folder, app_root, create_backup=True, dry_run=False):
    """
    Restore the system from a backup folder.
    
    Args:
        backup_folder (str): Path to the backup folder
        app_root (str): Root directory of the application
        create_backup (bool): Whether to create a backup of current state
        dry_run (bool): If True, only simulate the restoration
        
    Returns:
        bool: True if restoration was successful, False otherwise
    """
    # Validate backup folder
    if not validate_backup_folder(backup_folder):
        return False
    
    # Create backup of current state if requested
    if create_backup and not dry_run:
        current_backup = create_current_backup(app_root, exclude_folders=["tools"])
        if not current_backup:
            logger.error("Failed to create backup of current state.")
            return False
    
    logger.info(f"Starting restoration from: {os.path.basename(backup_folder)}")
    
    try:
        # Get list of files to restore (excluding backup folders)
        to_restore = []
        for item in os.listdir(backup_folder):
            if not item.startswith("backup_"):
                to_restore.append(item)
        
        logger.info(f"Found {len(to_restore)} items to restore")
        
        # Go through each item in the backup and restore it
        for item in to_restore:
            source_path = os.path.join(backup_folder, item)
            target_path = os.path.join(app_root, item)
            
            # If target exists, remove it first
            if os.path.exists(target_path) and not dry_run:
                if os.path.isdir(target_path):
                    logger.info(f"Removing existing directory: {item}")
                    shutil.rmtree(target_path)
                else:
                    logger.info(f"Removing existing file: {item}")
                    os.remove(target_path)
            
            # Copy from backup to target
            if os.path.isdir(source_path):
                logger.info(f"Restoring directory: {item}")
                if not dry_run:
                    shutil.copytree(source_path, target_path)
            else:
                logger.info(f"Restoring file: {item}")
                if not dry_run:
                    shutil.copy2(source_path, target_path)
        
        if dry_run:
            logger.info("Dry run completed. No changes were made.")
        else:
            logger.info("Restoration completed successfully.")
            
        return True
        
    except Exception as e:
        logger.error(f"Restoration failed: {str(e)}")
        return False

def list_available_backups(app_root):
    """
    List all available backup folders.
    
    Args:
        app_root (str): Root directory of the application
        
    Returns:
        list: List of backup folder paths sorted by timestamp (newest first)
    """
    backup_pattern = re.compile(r'backup_(\d{8})_(\d{6})')
    backups = []
    
    for item in os.listdir(app_root):
        item_path = os.path.join(app_root, item)
        if os.path.isdir(item_path) and backup_pattern.match(item):
            match = backup_pattern.match(item)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                timestamp_str = f"{date_str}_{time_str}"
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    backups.append((item_path, timestamp))
                except ValueError:
                    continue
    
    # Sort by timestamp (newest first)
    backups.sort(key=lambda x: x[1], reverse=True)
    return backups

def main():
    parser = argparse.ArgumentParser(description='Restore system from a backup folder.')
    parser.add_argument('--backup', type=str, help='Name or path of the backup folder to restore from')
    parser.add_argument('--list', action='store_true', help='List available backups')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating a backup of current state')
    parser.add_argument('--dry-run', action='store_true', help='Simulate restoration without making changes')
    parser.add_argument('--app-root', type=str, default='.', help='Root directory of the application')
    
    args = parser.parse_args()
    app_root = os.path.abspath(args.app_root)
    
    # List available backups
    if args.list:
        backups = list_available_backups(app_root)
        if not backups:
            logger.info("No backup folders found.")
            return
            
        logger.info(f"Found {len(backups)} backup folders:")
        for i, (backup_path, timestamp) in enumerate(backups):
            folder_name = os.path.basename(backup_path)
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"{i+1}. {folder_name} ({formatted_time})")
        return
    
    # Check if backup folder is specified
    if not args.backup:
        logger.error("No backup folder specified. Use --backup to specify a folder.")
        return
    
    # Resolve backup folder path
    backup_folder = args.backup
    if not os.path.isabs(backup_folder):
        # Try to find the backup in the app root
        if os.path.exists(os.path.join(app_root, backup_folder)):
            backup_folder = os.path.join(app_root, backup_folder)
        # Try to find by partial name
        else:
            backups = list_available_backups(app_root)
            for folder_path, _ in backups:
                if os.path.basename(folder_path).startswith(backup_folder):
                    backup_folder = folder_path
                    break
    
    # Perform restoration
    success = restore_from_backup(
        backup_folder=backup_folder,
        app_root=app_root,
        create_backup=not args.no_backup,
        dry_run=args.dry_run
    )
    
    if success:
        logger.info("Restoration process completed.")
    else:
        logger.error("Restoration process failed.")

if __name__ == "__main__":
    main() 