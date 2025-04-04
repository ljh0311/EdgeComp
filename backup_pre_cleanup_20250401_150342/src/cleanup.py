#!/usr/bin/env python
"""
Enhanced Cleanup Script for Baby Monitor System
===============================================
This script helps organize the source directory structure by:
1. Removing duplicate or unnecessary files
2. Consolidating similar functionality
3. Creating directory backups before removing anything
4. Ensuring proper import paths
5. Organizing modules into a consistent structure
"""

import os
import shutil
import datetime
import logging
import argparse
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).parent.absolute()
BACKUP_DIR = BASE_DIR / f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Directories that might contain unnecessary duplicates or should be cleaned
UNNECESSARY_DIRS = [
    BASE_DIR / "src",              # Remove nested src directory
    BASE_DIR / "babymonitor.egg-info",  # Auto-generated, can be rebuilt
    BASE_DIR / "__pycache__",       # Python cache files
]

# Specific files to remove (redundant implementations)
FILES_TO_REMOVE = [
    BASE_DIR / "babymonitor" / "audio.py",        # Use dedicated audio modules instead
    BASE_DIR / "babymonitor" / "camera.py",       # Use camera_wrapper.py instead
    BASE_DIR / "services" / "camera" / "camera.py",  # Duplicate camera implementation
    BASE_DIR / "services" / "audio" / "audio_processor.py",  # Redundant with audio modules
]

# Files/patterns to clean
PATTERN_CLEANUP = [
    r".*\.pyc$",                   # Python bytecode
    r".*\.pyo$",                   # Optimized bytecode
    r".*\.pyd$",                   # Python DLL
    r".*~$",                       # Temp files
    r".*\.bak$",                   # Backup files
    r".*\.swp$",                   # vim swap files
    r".*\.swo$",                   # vim swap files
]

# Modules that should be consolidated
MODULE_CONSOLIDATION = [
    # Format: (source_path, target_path, filename)
    # Camera modules
    ("babymonitor/utils/camera.py", "babymonitor/camera", "util_camera.py"),
    ("babymonitor/camera_wrapper.py", "babymonitor/camera", "camera_wrapper.py"),
    ("services/camera/camera.py", "babymonitor/camera", "service_camera.py"),
    
    # Audio modules
    ("babymonitor/audio.py", "babymonitor/audio", "audio_main.py"),
    ("services/audio/audio_processor.py", "babymonitor/audio", "processor.py"),
    ("services/audio/audio_features.py", "babymonitor/audio", "features.py"),
    
    # Configuration
    ("babymonitor/config.py", "babymonitor/core", "config.py"),
    ("babymonitor/utils/config.py", "babymonitor/core", "utils_config.py"),
    
    # Models
    ("models/person_detector.py", "babymonitor/detectors", "external_person_detector.py"),
    ("models/emotion/models", "babymonitor/emotion/models", None),
    
    # System files
    ("babymonitor/system.py", "babymonitor/core", "system.py"),
    ("babymonitor/main.py", "babymonitor/core", "main.py"),
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean up and organize the project structure")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without making changes")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--force", action="store_true", help="Force cleanup without confirmation")
    return parser.parse_args()

def create_backup():
    """Create a backup of the directory before making changes."""
    logger.info(f"Creating backup in {BACKUP_DIR}")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # Back up the entire src directory
    shutil.copytree(BASE_DIR, BACKUP_DIR / BASE_DIR.name, dirs_exist_ok=True)
    
    logger.info("Backup completed")

def remove_unnecessary_dirs(dry_run=False):
    """Remove unnecessary directories."""
    for dir_path in UNNECESSARY_DIRS:
        if dir_path.exists():
            logger.info(f"{'Would remove' if dry_run else 'Removing'} {dir_path}")
            if not dry_run:
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Successfully removed {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to remove {dir_path}: {str(e)}")

def remove_specific_files(dry_run=False):
    """Remove specific redundant files."""
    for file_path in FILES_TO_REMOVE:
        if file_path.exists():
            logger.info(f"{'Would remove' if dry_run else 'Removing'} {file_path}")
            if not dry_run:
                try:
                    os.remove(file_path)
                    logger.info(f"Successfully removed {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {str(e)}")

def clean_pattern_files(dry_run=False):
    """Clean files matching patterns."""
    for pattern in PATTERN_CLEANUP:
        regex = re.compile(pattern)
        for root, _, files in os.walk(BASE_DIR):
            for file in files:
                if regex.match(file):
                    file_path = Path(root) / file
                    logger.info(f"{'Would delete' if dry_run else 'Deleting'} {file_path}")
                    if not dry_run:
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.error(f"Failed to delete {file_path}: {str(e)}")

def consolidate_modules(dry_run=False):
    """Consolidate similar modules."""
    for source_rel, target_rel, filename in MODULE_CONSOLIDATION:
        source_path = BASE_DIR / source_rel
        target_dir = BASE_DIR / target_rel
        
        # Skip if source doesn't exist
        if not source_path.exists():
            logger.info(f"Source {source_path} does not exist, skipping")
            continue
        
        # Create target directory if it doesn't exist
        if not target_dir.exists():
            logger.info(f"{'Would create' if dry_run else 'Creating'} directory {target_dir}")
            if not dry_run:
                os.makedirs(target_dir, exist_ok=True)
                
                # Create __init__.py if it doesn't exist
                init_file = target_dir / "__init__.py"
                if not init_file.exists():
                    with open(init_file, 'w') as f:
                        f.write('"""Auto-generated by cleanup.py"""\n')
        
        # If source is a directory and filename is None, copy entire directory
        if source_path.is_dir() and filename is None:
            target_path = target_dir
            logger.info(f"{'Would copy' if dry_run else 'Copying'} directory {source_path} to {target_path}")
            if not dry_run:
                try:
                    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                    logger.info(f"Successfully copied {source_path} to {target_path}")
                except Exception as e:
                    logger.error(f"Failed to copy {source_path} to {target_path}: {str(e)}")
        # Otherwise, copy the file
        else:
            target_path = target_dir / (filename or source_path.name)
            logger.info(f"{'Would copy' if dry_run else 'Copying'} {source_path} to {target_path}")
            if not dry_run:
                try:
                    shutil.copy2(source_path, target_path)
                    logger.info(f"Successfully copied {source_path} to {target_path}")
                except Exception as e:
                    logger.error(f"Failed to copy {source_path} to {target_path}: {str(e)}")

def create_or_update_readme(dry_run=False):
    """Create or update README.md with project structure information."""
    readme_path = BASE_DIR.parent / "README.md"
    
    # Generate project structure
    structure_text = "## Project Structure\n\n```\n"
    structure_text += f"EdgeComp/\n"
    structure_text += f"├── src/                        # Main source code\n"
    structure_text += f"│   ├── babymonitor/            # Main package\n"
    structure_text += f"│   │   ├── camera/             # Camera-related modules\n"
    structure_text += f"│   │   ├── audio/              # Audio processing modules\n"
    structure_text += f"│   │   ├── detectors/          # Object detection modules\n"
    structure_text += f"│   │   ├── emotion/            # Emotion detection models\n"
    structure_text += f"│   │   ├── core/               # Core system components\n"
    structure_text += f"│   │   ├── web/                # Web interface\n"
    structure_text += f"│   │   └── utils/              # Utility functions\n"
    structure_text += f"│   ├── run_server.py           # Script to run the web server\n"
    structure_text += f"│   ├── test_server.py          # Script to test API endpoints\n"
    structure_text += f"│   └── cleanup.py              # This cleanup script\n"
    structure_text += f"├── requirements.txt            # Project dependencies\n"
    structure_text += f"└── README.md                   # This file\n"
    structure_text += "```\n\n"
    
    if readme_path.exists():
        # Update existing README
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Check if structure section exists
        structure_pattern = re.compile(r"## Project Structure.*?```.*?```", re.DOTALL)
        if structure_pattern.search(content):
            # Replace existing structure section
            content = structure_pattern.sub(structure_text, content)
        else:
            # Add structure section at the end
            content += "\n\n" + structure_text
        
        logger.info(f"{'Would update' if dry_run else 'Updating'} {readme_path}")
        if not dry_run:
            with open(readme_path, 'w') as f:
                f.write(content)
    else:
        # Create new README
        readme_content = """# Baby Monitor System

A comprehensive baby monitoring system with emotion detection, camera management, and web interface.

"""
        readme_content += structure_text
        
        logger.info(f"{'Would create' if dry_run else 'Creating'} {readme_path}")
        if not dry_run:
            with open(readme_path, 'w') as f:
                f.write(readme_content)

def create_or_update_requirements(dry_run=False):
    """Create or update requirements.txt."""
    req_path = BASE_DIR.parent / "requirements.txt"
    
    requirements = """# Core dependencies
flask==2.0.1
flask-socketio==5.1.1
opencv-python==4.5.3.56
numpy==1.21.2
scipy==1.7.1
torch==1.9.0
sounddevice==0.4.2
requests==2.26.0
psutil==5.9.0

# Development dependencies
pytest==6.2.5
flake8==3.9.2
black==21.8b0
"""
    
    logger.info(f"{'Would create/update' if dry_run else 'Creating/updating'} {req_path}")
    if not dry_run:
        with open(req_path, 'w') as f:
            f.write(requirements)

def main():
    """Main function to run the cleanup script."""
    args = parse_args()
    
    logger.info("Starting cleanup script")
    
    # Confirm cleanup if not forced
    if not args.force and not args.dry_run:
        confirm = input("This will reorganize your project files. Continue? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Cleanup cancelled by user")
            return
    
    # Create backup unless skipped
    if not args.no_backup and not args.dry_run:
        create_backup()
    
    # Perform cleanup operations
    remove_unnecessary_dirs(args.dry_run)
    remove_specific_files(args.dry_run)
    clean_pattern_files(args.dry_run)
    consolidate_modules(args.dry_run)
    create_or_update_readme(args.dry_run)
    create_or_update_requirements(args.dry_run)
    
    if args.dry_run:
        logger.info("Dry run completed. No changes were made.")
    else:
        logger.info("Cleanup completed successfully")
        if not args.no_backup:
            logger.info(f"Backup is available at: {BACKUP_DIR}")

if __name__ == "__main__":
    main() 