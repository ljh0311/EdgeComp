#!/bin/bash

echo "======================================"
echo "    Baby Monitor Final Cleanup"
echo "======================================"
echo
echo "This script will remove unnecessary files from your"
echo "root directory to make it clean and organized."
echo
echo "Files to be removed:"
echo "- cleanup_and_organize.bat (no longer needed)"
echo "- backup_before_cleanup.bat (functionality moved to tools/backup)"
echo "- README_update.md (already merged into README.md)"
echo "- Any remaining old scripts that have been replaced"
echo
echo "WARNING: Make sure you've already run the cleanup_and_organize script"
echo "and that you have a backup before proceeding."
echo

read -p "Are you sure you want to proceed? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo
echo "Removing unnecessary files..."

# Define files to keep in root directory
KEEP_FILES=("run_babymonitor.bat" "run_babymonitor.sh" "README.md" "requirements.txt" "LICENSE")

# Define directories to keep
KEEP_DIRS=("src" "tools" "tests" "venv")

# Remove specific files that are no longer needed
files_to_remove=(
    "cleanup_and_organize.bat"
    "backup_before_cleanup.bat"
    "README_update.md"
    "README_scripts.md"
    "final_cleanup.bat"  # Windows cleanup script
)

for file in "${files_to_remove[@]}"; do
    if [ -f "$file" ]; then
        echo "- Removing $file"
        rm "$file"
    fi
done

# Look for any remaining .bat files that might be obsolete
for file in *.bat; do
    if [ -f "$file" ]; then
        keep=false
        for keep_file in "${KEEP_FILES[@]}"; do
            if [ "$file" = "$keep_file" ]; then
                keep=true
                break
            fi
        done
        
        if [ "$keep" = false ]; then
            echo "- Removing $file"
            rm "$file"
        fi
    fi
done

# Look for any remaining .sh files that might be obsolete
for file in *.sh; do
    if [ -f "$file" ]; then
        keep=false
        for keep_file in "${KEEP_FILES[@]}"; do
            if [ "$file" = "$keep_file" ]; then
                keep=true
                break
            fi
        done
        
        if [ "$keep" = false ] && [ "$file" != "final_cleanup.sh" ]; then
            echo "- Removing $file"
            rm "$file"
        fi
    fi
done

# Look for any remaining .py files in root (should be moved to proper directories)
for file in *.py; do
    if [ -f "$file" ]; then
        echo "- Removing $file (Python scripts should be in appropriate directories)"
        rm "$file"
    fi
done

# Remove any empty directories in root that aren't in the keep list
for dir in */; do
    dir=${dir%/}  # Remove trailing slash
    keep=false
    for keep_dir in "${KEEP_DIRS[@]}"; do
        if [ "$dir" = "$keep_dir" ]; then
            keep=true
            break
        fi
    done
    
    if [ "$keep" = false ]; then
        # Check if directory is empty
        if [ -z "$(ls -A "$dir")" ]; then
            echo "- Removing empty directory: $dir"
            rmdir "$dir"
        else
            echo "- Directory not empty, skipping: $dir"
        fi
    fi
done

echo
echo "======================================"
echo "        Final Cleanup Complete!"
echo "======================================"
echo
echo "Your directory structure is now clean and organized."
echo
echo "Essential files in root directory:"
echo "- run_babymonitor.bat (Windows launcher)"
echo "- run_babymonitor.sh (Linux/macOS launcher)"
echo "- README.md (Documentation)"
echo "- requirements.txt (Dependencies)"
echo
echo "Essential directories:"
echo "- src/ (Source code)"
echo "- tools/ (Utility scripts)"
echo "- tests/ (Test scripts)"
echo
echo "You can safely delete this cleanup script (final_cleanup.sh)"
echo "when you're satisfied with the results."
echo

read -p "Press Enter to exit..." 