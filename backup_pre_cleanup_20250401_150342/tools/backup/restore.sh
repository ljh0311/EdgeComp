#!/bin/bash

function show_menu {
    clear
    echo "======================================"
    echo "   Baby Monitor Backup and Restore"
    echo "======================================"
    echo
    echo "Please select an option:"
    echo
    echo "[1] List backup folders"
    echo "[2] Create a new backup"
    echo "[3] Restore from backup"
    echo "[4] Clean up old backups"
    echo "[5] Return to main menu"
    echo
}

while true; do
    show_menu
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            echo
            echo "Available backups:"
            echo "-----------------"
            python tools/backup/list_backups.py
            echo
            read -p "Press Enter to continue..."
            ;;
        2)
            echo
            echo "Creating new backup..."
            python tools/backup/create_backup.py
            echo
            read -p "Press Enter to continue..."
            ;;
        3)
            echo
            echo "Available backups:"
            echo "-----------------"
            python tools/backup/list_backups.py
            echo
            read -p "Enter backup ID to restore (or press Enter to cancel): " backup_id
            if [ -z "$backup_id" ]; then
                continue
            fi
            
            echo
            echo "Warning: Restoring will overwrite current files."
            read -p "Are you sure you want to restore from backup $backup_id? (y/n): " confirm
            if [[ $confirm == [Yy]* ]]; then
                python tools/backup/restore_backup.py --backup "$backup_id"
            else
                echo "Restore cancelled."
            fi
            echo
            read -p "Press Enter to continue..."
            ;;
        4)
            echo
            read -p "How many recent backups would you like to keep? (default: 3): " keep
            keep=${keep:-3}
            
            echo
            echo "Warning: This will permanently delete old backups."
            read -p "Are you sure you want to clean up old backups, keeping the $keep most recent? (y/n): " confirm
            if [[ $confirm == [Yy]* ]]; then
                python tools/backup/cleanup_backups.py --keep "$keep"
            else
                echo "Cleanup cancelled."
            fi
            echo
            read -p "Press Enter to continue..."
            ;;
        5)
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            sleep 2
            ;;
    esac
done 