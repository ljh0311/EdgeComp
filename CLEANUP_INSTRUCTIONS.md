# Baby Monitor System - Final Cleanup Instructions

These instructions will help you complete the final cleanup of your Baby Monitor System files to ensure a clean, organized directory structure.

## Step 1: Run the Main Reorganization Script

First, make sure you've already run the main reorganization script:

```
cleanup_and_organize.bat
```

This script:
- Creates necessary directories
- Moves files to their proper locations
- Removes obsolete files
- Updates documentation
- Consolidates startup scripts

## Step 2: Verify System Functionality

Before proceeding with final cleanup, verify that your Baby Monitor System still works correctly:

1. Run the main launcher:
   ```
   run_babymonitor.bat    # Windows
   # OR
   ./run_babymonitor.sh   # Linux/macOS
   ```

2. Test the backup/restore functionality:
   ```
   tools\backup_restore.bat    # Windows
   # OR
   bash tools/backup/restore.sh  # Linux/macOS
   ```

3. Make sure you can access all necessary features through the launcher menu

## Step 3: Run the Final Cleanup Script

Once you've verified that everything works correctly, run the final cleanup script:

```
final_cleanup.bat    # Windows
# OR
bash final_cleanup.sh  # Linux/macOS
```

This script will:
- Remove all unnecessary files from the root directory
- Leave only essential files and directories
- Clean up any empty directories

## After Cleanup

After the cleanup process completes, your directory structure should look like this:

```
baby_monitor_system/
├── README.md               # Main documentation
├── requirements.txt        # Python dependencies
├── run_babymonitor.bat     # Windows launcher
├── run_babymonitor.sh      # Linux/macOS launcher
├── src/                    # Source code
│   └── babymonitor/        # Main application
│       ├── models/         # Pretrained models
│       └── ...
├── tools/                  # Utility scripts
│   ├── backup/             # Backup utilities
│   │   ├── cleanup_backups.py
│   │   ├── create_backup.py
│   │   ├── list_backups.py
│   │   ├── restore.sh
│   │   └── restore_backup.py
│   ├── backup_restore.bat  # Windows backup utility
│   └── system/             # System utilities
│       └── optimize_raspberry_pi.sh
└── tests/                  # Test scripts
    └── updated/            # Updated test scripts
        └── api_test.py     # API testing utility
```

You can now safely delete the `final_cleanup` script and this instruction file. 