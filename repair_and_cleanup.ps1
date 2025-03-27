# Baby Monitor Repair and Cleanup Utility
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Baby Monitor Repair and Cleanup Utility" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Check if python is installed and in PATH
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "Python detected: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "Python not found! Please install Python and add it to your PATH." -ForegroundColor Red
    exit
}

Write-Host "Checking for required Python packages..." -ForegroundColor Yellow
python -m pip install flask pyaudio psutil | Out-Null

function Show-Menu {
    Write-Host ""
    Write-Host "Select an operation:" -ForegroundColor Yellow
    Write-Host "1 - Run API Repair Server"
    Write-Host "2 - Clean up backup folders"
    Write-Host "3 - List backup folders"
    Write-Host "4 - Restore from backup"
    Write-Host "5 - Exit"
    Write-Host ""
}

function Run-Server {
    Write-Host ""
    Write-Host "Starting API Repair Server..." -ForegroundColor Green
    Write-Host ""
    Write-Host "The server will run on port 5001." -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
    Write-Host ""
    python repair_api_endpoints.py --debug
}

function Cleanup-Backups {
    Write-Host ""
    $keep = Read-Host "Number of backups to keep (default: 3)"
    
    if ([string]::IsNullOrEmpty($keep)) {
        $keep = 3
    }
    
    Write-Host ""
    Write-Host "Cleaning up backup folders, keeping the $keep most recent backups..." -ForegroundColor Yellow
    Write-Host ""
    python tools\cleanup_backups.py --keep $keep --dir .
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function List-Backups {
    Write-Host ""
    Write-Host "Listing available backup folders..." -ForegroundColor Yellow
    Write-Host ""
    python tools\restore_backup.py --list --app-root .
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Restore-FromBackup {
    Write-Host ""
    python tools\restore_backup.py --list --app-root .
    Write-Host ""
    $backup = Read-Host "Enter the name of the backup folder to restore from"
    Write-Host ""
    $confirm = Read-Host "Are you sure you want to restore from $backup? (y/n)"
    
    if ($confirm -ne "y") {
        return
    }
    
    Write-Host ""
    Write-Host "Restoring from backup $backup..." -ForegroundColor Yellow
    python tools\restore_backup.py --backup $backup --app-root .
    Write-Host ""
    Read-Host "Press Enter to continue"
}

$running = $true
while ($running) {
    Show-Menu
    $choice = Read-Host "Enter your choice (1-5)"
    
    switch ($choice) {
        "1" { Run-Server; $running = $false }
        "2" { Cleanup-Backups }
        "3" { List-Backups }
        "4" { Restore-FromBackup }
        "5" { $running = $false }
        default { Write-Host "Invalid choice. Please try again." -ForegroundColor Red }
    }
}

Write-Host "Goodbye!" -ForegroundColor Green 