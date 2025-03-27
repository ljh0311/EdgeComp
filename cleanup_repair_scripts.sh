#!/bin/bash

echo "Cleaning up unnecessary repair and fix scripts..."

# Check if files exist before deleting them
if [ -f "repair_api_endpoints.py" ]; then
    echo "Removing repair_api_endpoints.py"
    rm "repair_api_endpoints.py"
fi

if [ -f "api_fix.js" ]; then
    echo "Removing api_fix.js"
    rm "api_fix.js"
fi

if [ -f "fix_api_urls.js" ]; then
    echo "Removing fix_api_urls.js"
    rm "fix_api_urls.js"
fi

if [ -f "tools/repair_standalone.py" ]; then
    echo "Removing repair_standalone.py"
    rm "tools/repair_standalone.py"
fi

echo ""
echo "Keeping essential files:"
echo "- app.py (with integrated microphone and emotion detection)"
echo "- api_routes.py (with proper microphone selection)"
echo ""

echo "Adding documentation..."
echo ""
echo "# Baby Monitor Repair Tools" > README_repair.md
echo "" >> README_repair.md
echo "The repair functionality has been integrated into the main application." >> README_repair.md
echo "" >> README_repair.md
echo "## How to use the repair tools" >> README_repair.md
echo "" >> README_repair.md
echo "1. Start the main application:" >> README_repair.md
echo "   - Windows: \`start.bat\`" >> README_repair.md
echo "   - Linux/macOS: \`./start.sh\`" >> README_repair.md
echo "2. Access the repair tools at: http://localhost:5000/repair_tools" >> README_repair.md
echo "" >> README_repair.md
echo "## Features" >> README_repair.md
echo "" >> README_repair.md
echo "- Microphone selection and testing" >> README_repair.md
echo "- Audio system testing" >> README_repair.md
echo "- Emotion detection model selection" >> README_repair.md
echo "- System status monitoring" >> README_repair.md
echo "" >> README_repair.md
echo "All functionality now runs on port 5000 with the main app." >> README_repair.md
echo ""

echo "Done! Cleanup complete."
echo "The repair functionality is now integrated into the main application."
echo "Access it at http://localhost:5000/repair_tools"

# Make the script executable
chmod +x start.sh 2>/dev/null
chmod +x start_pi.sh 2>/dev/null 