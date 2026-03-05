#!/bin/bash
# Self Soul AGI System - Log Cleanup Script (Linux/Mac)
# This script deletes all log files from the system

echo "============================================"
echo "Self Soul AGI System - Log Cleanup Script"
echo "============================================"
echo ""
echo "This script will delete all log files from the system."
echo "WARNING: This action cannot be undone!"
echo ""

read -p "Are you sure you want to delete all log files? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo ""
echo "Deleting log files..."

# Delete logs from main logs directory
if [ -d "logs" ]; then
    echo "Deleting files from logs directory..."
    rm -f logs/*.log
    echo "Deleted: logs/*.log"
    
    rm -f logs/*.log.*
    echo "Deleted: logs/*.log.*"
else
    echo "logs directory not found."
fi

# Delete logs from core/logs directory
if [ -d "core/logs" ]; then
    echo "Deleting files from core/logs directory..."
    rm -f core/logs/*.log
    echo "Deleted: core/logs/*.log"
    
    rm -f core/logs/*.log.*
    echo "Deleted: core/logs/*.log.*"
else
    echo "core/logs directory not found."
fi

# Delete any other log files in project root
echo "Deleting log files in root directory..."
rm -f *.log
echo "Deleted: *.log"

echo ""
echo "============================================"
echo "Log cleanup completed!"
echo ""
echo "To clear conversation history in the browser:"
echo "1. Open http://localhost:5175/#/ in Chrome/Firefox"
echo "2. Press F12 to open Developer Tools"
echo "3. Go to Application tab (Chrome) or Storage tab (Firefox)"
echo "4. Find Local Storage and delete:"
echo "   - 'self_soul_conversation_history'"
echo "   - 'chat_messages'"
echo "   - Any keys starting with 'self_soul_'"
echo "5. Refresh the page"
echo ""
echo "Alternative: Open clear_conversation.html in your browser"
echo "or run in JavaScript console:"
echo "localStorage.removeItem('self_soul_conversation_history');"
echo "localStorage.removeItem('chat_messages');"
echo "Object.keys(localStorage).forEach(key => {"
echo "  if (key.startsWith('self_soul_')) {"
echo "    localStorage.removeItem(key);"
echo "  }"
echo "});"
echo "location.reload();"
echo "============================================"