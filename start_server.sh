#!/bin/bash

# Startup script for Flask server (uses port 5001 to avoid macOS AirPlay conflict)

echo "🚀 Starting Trading Assistant Flask Server..."
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found!"
    echo "Please create it first: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if litellm is installed
if ! python -c "import litellm" 2>/dev/null; then
    echo "⚠️  Warning: litellm not installed. Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if port 5001 is already in use
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 5001 is already in use!"
    echo "Kill the process with: kill \$(lsof -ti:5001)"
    exit 1
fi

# Note about port 5000
echo ""
echo "ℹ️  Note: Using port 5001 instead of 5000"
echo "   (macOS uses port 5000 for AirPlay Receiver)"
echo ""

# Start Flask server
echo "✅ Starting Flask server on http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

export FLASK_RUN_PORT=5001
export FLASK_ENV=development
python web_app.py



