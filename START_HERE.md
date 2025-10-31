# 🚀 START HERE - Quick Setup Guide

## Issue: macOS Port Conflict

macOS uses **port 5000** for AirPlay Receiver, so this app now runs on **port 5001**.

---

## ⚡ Quick Start (3 Steps)

### Step 1: Start the Server

**Option A: Use the startup script (Easiest)**
```bash
./start_server.sh
```

**Option B: Manual start**
```bash
source venv/bin/activate
export FLASK_RUN_PORT=5001
python web_app.py
```

The server will start on **http://localhost:5001**

### Step 2: Test the API

In a **new terminal** window:

**Option A: Run automated tests**
```bash
source venv/bin/activate
python test_upload_api.py
```

**Option B: Use the web interface**
```bash
open upload_example.html
```

**Option C: Test with cURL**
```bash
# List uploads
curl http://localhost:5001/api/list-uploads

# Upload an image
curl -X POST http://localhost:5001/api/upload-image \
  -F "image=@BTCUSDT.P_2025-09-02_22-55-40_545b4.png"
```

### Step 3: Check It's Working

Visit **http://localhost:5001** in your browser

---

## 📁 Files Updated for Port 5001

- ✅ `test_upload_api.py` - Test script
- ✅ `upload_example.html` - Web demo  
- ✅ `start_server.sh` - Startup script (NEW)

---

## 🛠️ Troubleshooting

### "Port 5001 is already in use"
```bash
# Kill the existing process
kill $(lsof -ti:5001)
```

### "Module not found" errors
```bash
# Install dependencies
source venv/bin/activate
pip install -r requirements.txt
```

### Want to use port 5000?
Disable AirPlay Receiver in **System Preferences > Sharing**

Or use a different port:
```bash
export FLASK_RUN_PORT=8000
python web_app.py
```

Then update `BASE_URL` in test files.

---

## 📚 Full Documentation

- **Quick Start**: `QUICK_START.md`
- **API Reference**: `API_DOCUMENTATION.md`
- **Feature Overview**: `FEATURE_SUMMARY.md`

---

## ✅ You're Ready!

1. Run `./start_server.sh`
2. Open `upload_example.html` in your browser
3. Upload a trading chart image
4. Done! 🎉

**Server URL**: http://localhost:5001

