# Telegram Bot Debugging Guide

If `/status` shows no analysis running when you've uploaded an image, follow these steps:

## 🔍 Quick Diagnostic Steps

### Step 1: Verify Bot is Running

Run the test script:
```bash
python test_bot_status.py
```

This will:
- ✅ Check your Telegram credentials
- ✅ Test bot communication
- ✅ Show current analysis state
- ✅ Send a test message to Telegram

### Step 2: Check Console Output

When you start `web_app.py` or `app.py`, you should see:
```
✅ Telegram bot started for command control
```

If you see:
```
⚠️  Could not start Telegram bot: [error]
```

Then there's a configuration issue.

### Step 3: Verify Auto-Analyze Parameter

When uploading via API, make sure you're passing `auto_analyze=true`:

**Correct:**
```bash
curl -X POST http://localhost:5001/api/upload-image \
  -F "image=@chart.png" \
  -F "auto_analyze=true"
```

**Incorrect:**
```bash
curl -X POST http://localhost:5001/api/upload-image \
  -F "image=@chart.png"
# Missing auto_analyze=true - analysis won't start!
```

### Step 4: Check Timing

The analysis state is set when `run_trading_analysis()` starts, which happens in a background thread. There might be a 1-2 second delay.

Try:
1. Upload image
2. Wait 2-3 seconds
3. Send `/status` in Telegram

## 🐛 Common Issues

### Issue 1: "No analysis running" immediately after upload

**Cause**: You checked too quickly or `auto_analyze` wasn't set to `true`

**Solution**:
```bash
# Wait a moment after upload
curl -X POST http://localhost:5001/api/upload-image \
  -F "image=@chart.png" \
  -F "auto_analyze=true"

# Wait 2-3 seconds
sleep 3

# Now check in Telegram
# Send: /status
```

### Issue 2: Bot not responding to commands

**Cause**: Bot token or chat ID incorrect

**Solution**:
1. Check `.env` file:
   ```bash
   cat .env | grep TELEGRAM
   ```

2. Verify with @userinfobot in Telegram to get your correct chat ID

3. Restart the app after fixing `.env`

### Issue 3: Analysis completes too quickly

**Cause**: The analysis might finish before you check status (especially for fast LLM responses)

**Solution**: 
- Check console output - you'll see the full analysis log
- Or send `/status` immediately after upload
- The bot sends notifications at key points, so you'll get updates even if you miss checking status

### Issue 4: Bot started but not polling

**Cause**: The bot instance exists but polling thread didn't start

**Check**: Run `python test_bot_status.py` - it shows polling status

**Solution**: Restart your app

## 📊 Understanding Analysis Flow

```
Upload Image (auto_analyze=true)
    ↓
Background Thread Starts
    ↓
[~1-2s delay] set_analysis_running(True) called
    ↓
Bot sends: "🚀 Analysis Started"
    ↓
/status now shows: "RUNNING"
    ↓
Analysis continues (may take minutes)
    ↓
Bot sends periodic updates every 5 cycles
    ↓
set_analysis_running(False) called
    ↓
Bot sends: "✅ Analysis Complete"
    ↓
/status now shows: "IDLE"
```

## 🧪 Test Sequence

Follow this exact sequence to test:

```bash
# Terminal 1: Start the server
python web_app.py

# You should see:
# ✅ Telegram bot started for command control
# * Running on http://0.0.0.0:5001

# Terminal 2: Upload a chart
curl -X POST http://localhost:5001/api/upload-image \
  -F "image=@your_chart.png" \
  -F "auto_analyze=true"

# Response should include:
# "job_id": "analysis_XXXXXXXXXX"
# "message": "Image uploaded and analysis started..."

# Immediately in Telegram: Send
# /status

# Expected Response:
# 📊 Analysis Status: RUNNING
# Symbol: [Symbol from chart]
# Direction: [Long/Short]
# Start Time: [timestamp]
# Elapsed Time: [seconds]
```

## 🔧 Manual State Check

If you want to manually check the state in Python:

```python
from telegram_bot import get_analysis_state

state = get_analysis_state()
status = state.get_status_info()
print(status)

# Output when running:
# {'running': True, 'symbol': 'BTCUSDT', 'direction': 'long', ...}

# Output when idle:
# {'running': False}
```

## 📱 Expected Telegram Messages

During a normal analysis, you should receive:

1. **On Upload** (if image sent to Telegram):
   ```
   📊 New Trading Chart Uploaded
   Filename: chart.png
   ...
   ```

2. **On Analysis Start** (~1-2s after upload):
   ```
   🚀 Analysis Started
   Preparing market data...
   ```

3. **After Initial Analysis**:
   ```
   📊 Chart Analyzed
   Symbol: BTCUSDT
   Direction: Long
   Timeframe: 1h
   Fetching market data...
   ```

4. **During Polling** (every 5 cycles):
   ```
   ⏳ Still Polling...
   Cycle 5
   Elapsed: 5m
   Status: pending
   ```

5. **On Completion**:
   ```
   ✅ Analysis Complete
   The analysis has finished.
   ```

## 🎯 If Nothing Works

1. **Stop everything**:
   ```bash
   # Kill any running instances
   pkill -f web_app.py
   pkill -f app.py
   ```

2. **Verify environment**:
   ```bash
   python test_bot_status.py
   ```

3. **Start fresh**:
   ```bash
   python web_app.py
   ```

4. **Watch console** for:
   - "✅ Telegram bot started for command control"
   - Any error messages

5. **Upload with verbose logging**:
   ```bash
   curl -v -X POST http://localhost:5001/api/upload-image \
     -F "image=@chart.png" \
     -F "auto_analyze=true"
   ```

6. **Check the response** - should have `job_id` field

7. **In Telegram**, wait 3 seconds then send: `/status`

## 💡 Pro Tips

1. **Use logging**: Check console output - it shows everything
2. **Telegram keeps history**: You'll get all notifications even if you miss them live
3. **Multiple uploads**: Bot prevents concurrent analyses - finish one before starting another
4. **Stop button**: Use `/stop` in Telegram if you need to cancel and retry

## 🆘 Still Not Working?

Share this info:
1. Output of `python test_bot_status.py`
2. Console output when starting web_app.py
3. Response from the curl upload command
4. What `/status` shows in Telegram
5. Any error messages in console

This will help identify the exact issue!

