# Troubleshooting: Not Receiving Analysis Notifications

## Problem
You uploaded an image and received the initial Telegram notification with the image, but no analysis result notification.

## Why This Happens

### 1. **Analysis Takes Time (Most Common)**
The analysis can take **5-30+ minutes** depending on:
- **Timeframe**: A 1-hour chart means polling happens every hour
- **Signal validation**: System polls market data until signal is valid or invalidated
- **LLM processing**: AI analysis takes time

**What's happening:**
```
Upload Image (1 sec)
  ↓
Send Image to Telegram ✅ (you see this)
  ↓
LLM Analysis (30-60 sec)
  ↓
Market Data Fetch (10 sec)
  ↓
Signal Validation Polling (CAN TAKE MANY MINUTES!)
  ↓ (waits 1 timeframe interval between checks)
  ↓
Trade Gate Decision (30 sec)
  ↓
Send Result to Telegram ✅ (you should see this)
```

### 2. **Server Not Running**
If the Flask server isn't running, analysis won't complete.

**Check if server is running:**
```bash
ps aux | grep web_app
```

**If not running, start it:**
```bash
cd /Users/hpk/trassist2
source venv/bin/activate
python web_app.py
```

### 3. **Missing Environment Variables**
Required in `.env` file:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
MEXC_API_KEY=your_api_key
MEXC_API_SECRET=your_api_secret  # THIS IS REQUIRED!
OPENAI_API_KEY=your_openai_key
LITELLM_VISION_MODEL=gpt-4o
LITELLM_TEXT_MODEL=deepseek/deepseek-chat
```

### 4. **Analysis Error (Silent Failure)**
If analysis fails, error notification should be sent, but might not work if Telegram itself fails.

## Solutions

### Quick Fix: Check Environment
```bash
cd /Users/hpk/trassist2
source venv/bin/activate
python diagnose_analysis.py
```

This will:
- ✅ Test Telegram connection
- ✅ Test MEXC API
- ✅ Test LLM connection
- ✅ Show recent uploads and analysis

### Solution 1: Wait Longer (Most Likely)
**Be patient!** Analysis can take a long time, especially for longer timeframes.

For a **15-minute chart**:
- Each polling cycle waits 15 minutes
- May need 2-3 cycles
- Total time: **30-45 minutes**

For a **1-minute chart**:
- Each polling cycle waits 1 minute
- May need 5-10 cycles
- Total time: **5-10 minutes**

### Solution 2: Check Server Logs
If server is running, check logs:

```bash
# If running in terminal, check the output

# If running with nohup
tail -f nohup.out

# If running with systemd
sudo journalctl -u trading-assistant -f
```

Look for:
- "Step 10: Starting signal validation polling..."
- "Polling cycle X: signal status = pending/valid/invalidated"
- Error messages

### Solution 3: Fix Missing MEXC_API_SECRET

Add to your `.env` file:
```bash
MEXC_API_SECRET=your_actual_secret_here
```

Without this, market data fetching will fail and analysis can't complete.

### Solution 4: Manual Analysis Test

Test with an existing uploaded image:

```bash
cd /Users/hpk/trassist2
source venv/bin/activate

# Test with one of your uploaded images
python test_manual_analysis.py uploads/trading_chart_20251031_114223.png
```

### Solution 5: Reduce Polling Time for Testing

For faster testing, use a 1-minute chart. The analysis will complete in 5-10 minutes instead of hours.

## How to Know If It's Working

### Immediate (Within seconds):
- ✅ Image upload notification in Telegram

### During Analysis (Minutes/Hours):
Check server logs for progress:
- "Step 1: Preparing market data..."
- "Step 2: Analyzing trading chart..."
- "Step 10: Starting signal validation polling..."
- "Polling cycle 1, 2, 3..." (this is the slow part)

### After Analysis (When complete):
One of these notifications in Telegram:
- ✅ "🚀 VALID TRADE SIGNAL" (trade approved)
- ✅ "❌ TRADE INVALIDATED" (signal invalidated)
- ✅ "❌ Analysis Error" (something went wrong)

## Common Issues

### "I waited 5 minutes, nothing happened"
**Wait longer!** For anything other than a 1-minute chart, 5 minutes is not enough.

### "Server was running but stopped"
Analysis runs in a background thread. Even if server stops after upload, the thread should continue (but won't with process kill).

Run server in background properly:
```bash
nohup python web_app.py > server.log 2>&1 &
```

### "I got the image but never got a response"
Check these in order:
1. Is server still running? `ps aux | grep web_app`
2. Check server logs for errors
3. Run `python diagnose_analysis.py` to test connections
4. Ensure MEXC_API_SECRET is set
5. Wait longer (seriously, polling takes time!)

## Testing End-to-End

1. **Ensure server is running:**
```bash
cd /Users/hpk/trassist2
source venv/bin/activate
python web_app.py
```

2. **Upload a test image (in another terminal):**
```bash
cd /Users/hpk/trassist2
./test_api_upload.sh uploads/test_chart_custom_name.png
```

3. **Check Telegram:**
- Should receive image immediately
- Wait 10-30 minutes (yes, really!)
- Should receive analysis result

4. **Monitor server logs:**
```bash
# In the terminal where server is running, watch for:
# - Step progress updates
# - Polling cycles
# - Final notification status
```

## Expected Timeline

| Timeframe | Analysis Duration |
|-----------|-------------------|
| 1m | 5-15 minutes |
| 5m | 15-30 minutes |
| 15m | 30-60 minutes |
| 1h | 1-3 hours |
| 4h | 4-12 hours |

**Why so long?**
The system polls market data at timeframe intervals, waiting for technical indicators to confirm or invalidate the signal. This is intentional to avoid false signals!

## Quick Test (5 minutes)

Want to test quickly? Use a 1-minute chart:

1. Find a 1-minute crypto chart (TradingView, MEXC, etc.)
2. Take screenshot
3. Upload via API with `auto_analyze=true`
4. Wait 5-10 minutes
5. You should get a response!

## Still Not Working?

If you've tried everything:

1. Run full diagnostics:
```bash
python diagnose_analysis.py
```

2. Check your `.env` file has all required variables

3. Check server logs for specific error messages

4. Try manual analysis of a test image:
```bash
python -c "
from web_app import run_trading_analysis
result = run_trading_analysis('uploads/test_chart_custom_name.png')
print(result)
"
```

5. Ensure you're in the virtual environment:
```bash
which python  # Should show /Users/hpk/trassist2/venv/bin/python
```

## Pro Tip: Apple Shortcuts

If using Apple Shortcuts, you can add a "Wait" action to remind yourself:
1. Upload image via shortcut
2. Show alert: "Analysis started! Check Telegram in 10 minutes"
3. Set a reminder for 10 minutes later

## Contact/Debug

If still having issues, check:
- `server.log` or `nohup.out` for detailed error messages
- `llm_outputs/` directory for recent analysis files
- Ensure all services (Telegram, MEXC, OpenAI) are accessible

