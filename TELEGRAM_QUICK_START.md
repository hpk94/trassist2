# Telegram Integration - Quick Start

Get your Trading Assistant sending charts and analysis to Telegram in 5 minutes!

## Prerequisites

✅ Server running on port 5001
✅ Telegram account

## Setup (3 Steps)

### Step 1: Create Telegram Bot (2 minutes)

1. Open Telegram, search for `@BotFather`
2. Send: `/newbot`
3. Choose name: `My Trading Bot`
4. Choose username: `my_trading_bot` (must end with 'bot')
5. **Copy the token** (looks like: `123456789:ABCdefGHI...`)

### Step 2: Get Your Chat ID (1 minute)

1. Search for `@userinfobot` in Telegram
2. Send any message
3. **Copy your ID** (a number like: `987654321`)

### Step 3: Configure .env (1 minute)

Add to your `.env` file:

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
```

## Test It (30 seconds)

```bash
python test_telegram.py
```

You should receive 3 messages in Telegram! ✅

## Usage

### Option 1: Upload via Web Interface

```bash
./start_server.sh
open upload_example.html
```

Upload a chart → Check Telegram → See image & analysis! 📊

### Option 2: Upload via API

```bash
python example_telegram_upload.py your_chart.png
```

### Option 3: Upload via cURL

```bash
curl -X POST http://localhost:5001/api/upload-image \
  -F "image=@your_chart.png" \
  -F "auto_analyze=true"
```

## What You Get in Telegram

**Immediately:** Uploaded chart image
```
📊 New Trading Chart Uploaded
Filename: trading_chart_20250131_143022.png
Size: 245.67 KB
```

**After Analysis:** Analysis results
```
📊 Trading Chart Analysis
Symbol: BTCUSDT
Timeframe: 1h
Direction: LONG
Confidence: 85.0%
```

## Troubleshooting

### Not working?

1. **Check .env file**: Make sure both variables are set
2. **Restart server**: `./start_server.sh`
3. **Run test**: `python test_telegram.py`
4. **Check bot**: Make sure you started a chat with your bot

### Still not working?

See detailed troubleshooting in [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md)

## Files Reference

- 📖 **Detailed Setup**: `TELEGRAM_SETUP.md`
- 🧪 **Test Script**: `test_telegram.py`
- 📝 **Example Usage**: `example_telegram_upload.py`
- 📊 **Full Summary**: `TELEGRAM_INTEGRATION_SUMMARY.md`

## Benefits

✅ Instant visual feedback
✅ See actual chart images
✅ Get detailed analysis
✅ Works on all devices
✅ Message history saved

---

**That's it!** Start uploading charts and receive them in Telegram instantly! 🚀

