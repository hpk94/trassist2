# Telegram Bot Commands

Your trading assistant now includes a Telegram bot that allows you to control the analysis status remotely!

## 🤖 Available Commands

### `/status`
Check the current status of any running analysis:
- Shows if an analysis is currently running
- Displays the symbol and direction being analyzed
- Shows elapsed time since analysis started
- Shows the current status

**Example:**
```
/status
```

**Response when running:**
```
📊 Analysis Status: RUNNING

Symbol: BTCUSDT
Direction: Long
Start Time: 2025-10-31 12:34:56
Elapsed Time: 3m 45s

Status: Analysis in progress...

Use /stop to cancel this analysis.
```

**Response when idle:**
```
📊 Analysis Status: IDLE

No analysis is currently running.

Upload a chart to start a new analysis.
```

### `/stop`
Stop the currently running analysis:
- Cancels analysis gracefully at the next checkpoint
- Allows you to start a new analysis
- Useful if you uploaded the wrong chart or want to start over

**Example:**
```
/stop
```

**Response:**
```
⏹️ Stop Request Received

The current analysis will stop gracefully at the next checkpoint.

You can upload a new chart once the analysis has stopped.
```

### `/help`
Show help information and available commands.

**Example:**
```
/help
```

## 🔄 How It Works

1. **Start Analysis**: Upload a chart image through your web interface or API
2. **Check Status**: Use `/status` in Telegram to monitor progress
3. **Stop if Needed**: Use `/stop` to cancel and start fresh
4. **Get Notifications**: Receive automatic updates when:
   - Analysis starts
   - Chart is analyzed
   - Signal is validated or invalidated
   - Trade is approved or rejected
   - Analysis completes

## ⏸️ What Happens When You Stop

When you use `/stop`:
1. The bot marks the analysis for graceful shutdown
2. The current polling cycle completes
3. Analysis stops at the next checkpoint
4. You receive a confirmation message
5. You can immediately start a new analysis

## 📊 Status Updates

The bot automatically sends periodic status updates:
- Every 5 polling cycles during signal validation
- When important events occur (validation, invalidation, etc.)
- When analysis completes

## 🚀 Quick Start

1. Make sure your `.env` file has:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```

2. Start your trading app:
   ```bash
   python web_app.py
   # or
   python app.py
   ```

3. The bot automatically starts and listens for commands!

4. Try it out:
   - Send `/status` to check current status
   - Upload a chart to start analysis
   - Send `/stop` if you need to cancel

## 💡 Tips

- **Multiple Analyses**: The bot prevents multiple analyses from running simultaneously
- **Graceful Shutdown**: Stopping doesn't lose data - all progress is saved
- **Real-time Updates**: Get instant notifications about your trades
- **Command Any Time**: You can check status or stop analysis at any point

## 🔧 Troubleshooting

### Bot not responding?
1. Check your `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`
2. Make sure the app is running (web_app.py or app.py)
3. Try `/help` to test if the bot is listening

### Can't stop analysis?
- If `/stop` doesn't work, the analysis may have just finished
- Check `/status` to see current state
- Restart the app as a last resort

### Bot stopped working?
- The app automatically reconnects if there's a temporary network issue
- Check the console for any error messages
- Restart the app to reset the bot connection

## 📱 Integration

The Telegram bot works seamlessly with:
- **Web Interface** (`web_app.py`)
- **CLI Script** (`app.py`)
- **API Uploads** (`/api/upload-image`)

All methods share the same analysis state, so you can:
- Upload via API
- Check status via Telegram
- Stop via Telegram
- Start new analysis via Web Interface

Enjoy your remote trading assistant! 🎉

