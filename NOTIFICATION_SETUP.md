# Notification Setup Guide

This guide will help you set up Telegram notifications for valid trades in your Trading Assistant.

## Telegram (Recommended for Images & Analysis)

Telegram is the best option for receiving trading chart images and analysis results.

### Setup Steps:
1. **Create a Telegram bot** with @BotFather
2. **Get your chat ID** from @userinfobot
3. **Add credentials to your .env file:**

```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### Benefits:
- ✅ Receive chart images instantly
- ✅ Get detailed analysis results
- ✅ Visual notifications with rich formatting
- ✅ Works on all platforms (phone, tablet, desktop)
- ✅ Message history saved in chat

**For detailed Telegram setup instructions, see [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md)**

## Testing Your Setup

Run this command to test your notifications:

```bash
cd /Users/hpk/trassist2
python -c "from services.notification_service import test_notification_system; test_notification_system()"
```

## What You'll Receive

When a valid trade is detected, you'll get a notification like:

```
🚀 VALID TRADE SIGNAL

📊 Symbol: BTCUSDT
📈 Direction: LONG
💰 Price: $50,000.00
📊 RSI: 45.2
🎯 Confidence: 85%

✅ Trade approved by AI gate
⏰ Time: 2025-01-07 15:30:45

Check your trading platform!
```

## Troubleshooting

### Telegram Issues:
- Verify your bot token and chat ID are correct
- Check that the bot is running and has permission to send messages
- Ensure you've started a conversation with the bot
- See [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md) for detailed troubleshooting

### General Issues:
- Check your .env file is in the project root
- Ensure all required environment variables are set
- Check the console output for error messages
