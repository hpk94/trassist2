# Telegram Integration Setup Guide

This guide will help you set up Telegram notifications for your Trading Assistant, so you can receive chart uploads and analysis results directly in Telegram.

## Features

When integrated with Telegram, you'll receive:
- 📊 **Uploaded trading chart images** - Images are sent immediately when uploaded
- 🤖 **Analysis results** - Automated analysis results with symbol, timeframe, direction, and confidence
- 📱 **Trade notifications** - Valid and invalidated trade signals

## Setup Steps

### 1. Create a Telegram Bot

1. **Open Telegram** and search for `@BotFather`
2. **Start a chat** with BotFather
3. **Send the command**: `/newbot`
4. **Choose a name** for your bot (e.g., "My Trading Assistant")
5. **Choose a username** for your bot (must end in 'bot', e.g., "my_trading_assistant_bot")
6. **Save the bot token** - BotFather will give you a token that looks like:
   ```
   123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

### 2. Get Your Chat ID

You need to find your chat ID to receive messages from the bot.

**Option A: Use the @userinfobot method (Easiest)**
1. Open Telegram and search for `@userinfobot`
2. Start a chat and send any message
3. The bot will reply with your user information, including your **ID** (this is your chat ID)

**Option B: Use the API method**
1. Start a chat with your newly created bot
2. Send any message to the bot (e.g., "/start")
3. Open this URL in your browser (replace `YOUR_BOT_TOKEN` with your actual token):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
4. Look for `"chat":{"id":` in the response - the number after it is your chat ID

### 3. Add Credentials to .env File

Add the following lines to your `.env` file in the project root:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

**Example:**
```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
```

### 4. Test Your Setup

Run the notification test script to verify everything is working:

```bash
cd /Users/hpk/trassist2
source venv/bin/activate
python -c "from services.notification_service import test_notification_system; test_notification_system()"
```

You should receive a test notification in your Telegram chat! ✅

## What You'll Receive

### 1. Image Upload Notification

When you upload a trading chart, you'll immediately receive:

```
📊 New Trading Chart Uploaded

Filename: trading_chart_20250131_143022.png
Size: 245.67 KB
Time: 20250131_143022
```

The uploaded image will be attached to this message.

### 2. Analysis Results

After the analysis completes, you'll receive another message with the chart image and:

```
📊 Trading Chart Analysis

Symbol: BTCUSDT
Timeframe: 1h
Direction: LONG
Confidence: 85.0%

Analysis complete!
```

### 3. Trade Signals

When a valid trade is detected, you'll get a full trade notification:

```
🚀 VALID TRADE SIGNAL

📊 Symbol: BTCUSDT
📈 Direction: LONG
💰 Price: $50,000.00
📊 RSI: 45.2
🎯 Confidence: 85%

✅ Trade approved by AI gate
⏰ Time: 2025-01-31 15:30:45

Check your trading platform!
```

## Usage

### Upload Image via API

Upload a trading chart image and it will be automatically sent to Telegram:

```bash
curl -X POST http://localhost:5001/api/upload-image \
  -F "image=@your_chart.png" \
  -F "auto_analyze=true"
```

### Web Interface

1. Start the server: `./start_server.sh`
2. Open `upload_example.html` in your browser
3. Upload a chart image
4. Check your Telegram for the image and analysis results!

## Troubleshooting

### Not receiving messages?

1. **Check your .env file**
   - Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set correctly
   - No quotes needed around the values
   - No spaces around the `=` sign

2. **Verify the bot token**
   - Make sure you copied the entire token from BotFather
   - Test the token by opening: `https://api.telegram.org/botYOUR_TOKEN/getMe`

3. **Verify the chat ID**
   - Make sure you started a chat with your bot first
   - Chat ID should be a number (can be negative for group chats)

4. **Check bot permissions**
   - Make sure you haven't blocked the bot
   - If using a group chat, ensure the bot is added to the group

### Bot is not responding?

- Restart the Flask server after adding the credentials
- Check the console logs for any error messages
- Run the test script to see detailed error messages

### Images not sending?

- Make sure the image file exists and is accessible
- Check file size - Telegram has a 50MB limit for photos
- Verify the image format is supported (PNG, JPG, GIF)

## Advanced Configuration

### Using with Group Chats

To send notifications to a Telegram group:

1. Add your bot to the group
2. Make the bot an administrator (optional, but recommended)
3. Get the group chat ID (it will be a negative number)
4. Update `TELEGRAM_CHAT_ID` in your `.env` file with the group chat ID

### Custom Captions

You can customize the message captions by editing the functions in:
- `services/notification_service.py` - Modify `send_telegram_analysis()` function
- `web_app.py` - Modify the caption in the upload endpoint

## Benefits of Telegram Integration

✅ **Instant Notifications** - Receive updates in real-time on your phone
✅ **Visual Charts** - See the actual chart images in your chat
✅ **Analysis Results** - Get detailed trading analysis with confidence scores
✅ **Trade Signals** - Never miss a valid trade opportunity
✅ **Cross-Platform** - Works on phone, tablet, desktop, and web
✅ **Message History** - All notifications are saved in your chat history
✅ **No Phone Number Required** - Works with just your Telegram account

## Next Steps

- Configure other notification methods: See `NOTIFICATION_SETUP.md`
- Learn about the full API: See `API_DOCUMENTATION.md`
- Explore other features: See `FEATURE_SUMMARY.md`

---

**Need Help?** Check the console logs for detailed error messages when testing your setup.

