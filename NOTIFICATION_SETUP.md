# Notification Setup Guide

This guide will help you set up notifications for valid trades in your Trading Assistant. Multiple notification methods are supported and can be used simultaneously.

## Option 1: Telegram (Recommended for Images & Analysis)

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

## Option 2: Pushover (Recommended for iPhone)

Pushover is the most reliable method for iPhone notifications.

### Setup Steps:
1. **Download Pushover app** from the App Store
2. **Create account** at https://pushover.net/
3. **Get your User Key** from the Pushover dashboard
4. **Create an application** to get your API Token
5. **Add credentials to your .env file:**

```bash
PUSHOVER_TOKEN=your_app_token_here
PUSHOVER_USER=your_user_key_here
```

### Benefits:
- ✅ Instant delivery
- ✅ Custom sounds
- ✅ Priority levels
- ✅ Works even when app is closed
- ✅ Reliable delivery

## Option 3: Email to SMS (Backup)

Send email notifications that get converted to SMS on your phone.

### Setup Steps:
1. **Use Gmail with App Password:**
   - Enable 2-factor authentication
   - Generate an App Password
   - Add to your .env file:

```bash
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
EMAIL_TO=your_phone_number@your_carrier.com
```

2. **Find your carrier's SMS email address:**
   - Verizon: `your_number@vtext.com`
   - AT&T: `your_number@txt.att.net`
   - T-Mobile: `your_number@tmomail.net`
   - Sprint: `your_number@messaging.sprintpcs.com`

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

### Pushover Issues:
- Verify your token and user key are correct
- Check that the Pushover app is installed and logged in
- Ensure you have internet connectivity

### Email Issues:
- Verify your email credentials
- Check that 2FA is enabled and you're using an App Password
- Confirm your carrier's SMS email address is correct

### General Issues:
- Check your .env file is in the project root
- Ensure all required environment variables are set
- Check the console output for error messages
