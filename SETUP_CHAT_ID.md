# Get Your Telegram Chat ID - Step by Step

You already have your bot token ✅. Now you just need your **Chat ID**.

## Quick Method (Easiest) - 2 Minutes

### Step 1: Use @userinfobot
1. Open Telegram (phone or desktop)
2. Search for: **@userinfobot**
3. Start a chat and send **any message**
4. The bot will reply with your info, including your **ID** number
5. **Copy that ID number** ← This is your TELEGRAM_CHAT_ID!

### Step 2: Add to .env file
Open `/Users/hpk/trassist2/.env` and add this line:

```bash
TELEGRAM_CHAT_ID=YOUR_ID_NUMBER_HERE
```

For example, if your ID is `987654321`:
```bash
TELEGRAM_CHAT_ID=987654321
```

### Step 3: Restart server
```bash
./start_server.sh
```

### Step 4: Test it!
```bash
python test_telegram.py
```

---

## Alternative Method - Using Your Bot

If @userinfobot doesn't work, use this method:

### Step 1: Message your bot
1. Open Telegram
2. Search for **your bot** (the one you created with @BotFather)
3. Start a chat and send: **/start** or **Hello**

### Step 2: Run the helper script
```bash
python get_chat_id.py
```

This will fetch your chat ID from your bot's messages.

### Step 3: Copy the chat ID
The script will show you your Chat ID. Copy it.

### Step 4: Add to .env file
```bash
TELEGRAM_CHAT_ID=YOUR_ID_HERE
```

### Step 5: Restart and test
```bash
./start_server.sh
python test_telegram.py
```

---

## Your .env file should look like this:

```bash
# Telegram Configuration
TELEGRAM_BOT_TOKEN=8433274774:AAHa3bKiR... (already set ✅)
TELEGRAM_CHAT_ID=987654321 (← add this line)

# Other configurations...
```

---

## Troubleshooting

### "No messages found"
- Make sure you sent a message to your bot first
- Try sending another message
- Wait a few seconds and try again

### "Bot not responding"
- Make sure you're messaging the correct bot
- Check the bot username matches what @BotFather gave you
- Try the @userinfobot method instead

### Still stuck?
Run the config check:
```bash
python check_telegram_config.py
```

---

## What happens next?

Once you add the TELEGRAM_CHAT_ID:
1. ✅ Images uploaded will appear in Telegram
2. ✅ Analysis results will be sent to Telegram
3. ✅ Trade signals will notify you in Telegram

Ready? Add that Chat ID and restart the server! 🚀


