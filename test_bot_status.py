#!/usr/bin/env python3
"""
Test script to verify Telegram bot and analysis state
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the bot module
try:
    from telegram_bot import get_bot, get_analysis_state
    print("✅ Successfully imported telegram_bot module")
except Exception as e:
    print(f"❌ Failed to import telegram_bot: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("TELEGRAM BOT STATUS CHECK")
print("="*60 + "\n")

# Check environment variables
print("1. Environment Variables:")
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

if bot_token:
    print(f"   ✅ TELEGRAM_BOT_TOKEN is set (length: {len(bot_token)})")
else:
    print("   ❌ TELEGRAM_BOT_TOKEN is NOT set")

if chat_id:
    print(f"   ✅ TELEGRAM_CHAT_ID is set ({chat_id})")
else:
    print("   ❌ TELEGRAM_CHAT_ID is NOT set")

if not bot_token or not chat_id:
    print("\n   ⚠️  Bot cannot function without these credentials!")
    sys.exit(1)

print("\n2. Getting Bot Instance:")
try:
    bot = get_bot()
    print("   ✅ Bot instance created successfully")
except Exception as e:
    print(f"   ❌ Failed to create bot instance: {e}")
    sys.exit(1)

print("\n3. Getting Analysis State:")
try:
    state = get_analysis_state()
    print("   ✅ Analysis state instance created successfully")
    
    # Get current status
    status_info = state.get_status_info()
    print(f"\n   Current Status:")
    print(f"   - Running: {status_info.get('running')}")
    if status_info.get('running'):
        print(f"   - Symbol: {status_info.get('symbol')}")
        print(f"   - Direction: {status_info.get('direction')}")
        print(f"   - Start Time: {status_info.get('start_time')}")
        print(f"   - Elapsed: {status_info.get('elapsed_seconds')}s")
    else:
        print(f"   - No analysis currently running")
        
except Exception as e:
    print(f"   ❌ Failed to get analysis state: {e}")
    sys.exit(1)

print("\n4. Testing Bot Communication:")
try:
    success = bot.send_message("""
<b>🧪 Test Message from test_bot_status.py</b>

If you see this message, your Telegram bot is configured correctly!

Try these commands:
• /status - Check analysis status
• /help - See all commands
    """.strip())
    
    if success:
        print("   ✅ Successfully sent test message to Telegram")
        print("   📱 Check your Telegram app for the test message!")
    else:
        print("   ❌ Failed to send test message (check bot token/chat ID)")
except Exception as e:
    print(f"   ❌ Exception while sending test message: {e}")

print("\n5. Bot Polling Status:")
if bot._running:
    print("   ✅ Bot is actively polling for commands")
else:
    print("   ⚠️  Bot is NOT polling (you need to call start_bot())")
    print("   💡 The bot auto-starts when you run web_app.py or app.py")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60 + "\n")

print("📋 Summary:")
print("   - Environment variables: " + ("✅ OK" if bot_token and chat_id else "❌ Missing"))
print("   - Bot instance: ✅ OK")
print("   - Analysis state: ✅ OK")
print("   - Current analysis: " + ("🟢 Running" if status_info.get('running') else "⚪ Idle"))

print("\n💡 Next Steps:")
if not status_info.get('running'):
    print("   1. Start web_app.py or app.py")
    print("   2. Upload a chart image")
    print("   3. In Telegram, send: /status")
    print("   4. You should see the analysis running!")
else:
    print("   Analysis is currently running!")
    print("   In Telegram, send: /status to see details")

print("\n✅ If you received the test message in Telegram, everything is working!")
print("   If not, check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env\n")

