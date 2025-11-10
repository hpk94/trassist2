#!/usr/bin/env python3
"""
Check Telegram configuration and guide setup
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("TELEGRAM CONFIGURATION CHECK")
print("=" * 60)
print()

# Check for .env file
env_file = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_file):
    print(f"✅ .env file found at: {env_file}")
else:
    print(f"❌ .env file NOT found at: {env_file}")
    print()
    print("Creating .env file...")
    with open(env_file, 'w') as f:
        f.write("# Telegram Configuration\n")
        f.write("TELEGRAM_BOT_TOKEN=\n")
        f.write("TELEGRAM_CHAT_ID=\n")
        f.write("\n")
    print(f"✅ Created .env file at: {env_file}")

print()

# Check environment variables
token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

print("TELEGRAM_BOT_TOKEN:", end=" ")
if token:
    print(f"✅ Set (length: {len(token)} characters)")
    print(f"   Preview: {token[:20]}...")
else:
    print("❌ NOT SET or EMPTY")

print()
print("TELEGRAM_CHAT_ID:", end=" ")
if chat_id:
    print(f"✅ Set ({chat_id})")
else:
    print("❌ NOT SET or EMPTY")

print()
print("=" * 60)

if not token or not chat_id:
    print("SETUP REQUIRED")
    print("=" * 60)
    print()
    print("You need to set up your Telegram bot. Here's how:")
    print()
    print("STEP 1: Create a Telegram Bot (2 minutes)")
    print("  1. Open Telegram and search for: @BotFather")
    print("  2. Send command: /newbot")
    print("  3. Choose a name (e.g., 'My Trading Bot')")
    print("  4. Choose a username (must end with 'bot')")
    print("  5. Copy the TOKEN that BotFather gives you")
    print()
    print("STEP 2: Get Your Chat ID (1 minute)")
    print("  1. Search for: @userinfobot")
    print("  2. Send any message")
    print("  3. Copy your ID number")
    print()
    print("STEP 3: Add to .env file")
    print(f"  Edit: {env_file}")
    print()
    print("  Add these lines:")
    print("  TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz")
    print("  TELEGRAM_CHAT_ID=987654321")
    print()
    print("  (Replace with your actual token and chat ID)")
    print()
    print("STEP 4: Restart the server")
    print("  ./start_server.sh")
    print()
    print("Then run: python test_telegram.py")
    print()
else:
    print("CONFIGURATION LOOKS GOOD!")
    print("=" * 60)
    print()
    print("Next step: Test your configuration")
    print("  python test_telegram.py")
    print()


