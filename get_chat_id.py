#!/usr/bin/env python3
"""
Helper script to get your Telegram Chat ID
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

print()
print("=" * 60)
print("GET YOUR TELEGRAM CHAT ID")
print("=" * 60)
print()

token = os.getenv("TELEGRAM_BOT_TOKEN")

if not token:
    print("❌ TELEGRAM_BOT_TOKEN not found in .env file")
    print()
    exit(1)

print("✅ Bot token found!")
print()
print("INSTRUCTIONS:")
print("=" * 60)
print()
print("1. Open Telegram on your phone or desktop")
print("2. Search for your bot (the one you created with @BotFather)")
print("3. Start a chat and send ANY message to your bot")
print("   (For example: /start or Hello)")
print()
print("4. After you send a message, press ENTER here...")
input()

print()
print("Fetching chat updates from Telegram API...")
print()

try:
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    
    if not data.get("ok"):
        print(f"❌ API Error: {data}")
        exit(1)
    
    results = data.get("result", [])
    
    if not results:
        print("❌ No messages found!")
        print()
        print("Make sure you:")
        print("  1. Started a chat with your bot")
        print("  2. Sent at least one message")
        print("  3. Used the correct bot (check the username)")
        print()
        print("Then run this script again.")
        exit(1)
    
    # Get the most recent chat
    latest_message = results[-1]
    chat = latest_message.get("message", {}).get("chat", {})
    chat_id = chat.get("id")
    first_name = chat.get("first_name", "")
    username = chat.get("username", "")
    
    print("✅ SUCCESS! Found your chat:")
    print()
    print(f"   Name: {first_name}")
    if username:
        print(f"   Username: @{username}")
    print(f"   Chat ID: {chat_id}")
    print()
    print("=" * 60)
    print("NEXT STEP: Add this to your .env file")
    print("=" * 60)
    print()
    print("Add this line to your .env file:")
    print()
    print(f"   TELEGRAM_CHAT_ID={chat_id}")
    print()
    print("Then restart your server:")
    print("   ./start_server.sh")
    print()
    print("And test it:")
    print("   python test_telegram.py")
    print()
    
except requests.exceptions.RequestException as e:
    print(f"❌ Network error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)



