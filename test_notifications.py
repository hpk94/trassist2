#!/usr/bin/env python3
"""
Test script for the notification system
Run this to verify your Telegram notifications are working
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.notification_service import test_notification_system, NotificationService

def main():
    print("🧪 Testing Trading Assistant Notification System")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check if required environment variables are set
    print("\n📋 Checking configuration...")
    
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    print(f"Telegram Bot Token: {'✅ Set' if telegram_bot_token else '❌ Not set'}")
    print(f"Telegram Chat ID: {'✅ Set' if telegram_chat_id else '❌ Not set'}")
    
    if not telegram_bot_token or not telegram_chat_id:
        print("\n⚠️  Telegram notifications not configured!")
        print("Please set up Telegram notifications:")
        print("1. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        print("\nSee TELEGRAM_SETUP.md or NOTIFICATION_SETUP.md for detailed instructions.")
        return
    
    print("\n🚀 Running notification tests...")
    
    # Test the notification system
    results = test_notification_system()
    
    print("\n📊 Test Results:")
    print(f"Telegram: {'✅ Success' if results.get('telegram') else '❌ Failed'}")
    
    if results.get('telegram'):
        print("\n🎉 Telegram notifications are working!")
        print("You should receive a test notification on Telegram.")
    else:
        print("\n❌ Telegram notification failed.")
        print("Please check your configuration and try again.")
        print("See TELEGRAM_SETUP.md for troubleshooting.")
    
    print("\n" + "=" * 50)
    print("Test complete!")

if __name__ == "__main__":
    main()
