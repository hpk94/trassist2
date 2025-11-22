#!/usr/bin/env python3
"""
Interactive setup script for Telegram notifications
This script helps you configure your Telegram notification settings
"""

import os
import sys
from pathlib import Path

def main():
    print("🔧 Trading Assistant - Telegram Notification Setup")
    print("=" * 50)
    
    env_file = Path(".env")
    
    # Check if .env file exists
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create a .env file in the project root first.")
        print("You can copy from .env.example if it exists.")
        return
    
    print("💬 Telegram Notification Setup")
    print("\nThis will help you set up Telegram notifications.")
    print("You can skip this and configure manually if you prefer.")
    
    choice = input("\nDo you want to set up Telegram notifications? (y/n): ").strip().lower()
    
    if choice != 'y':
        print("Setup skipped. You can configure notifications manually later.")
        print("See TELEGRAM_SETUP.md for instructions.")
        return
    
    # Read current .env file
    env_content = env_file.read_text()
    env_lines = env_content.split('\n')
    
    print("\n💬 Telegram Setup:")
    print("1. Create a Telegram bot with @BotFather")
    print("2. Get your chat ID from @userinfobot")
    print("3. Enter the credentials below:")
    
    telegram_bot_token = input("\nEnter your Telegram Bot Token: ").strip()
    telegram_chat_id = input("Enter your Telegram Chat ID: ").strip()
    
    # Update or add Telegram settings
    update_env_line(env_lines, "TELEGRAM_BOT_TOKEN", telegram_bot_token)
    update_env_line(env_lines, "TELEGRAM_CHAT_ID", telegram_chat_id)
    
    # Write updated .env file
    env_file.write_text('\n'.join(env_lines))
    
    print("\n✅ Configuration saved!")
    print("\n🧪 Testing your setup...")
    
    # Test the configuration
    try:
        from services.notification_service import test_notification_system
        results = test_notification_system()
        
        print("\n📊 Test Results:")
        print(f"Telegram: {'✅ Success' if results.get('telegram') else '❌ Failed'}")
        
        if any(results.values()):
            print("\n🎉 Setup complete! You should receive a test notification.")
        else:
            print("\n⚠️  Setup saved but tests failed. Check your credentials.")
            print("See TELEGRAM_SETUP.md for troubleshooting.")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Configuration saved but please check your settings.")
        print("See TELEGRAM_SETUP.md for troubleshooting.")
    
    print("\n" + "=" * 50)
    print("Setup complete!")

def update_env_line(env_lines, key, value):
    """Update or add a line in the .env file"""
    found = False
    for i, line in enumerate(env_lines):
        if line.startswith(f"{key}="):
            env_lines[i] = f"{key}={value}"
            found = True
            break
    
    if not found:
        env_lines.append(f"{key}={value}")

if __name__ == "__main__":
    main()
