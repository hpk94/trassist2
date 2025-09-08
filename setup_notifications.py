#!/usr/bin/env python3
"""
Interactive setup script for iPhone notifications
This script helps you configure your notification settings
"""

import os
import sys
from pathlib import Path

def main():
    print("🔧 Trading Assistant - iPhone Notification Setup")
    print("=" * 50)
    
    env_file = Path(".env")
    
    # Check if .env file exists
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create a .env file in the project root first.")
        print("You can copy from .env.example if it exists.")
        return
    
    print("📱 iPhone Notification Setup")
    print("\nChoose your preferred notification method:")
    print("1. Pushover (Recommended - most reliable)")
    print("2. Email to SMS (Backup method)")
    print("3. Both methods")
    print("4. Skip setup")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "4":
        print("Setup skipped. You can configure notifications manually later.")
        return
    
    # Read current .env file
    env_content = env_file.read_text()
    env_lines = env_content.split('\n')
    
    # Update .env file based on choice
    if choice in ["1", "3"]:
        print("\n📱 Pushover Setup:")
        print("1. Download Pushover app from App Store")
        print("2. Create account at https://pushover.net/")
        print("3. Get your User Key from the dashboard")
        print("4. Create an application to get your API Token")
        
        pushover_token = input("\nEnter your Pushover API Token: ").strip()
        pushover_user = input("Enter your Pushover User Key: ").strip()
        
        # Update or add Pushover settings
        update_env_line(env_lines, "PUSHOVER_TOKEN", pushover_token)
        update_env_line(env_lines, "PUSHOVER_USER", pushover_user)
    
    if choice in ["2", "3"]:
        print("\n📧 Email Setup:")
        print("1. Use Gmail with App Password (recommended)")
        print("2. Enable 2-factor authentication")
        print("3. Generate an App Password")
        print("4. Use your carrier's SMS email address")
        
        email_username = input("\nEnter your email address: ").strip()
        email_password = input("Enter your email password (or App Password): ").strip()
        email_to = input("Enter recipient (your phone's SMS email): ").strip()
        
        # Update or add email settings
        update_env_line(env_lines, "EMAIL_USERNAME", email_username)
        update_env_line(env_lines, "EMAIL_PASSWORD", email_password)
        update_env_line(env_lines, "EMAIL_TO", email_to)
    
    # Write updated .env file
    env_file.write_text('\n'.join(env_lines))
    
    print("\n✅ Configuration saved!")
    print("\n🧪 Testing your setup...")
    
    # Test the configuration
    try:
        from services.notification_service import test_notification_system
        results = test_notification_system()
        
        print("\n📊 Test Results:")
        print(f"Pushover: {'✅ Success' if results.get('pushover') else '❌ Failed'}")
        print(f"Email: {'✅ Success' if results.get('email') else '❌ Failed'}")
        
        if any(results.values()):
            print("\n🎉 Setup complete! You should receive a test notification.")
        else:
            print("\n⚠️  Setup saved but tests failed. Check your credentials.")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Configuration saved but please check your settings.")
    
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
