#!/usr/bin/env python3
"""
Test script for the notification system
Run this to verify your iPhone notifications are working
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
    
    pushover_token = os.getenv("PUSHOVER_TOKEN")
    pushover_user = os.getenv("PUSHOVER_USER")
    email_username = os.getenv("EMAIL_USERNAME")
    email_password = os.getenv("EMAIL_PASSWORD")
    email_to = os.getenv("EMAIL_TO")
    
    print(f"Pushover Token: {'✅ Set' if pushover_token else '❌ Not set'}")
    print(f"Pushover User: {'✅ Set' if pushover_user else '❌ Not set'}")
    print(f"Email Username: {'✅ Set' if email_username else '❌ Not set'}")
    print(f"Email Password: {'✅ Set' if email_password else '❌ Not set'}")
    print(f"Email To: {'✅ Set' if email_to else '❌ Not set'}")
    
    if not any([pushover_token, email_username]):
        print("\n⚠️  No notification methods configured!")
        print("Please set up at least one notification method:")
        print("1. Pushover (recommended): Set PUSHOVER_TOKEN and PUSHOVER_USER")
        print("2. Email: Set EMAIL_USERNAME, EMAIL_PASSWORD, and EMAIL_TO")
        print("\nSee NOTIFICATION_SETUP.md for detailed instructions.")
        return
    
    print("\n🚀 Running notification tests...")
    
    # Test the notification system
    results = test_notification_system()
    
    print("\n📊 Test Results:")
    print(f"Pushover: {'✅ Success' if results.get('pushover') else '❌ Failed'}")
    print(f"Email: {'✅ Success' if results.get('email') else '❌ Failed'}")
    
    if any(results.values()):
        print("\n🎉 At least one notification method is working!")
        print("You should receive a test notification on your iPhone.")
    else:
        print("\n❌ All notification methods failed.")
        print("Please check your configuration and try again.")
    
    print("\n" + "=" * 50)
    print("Test complete!")

if __name__ == "__main__":
    main()
