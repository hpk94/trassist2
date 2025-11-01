#!/usr/bin/env python3
"""
Test script for Telegram integration

This script tests the Telegram notification functionality including:
1. Sending a simple text message
2. Sending an image with caption
3. Sending a trading analysis notification
"""

import os
import sys
from dotenv import load_dotenv
from services.notification_service import NotificationService

# Load environment variables
load_dotenv()

def test_telegram_setup():
    """Test if Telegram credentials are configured"""
    print("=" * 60)
    print("Testing Telegram Setup")
    print("=" * 60)
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found in .env file")
        return False
    
    if not chat_id:
        print("❌ TELEGRAM_CHAT_ID not found in .env file")
        return False
    
    print(f"✅ TELEGRAM_BOT_TOKEN: {token[:20]}..." if len(token) > 20 else f"✅ TELEGRAM_BOT_TOKEN: {token}")
    print(f"✅ TELEGRAM_CHAT_ID: {chat_id}")
    print()
    return True

def test_text_message():
    """Test sending a text message"""
    print("=" * 60)
    print("Test 1: Sending Text Message")
    print("=" * 60)
    
    service = NotificationService()
    
    test_data = {
        "symbol": "BTCUSDT",
        "direction": "long",
        "current_price": 50000.00,
        "confidence": 0.85,
        "current_rsi": 45.2,
        "triggered_conditions": []
    }
    
    print("Sending test trade notification...")
    results = service.send_trade_notification(test_data, "valid_trade")
    
    if results.get("telegram"):
        print("✅ Text message sent successfully!")
        print("   Check your Telegram app for the notification")
        print()
        return True
    else:
        print("❌ Failed to send text message")
        print("   Check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        print()
        return False

def test_image_message():
    """Test sending an image with caption"""
    print("=" * 60)
    print("Test 2: Sending Image with Caption")
    print("=" * 60)
    
    # Find a test image in the project
    test_images = [
        "BTCUSDT.P_2025-09-02_22-55-40_545b4.png",
        "BTCUSDT.P_2025-09-02_19-28-40_20635.png",
        "image_example.png"
    ]
    
    test_image = None
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break
    
    if not test_image:
        print("⚠️  No test images found in project directory")
        print("   Skipping image test")
        print()
        return None
    
    print(f"Using test image: {test_image}")
    
    service = NotificationService()
    
    caption = """<b>🧪 Test Image Upload</b>

<b>This is a test message from your Trading Assistant</b>

If you can see this image and caption, your Telegram integration is working correctly! ✅
"""
    
    print("Sending test image...")
    success = service.send_telegram_image(test_image, caption)
    
    if success:
        print("✅ Image sent successfully!")
        print("   Check your Telegram app for the image")
        print()
        return True
    else:
        print("❌ Failed to send image")
        print()
        return False

def test_analysis_message():
    """Test sending a trading analysis"""
    print("=" * 60)
    print("Test 3: Sending Trading Analysis")
    print("=" * 60)
    
    # Find a test image
    test_images = [
        "BTCUSDT.P_2025-09-02_22-55-40_545b4.png",
        "BTCUSDT.P_2025-09-02_19-28-40_20635.png",
        "image_example.png"
    ]
    
    test_image = None
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break
    
    if not test_image:
        print("⚠️  No test images found in project directory")
        print("   Skipping analysis test")
        print()
        return None
    
    print(f"Using test image: {test_image}")
    
    service = NotificationService()
    
    analysis_data = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "direction": "long",
        "confidence": 0.875
    }
    
    print("Sending test trading analysis...")
    success = service.send_telegram_analysis(test_image, analysis_data)
    
    if success:
        print("✅ Trading analysis sent successfully!")
        print("   Check your Telegram app for the analysis")
        print()
        return True
    else:
        print("❌ Failed to send trading analysis")
        print()
        return False

def main():
    """Run all Telegram tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "TELEGRAM INTEGRATION TEST" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Test 0: Check setup
    if not test_telegram_setup():
        print("\n❌ Telegram is not configured properly")
        print("\nPlease follow these steps:")
        print("1. Create a Telegram bot with @BotFather")
        print("2. Get your chat ID from @userinfobot")
        print("3. Add these to your .env file:")
        print("   TELEGRAM_BOT_TOKEN=your_token_here")
        print("   TELEGRAM_CHAT_ID=your_chat_id_here")
        print("\nSee TELEGRAM_SETUP.md for detailed instructions")
        return 1
    
    # Test 1: Text message
    test1_result = test_text_message()
    
    # Test 2: Image message
    test2_result = test_image_message()
    
    # Test 3: Analysis message
    test3_result = test_analysis_message()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Text Message", test1_result),
        ("Image Message", test2_result),
        ("Analysis Message", test3_result)
    ]
    
    passed = sum(1 for _, result in tests if result is True)
    skipped = sum(1 for _, result in tests if result is None)
    failed = sum(1 for _, result in tests if result is False)
    
    for name, result in tests:
        status = "✅ PASSED" if result is True else ("⚠️  SKIPPED" if result is None else "❌ FAILED")
        print(f"{name:20} {status}")
    
    print()
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    print()
    
    if failed == 0 and passed > 0:
        print("🎉 All tests passed! Your Telegram integration is working!")
        print("   You can now receive notifications in Telegram.")
        return 0
    elif failed > 0:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("   See TELEGRAM_SETUP.md for troubleshooting tips.")
        return 1
    else:
        print("⚠️  Tests were skipped. Check the warnings above.")
        return 0

if __name__ == "__main__":
    sys.exit(main())

