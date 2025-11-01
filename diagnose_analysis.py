#!/usr/bin/env python3
"""
Diagnostic script to check why analysis notifications are not being sent
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def check_environment():
    """Check if all required environment variables are set"""
    print("=" * 60)
    print("ENVIRONMENT VARIABLES CHECK")
    print("=" * 60)
    
    required_vars = {
        "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
        "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
        "MEXC_API_KEY": os.getenv("MEXC_API_KEY"),
        "MEXC_API_SECRET": os.getenv("MEXC_API_SECRET"),
    }
    
    llm_vars = {
        "LITELLM_VISION_MODEL": os.getenv("LITELLM_VISION_MODEL", "gpt-4o"),
        "LITELLM_TEXT_MODEL": os.getenv("LITELLM_TEXT_MODEL", "gpt-4o"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }
    
    all_good = True
    for var, value in required_vars.items():
        status = "✅" if value else "❌"
        masked_value = f"{value[:10]}..." if value else "NOT SET"
        print(f"{status} {var}: {masked_value}")
        if not value:
            all_good = False
    
    print("\nLLM Configuration:")
    for var, value in llm_vars.items():
        status = "✅" if value else "❌"
        masked_value = f"{value[:10]}..." if value and "KEY" in var else value
        print(f"{status} {var}: {masked_value}")
    
    return all_good

def test_telegram():
    """Test Telegram connection"""
    print("\n" + "=" * 60)
    print("TELEGRAM CONNECTION TEST")
    print("=" * 60)
    
    try:
        from services.notification_service import NotificationService
        service = NotificationService()
        
        if not service.telegram_bot_token or not service.telegram_chat_id:
            print("❌ Telegram not configured in .env file")
            return False
        
        print(f"✅ Telegram Bot Token: {service.telegram_bot_token[:10]}...")
        print(f"✅ Telegram Chat ID: {service.telegram_chat_id}")
        
        # Test sending a message
        print("\n📱 Sending test message to Telegram...")
        test_data = {
            "symbol": "BTCUSDT",
            "direction": "long",
            "current_price": 67234.56,
            "confidence": 0.85,
            "current_rsi": 45.2,
            "stop_loss": 66000.00,
            "risk_reward": 2.5,
            "take_profits": [{"price": 68500.00}]
        }
        
        result = service.send_trade_notification(test_data, "valid_trade")
        
        if result.get("telegram"):
            print("✅ Test notification sent successfully!")
            return True
        else:
            print("❌ Failed to send test notification")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Telegram: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_llm():
    """Test LLM connection"""
    print("\n" + "=" * 60)
    print("LLM CONNECTION TEST")
    print("=" * 60)
    
    try:
        import litellm
        vision_model = os.getenv("LITELLM_VISION_MODEL", "gpt-4o")
        
        print(f"Testing {vision_model}...")
        
        response = litellm.completion(
            model=vision_model,
            messages=[{"role": "user", "content": "Say 'hello' in JSON format"}],
            response_format={"type": "json_object"},
            max_tokens=50
        )
        
        print("✅ LLM connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {str(e)}")
        return False

def test_mexc():
    """Test MEXC API connection"""
    print("\n" + "=" * 60)
    print("MEXC API CONNECTION TEST")
    print("=" * 60)
    
    try:
        from pymexc import spot
        public_client = spot.HTTP()
        
        print("Fetching BTCUSDT klines...")
        klines = public_client.klines(symbol="BTCUSDT", interval="1m", limit=10)
        
        if klines and len(klines) > 0:
            print(f"✅ Successfully fetched {len(klines)} klines")
            return True
        else:
            print("❌ No klines data returned")
            return False
            
    except Exception as e:
        print(f"❌ MEXC API test failed: {str(e)}")
        return False

def check_recent_uploads():
    """Check for recent uploads"""
    print("\n" + "=" * 60)
    print("RECENT UPLOADS")
    print("=" * 60)
    
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print("❌ No uploads directory found")
        return
    
    files = []
    for filename in os.listdir(uploads_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(uploads_dir, filename)
            mtime = os.path.getmtime(filepath)
            files.append((filename, mtime))
    
    files.sort(key=lambda x: x[1], reverse=True)
    
    if files:
        print(f"Found {len(files)} uploaded images:\n")
        for filename, mtime in files[:5]:
            dt = datetime.fromtimestamp(mtime)
            print(f"  📊 {filename}")
            print(f"     Uploaded: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("No uploaded images found")

def check_recent_analysis():
    """Check for recent analysis outputs"""
    print("\n" + "=" * 60)
    print("RECENT ANALYSIS OUTPUTS")
    print("=" * 60)
    
    output_dir = "llm_outputs"
    if not os.path.exists(output_dir):
        print("❌ No llm_outputs directory found")
        return
    
    files = []
    for filename in os.listdir(output_dir):
        if filename.startswith("llm_output_") and filename.endswith(".json"):
            filepath = os.path.join(output_dir, filename)
            mtime = os.path.getmtime(filepath)
            files.append((filename, filepath, mtime))
    
    files.sort(key=lambda x: x[2], reverse=True)
    
    if files:
        print(f"Found {len(files)} analysis outputs:\n")
        for filename, filepath, mtime in files[:5]:
            dt = datetime.fromtimestamp(mtime)
            print(f"  📈 {filename}")
            print(f"     Created: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Try to read and show symbol
            try:
                import json
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    symbol = data.get('symbol', 'Unknown')
                    timeframe = data.get('timeframe', 'Unknown')
                    print(f"     Symbol: {symbol}, Timeframe: {timeframe}")
            except Exception:
                pass
    else:
        print("No analysis outputs found")

def main():
    """Run all diagnostics"""
    print("\n")
    print("🔍 Trading Assistant Diagnostics")
    print("=" * 60)
    print()
    
    # Run all checks
    env_ok = check_environment()
    telegram_ok = test_telegram()
    mexc_ok = test_mexc()
    llm_ok = test_llm()
    
    # Check files
    check_recent_uploads()
    check_recent_analysis()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_ok = env_ok and telegram_ok and mexc_ok and llm_ok
    
    if all_ok:
        print("✅ All systems operational!")
        print("\nIf you're not receiving analysis notifications:")
        print("1. Analysis may still be running (check server logs)")
        print("2. Analysis may take several minutes due to polling")
        print("3. Check server is running: ps aux | grep web_app")
    else:
        print("❌ Some systems are not working properly")
        print("\nPlease fix the issues above before testing again")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

