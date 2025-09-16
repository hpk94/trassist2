#!/usr/bin/env python3
"""Test script to validate Telegram bot setup and dependencies."""

import os
import sys
from dotenv import load_dotenv

def test_environment_variables():
    """Test that all required environment variables are set."""
    print("🔧 Testing environment variables...")
    
    load_dotenv()
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key',
        'MEXC_API_KEY': 'MEXC API key', 
        'MEXC_API_SECRET': 'MEXC API secret',
        'TELEGRAM_BOT_TOKEN': 'Telegram bot token'
    }
    
    optional_vars = {
        'TELEGRAM_AUTHORIZED_USERS': 'Authorized user IDs',
        'PUSHOVER_TOKEN': 'Pushover token',
        'EMAIL_USERNAME': 'Email username'
    }
    
    missing_required = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {description}: {'*' * 10}{value[-4:] if len(value) > 4 else value}")
        else:
            print(f"❌ {description}: Missing")
            missing_required.append(var)
    
    print("\n🔧 Optional configurations:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {description}: Configured")
        else:
            print(f"⚠️ {description}: Not configured")
    
    return len(missing_required) == 0, missing_required

def test_imports():
    """Test that all required packages can be imported."""
    print("\n📦 Testing package imports...")
    
    test_packages = [
        ('telegram', 'python-telegram-bot'),
        ('openai', 'openai'),
        ('pandas', 'pandas'),
        ('requests', 'requests'),
        ('dotenv', 'python-dotenv'),
        ('sqlite3', 'sqlite3 (built-in)'),
    ]
    
    failed_imports = []
    for package, description in test_packages:
        try:
            __import__(package)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0, failed_imports

def test_services():
    """Test that custom services can be imported."""
    print("\n🔧 Testing custom services...")
    
    services = [
        'services.trading_service',
        'services.telegram_bot',
        'services.status_service',
        'services.notification_service',
        'services.indicator_service'
    ]
    
    failed_services = []
    for service in services:
        try:
            __import__(service)
            print(f"✅ {service}")
        except ImportError as e:
            print(f"❌ {service}: {e}")
            failed_services.append(service)
    
    return len(failed_services) == 0, failed_services

def test_database():
    """Test database connectivity."""
    print("\n💾 Testing database setup...")
    
    try:
        from services.status_service import StatusService
        status_service = StatusService()
        print("✅ Database connection successful")
        print("✅ Database tables created")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection."""
    print("\n🤖 Testing OpenAI connection...")
    
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        if response.choices:
            print("✅ OpenAI API connection successful")
            return True
        else:
            print("❌ OpenAI API returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Telegram Trading Bot Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test environment variables
    env_passed, missing_vars = test_environment_variables()
    if not env_passed:
        print(f"\n❌ Missing required environment variables: {', '.join(missing_vars)}")
        all_passed = False
    
    # Test imports
    imports_passed, failed_imports = test_imports()
    if not imports_passed:
        print(f"\n❌ Failed to import packages: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        all_passed = False
    
    # Test services (only if imports passed)
    if imports_passed:
        services_passed, failed_services = test_services()
        if not services_passed:
            print(f"\n❌ Failed to import services: {', '.join(failed_services)}")
            all_passed = False
    
        # Test database
        db_passed = test_database()
        if not db_passed:
            all_passed = False
    
        # Test OpenAI (only if environment is set)
        if env_passed and os.getenv('OPENAI_API_KEY'):
            openai_passed = test_openai_connection()
            if not openai_passed:
                all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your Telegram bot is ready to run.")
        print("Start the bot with: python3 telegram_bot_main.py")
    else:
        print("❌ Some tests failed. Please fix the issues above before running the bot.")
        sys.exit(1)

if __name__ == "__main__":
    main()