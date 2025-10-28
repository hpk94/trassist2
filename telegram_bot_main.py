#!/usr/bin/env python3
"""Main launcher for the Telegram trading bot."""

import os
import sys
from dotenv import load_dotenv

# Add services directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

from services.telegram_bot import TradingTelegramBot

def main():
    """Main function to start the Telegram bot."""
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    required_vars = [
        'TELEGRAM_BOT_TOKEN',
        'OPENAI_API_KEY',
        'MEXC_API_KEY',
        'MEXC_API_SECRET'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all required variables are set.")
        sys.exit(1)
    
    # Optional variables with warnings
    optional_vars = {
        'TELEGRAM_AUTHORIZED_USERS': 'No authorized users configured - bot will accept requests from anyone',
        'PUSHOVER_TOKEN': 'Pushover notifications disabled',
        'EMAIL_USERNAME': 'Email notifications disabled'
    }
    
    for var, warning in optional_vars.items():
        if not os.getenv(var):
            print(f"⚠️ {warning}")
    
    print("🚀 Starting Telegram Trading Bot...")
    print("📱 Bot will handle chart analysis and trading signals")
    print("🔧 Press Ctrl+C to stop the bot")
    print()
    
    try:
        # Create and run the bot
        bot = TradingTelegramBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()