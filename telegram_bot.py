#!/usr/bin/env python3
"""
Telegram Bot for controlling trading analysis
Allows users to stop/start analysis via Telegram commands
"""

import os
import json
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

class AnalysisState:
    """Thread-safe singleton to track analysis state"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._running = False
                    cls._instance._current_symbol = None
                    cls._instance._current_direction = None
                    cls._instance._start_time = None
                    cls._instance._should_stop = False
                    cls._instance._analysis_data = {}
        return cls._instance
    
    @property
    def running(self) -> bool:
        with self._lock:
            return self._running
    
    @running.setter
    def running(self, value: bool):
        with self._lock:
            self._running = value
            if value:
                self._start_time = datetime.now()
                self._should_stop = False
            else:
                self._current_symbol = None
                self._current_direction = None
    
    @property
    def should_stop(self) -> bool:
        with self._lock:
            return self._should_stop
    
    def request_stop(self):
        """Request analysis to stop gracefully"""
        with self._lock:
            self._should_stop = True
    
    def set_analysis_info(self, symbol: str, direction: str):
        with self._lock:
            self._current_symbol = symbol
            self._current_direction = direction
    
    def get_status_info(self) -> Dict[str, Any]:
        with self._lock:
            if self._running:
                elapsed = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
                return {
                    "running": True,
                    "symbol": self._current_symbol,
                    "direction": self._current_direction,
                    "elapsed_seconds": elapsed,
                    "start_time": self._start_time.strftime("%Y-%m-%d %H:%M:%S") if self._start_time else None
                }
            return {"running": False}
    
    def update_data(self, key: str, value: Any):
        """Store additional analysis data"""
        with self._lock:
            self._analysis_data[key] = value
    
    def get_data(self, key: str) -> Any:
        """Retrieve analysis data"""
        with self._lock:
            return self._analysis_data.get(key)


class TelegramBot:
    """Telegram bot for controlling trading analysis"""
    
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.state = AnalysisState()
        self._running = False
        self._last_update_id = 0
        
        if not self.bot_token or not self.chat_id:
            print("⚠️  Telegram bot not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
    
    def send_message(self, text: str, parse_mode: str = "HTML", show_keyboard: bool = False) -> bool:
        """Send a message to the configured chat"""
        if not self.bot_token or not self.chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            
            # Add reply keyboard if requested
            if show_keyboard:
                keyboard = {
                    "keyboard": [
                        [{"text": "/status"}, {"text": "/stop"}, {"text": "/help"}]
                    ],
                    "resize_keyboard": True,
                    "persistent": True
                }
                data["reply_markup"] = json.dumps(keyboard)
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Failed to send Telegram message: {e}")
            return False
    
    def get_updates(self, offset: Optional[int] = None) -> list:
        """Get updates from Telegram"""
        if not self.bot_token:
            return []
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            params = {"timeout": 30, "allowed_updates": ["message"]}
            if offset:
                params["offset"] = offset
            
            response = requests.get(url, params=params, timeout=35)
            response.raise_for_status()
            result = response.json()
            
            if result.get("ok"):
                return result.get("result", [])
            return []
        except Exception as e:
            print(f"❌ Failed to get Telegram updates: {e}")
            return []
    
    def handle_command(self, command: str, message_text: str):
        """Handle bot commands"""
        
        if command == "/start" or command == "/help":
            help_text = """
<b>🤖 Trading Assistant Bot</b>

<b>Available Commands:</b>

/status - Check current analysis status
/stop - Stop the running analysis
/help - Show this help message

<b>How it works:</b>
• Upload a chart to start analysis
• Use /status to check progress
• Use /stop to cancel and start a new analysis

<b>💡 Tip:</b> Use the buttons below for quick access!
            """.strip()
            self.send_message(help_text, show_keyboard=True)
        
        elif command == "/status":
            status = self.state.get_status_info()
            
            if status["running"]:
                elapsed_min = int(status["elapsed_seconds"] / 60)
                elapsed_sec = int(status["elapsed_seconds"] % 60)
                
                status_text = f"""
<b>📊 Analysis Status: RUNNING</b>

<b>Symbol:</b> {status['symbol'] or 'Loading...'}
<b>Direction:</b> {status['direction'] or 'Analyzing...'}
<b>Start Time:</b> {status['start_time']}
<b>Elapsed Time:</b> {elapsed_min}m {elapsed_sec}s

<b>Status:</b> Analysis in progress...

Use /stop to cancel this analysis.
                """.strip()
            else:
                status_text = """
<b>📊 Analysis Status: IDLE</b>

No analysis is currently running.

Upload a chart to start a new analysis.
                """.strip()
            
            self.send_message(status_text)
        
        elif command == "/stop":
            if self.state.running:
                self.state.request_stop()
                stop_text = """
<b>⏹️ Stop Request Received</b>

The current analysis will stop gracefully at the next checkpoint.

You can upload a new chart once the analysis has stopped.
                """.strip()
                self.send_message(stop_text)
            else:
                self.send_message("<b>ℹ️ No Active Analysis</b>\n\nThere is no analysis currently running.")
        
        else:
            self.send_message(f"❓ Unknown command: {command}\n\nUse /help to see available commands.")
    
    def process_updates(self):
        """Process incoming updates"""
        updates = self.get_updates(self._last_update_id + 1 if self._last_update_id else None)
        
        for update in updates:
            self._last_update_id = update.get("update_id", 0)
            
            message = update.get("message", {})
            text = message.get("text", "")
            
            # Only process messages from the configured chat
            if str(message.get("chat", {}).get("id")) == str(self.chat_id):
                if text.startswith("/"):
                    command = text.split()[0].lower()
                    self.handle_command(command, text)
    
    def start_polling(self):
        """Start polling for commands in a background thread"""
        if self._running:
            print("⚠️  Bot polling already running")
            return
        
        if not self.bot_token or not self.chat_id:
            print("⚠️  Cannot start bot polling - Telegram not configured")
            return
        
        self._running = True
        
        def poll_loop():
            print("✅ Telegram bot polling started")
            while self._running:
                try:
                    self.process_updates()
                except Exception as e:
                    print(f"❌ Error in bot polling: {e}")
                time.sleep(1)  # Small delay between polls
            print("🛑 Telegram bot polling stopped")
        
        thread = threading.Thread(target=poll_loop, daemon=True)
        thread.start()
        
        # Send startup message with keyboard
        self.send_message("""
<b>🤖 Trading Assistant Bot Started</b>

The bot is now listening for commands.
Use /help to see available commands.

<b>💡 Use the buttons below for quick access!</b>
        """.strip(), show_keyboard=True)
    
    def stop_polling(self):
        """Stop polling for commands"""
        self._running = False


# Global bot instance
_bot_instance = None

def get_bot() -> TelegramBot:
    """Get the global bot instance"""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = TelegramBot()
    return _bot_instance

def start_bot():
    """Start the Telegram bot"""
    bot = get_bot()
    bot.start_polling()

def stop_bot():
    """Stop the Telegram bot"""
    bot = get_bot()
    bot.stop_polling()

# Convenience functions for integration with main app
def get_analysis_state() -> AnalysisState:
    """Get the global analysis state"""
    return AnalysisState()

def should_stop_analysis() -> bool:
    """Check if analysis should stop"""
    return get_analysis_state().should_stop

def set_analysis_running(running: bool):
    """Set analysis running state"""
    get_analysis_state().running = running

def set_analysis_info(symbol: str, direction: str):
    """Set current analysis info"""
    get_analysis_state().set_analysis_info(symbol, direction)

def send_telegram_status(message: str):
    """Send a status update via Telegram"""
    bot = get_bot()
    bot.send_message(message)


if __name__ == "__main__":
    # Test the bot
    print("Starting Telegram bot...")
    start_bot()
    
    try:
        print("Bot is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping bot...")
        stop_bot()

