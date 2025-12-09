#!/usr/bin/env python3
"""
Telegram Bot for controlling trading analysis
Allows users to stop/start analysis via Telegram commands
Supports inline buttons for trade management
"""

import os
import json
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# ============================================================================
# Callback Handlers for Inline Buttons
# ============================================================================

# Registry for callback handlers
_callback_handlers: Dict[str, Callable] = {}

def register_callback_handler(action: str, handler: Callable):
    """Register a handler for a callback action"""
    _callback_handlers[action] = handler

def get_callback_handler(action: str) -> Optional[Callable]:
    """Get a registered callback handler"""
    return _callback_handlers.get(action)


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
    """Telegram bot for controlling trading analysis with inline button support"""
    
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.state = AnalysisState()
        self._running = False
        self._last_update_id = 0
        self._active_trades: Dict[str, Dict[str, Any]] = {}  # Track active trades for button callbacks
        
        if not self.bot_token or not self.chat_id:
            print("⚠️  Telegram bot not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
    
    def send_message(self, text: str, parse_mode: str = "HTML", show_keyboard: bool = False,
                     inline_keyboard: Optional[List[List[Dict[str, str]]]] = None) -> Optional[Dict]:
        """
        Send a message to the configured chat.
        
        Args:
            text: Message text
            parse_mode: HTML or Markdown
            show_keyboard: Show persistent reply keyboard
            inline_keyboard: List of rows of inline buttons. Each button is a dict with 'text' and 'callback_data'
        
        Returns:
            Message response dict or None on failure
        """
        if not self.bot_token or not self.chat_id:
            return None
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            
            # Add inline keyboard if provided
            if inline_keyboard:
                data["reply_markup"] = json.dumps({
                    "inline_keyboard": inline_keyboard
                })
            # Add reply keyboard if requested
            elif show_keyboard:
                keyboard = {
                    "keyboard": [
                        [{"text": "/status"}, {"text": "/balance"}, {"text": "/positions"}],
                        [{"text": "/sltp"}, {"text": "/close"}, {"text": "/stop"}],
                        [{"text": "/help"}]
                    ],
                    "resize_keyboard": True,
                    "persistent": True
                }
                data["reply_markup"] = json.dumps(keyboard)
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            return response.json().get("result")
        except Exception as e:
            print(f"❌ Failed to send Telegram message: {e}")
            return None
    
    def send_trade_notification(self, trade_data: Dict[str, Any], trade_id: str) -> Optional[Dict]:
        """
        Send a trade notification with management buttons.
        
        Args:
            trade_data: Trade information (symbol, direction, entry_price, etc.)
            trade_id: Unique identifier for this trade (used in callbacks)
        
        Returns:
            Message response or None
        """
        # Store trade data for callback handling
        self._active_trades[trade_id] = trade_data
        
        symbol = trade_data.get("symbol", "Unknown")
        coin = trade_data.get("coin", symbol)
        direction = trade_data.get("direction", "Unknown").upper()
        entry_price = trade_data.get("entry_price", 0)
        size = trade_data.get("size", 0)
        leverage = trade_data.get("leverage", 1)
        
        direction_emoji = "🟢" if direction == "LONG" else "🔴"
        
        message = f"""
<b>🚀 TRADE OPENED</b>

{direction_emoji} <b>{coin}</b> {direction}
━━━━━━━━━━━━━━━━━━━━

<b>Entry Price:</b> ${entry_price:,.2f}
<b>Size:</b> {size:.6f} {coin}
<b>Leverage:</b> {leverage}x

<i>Use buttons below to manage this trade</i>
        """.strip()
        
        # Create inline keyboard with trade management buttons
        inline_keyboard = [
            [
                {"text": "📊 Check P&L", "callback_data": f"pnl:{trade_id}"},
                {"text": "📈 Prices", "callback_data": f"prices:{trade_id}"},
                {"text": "🎯 SL/TP", "callback_data": f"sltp:{trade_id}"}
            ],
            [
                {"text": "🔒 Close 50%", "callback_data": f"close50:{trade_id}"},
                {"text": "🔒 Close 100%", "callback_data": f"close100:{trade_id}"}
            ],
            [
                {"text": "🚨 Emergency Close (Market)", "callback_data": f"emergency:{trade_id}"}
            ]
        ]
        
        return self.send_message(message, inline_keyboard=inline_keyboard)
    
    def update_message(self, message_id: int, text: str, parse_mode: str = "HTML",
                      inline_keyboard: Optional[List[List[Dict[str, str]]]] = None) -> bool:
        """Update an existing message"""
        if not self.bot_token or not self.chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/editMessageText"
            data = {
                "chat_id": self.chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": parse_mode
            }
            
            if inline_keyboard:
                data["reply_markup"] = json.dumps({
                    "inline_keyboard": inline_keyboard
                })
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Failed to update Telegram message: {e}")
            return False
    
    def answer_callback(self, callback_query_id: str, text: str = "", show_alert: bool = False) -> bool:
        """Answer a callback query (acknowledges button press)"""
        if not self.bot_token:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery"
            data = {
                "callback_query_id": callback_query_id,
                "text": text,
                "show_alert": show_alert
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Failed to answer callback: {e}")
            return False
    
    def get_updates(self, offset: Optional[int] = None) -> list:
        """Get updates from Telegram (messages and callback queries)"""
        if not self.bot_token:
            return []
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            params = {"timeout": 30, "allowed_updates": ["message", "callback_query"]}
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
/balance - View wallet balance
/positions - View open positions
/sltp - View SL/TP levels for tracked positions
/close - Close all positions
/stop - Stop the running analysis
/help - Show this help message

<b>How it works:</b>
• Upload a chart to start analysis
• Use /status to check progress
• Use /stop to cancel and start a new analysis
• Use inline buttons to manage trades

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
        
        elif command == "/positions":
            self._show_positions()
        
        elif command == "/sltp":
            self._show_sltp_levels()
        
        elif command == "/balance":
            self._show_balance()
        
        elif command == "/close":
            self._close_all_positions()
        
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
    
    def _show_balance(self):
        """Show wallet balance"""
        try:
            from services.hyperliquid_service import get_account_info, HyperliquidConfig
            
            account = get_account_info()
            
            if not account:
                self.send_message("❌ Could not fetch account info")
                return
            
            network = "TESTNET" if HyperliquidConfig.is_testnet() else "MAINNET"
            leverage = HyperliquidConfig.get_leverage()
            
            # Calculate buying power
            buying_power = account['account_value'] * leverage
            
            msg = "<b>💰 Wallet Balance</b>\n\n"
            msg += f"<b>Network:</b> {network}\n"
            msg += f"━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"<b>Account Value:</b> ${account['account_value']:,.2f}\n"
            msg += f"<b>Margin Used:</b> ${account['total_margin_used']:,.2f}\n"
            msg += f"<b>Available:</b> ${account['withdrawable']:,.2f}\n"
            msg += f"━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"<b>Leverage:</b> {leverage}x\n"
            msg += f"<b>Buying Power:</b> ${buying_power:,.2f}\n"
            
            if account.get('wallet_address'):
                short_addr = f"{account['wallet_address'][:6]}...{account['wallet_address'][-4:]}"
                msg += f"\n<b>Wallet:</b> <code>{short_addr}</code>"
            
            # Add refresh button
            inline_keyboard = [
                [{"text": "🔄 Refresh", "callback_data": "refresh:balance"}]
            ]
            
            self.send_message(msg, inline_keyboard=inline_keyboard)
            
        except ImportError:
            self.send_message("❌ Hyperliquid service not available")
        except Exception as e:
            self.send_message(f"❌ Error: {e}")
    
    def _show_positions(self):
        """Show current open positions (optimized for speed)"""
        try:
            # Use fast method that gets positions + account in single API call
            from services.hyperliquid_service import get_positions_fast
            
            result = get_positions_fast()
            
            if result.get("error"):
                self.send_message(f"❌ Error: {result['error']}")
                return
            
            positions = result.get("positions", [])
            account = result.get("account")
            
            if not positions:
                self.send_message("<b>📍 No Open Positions</b>\n\nYou have no active trades on Hyperliquid.")
                return
            
            message = "<b>📍 Open Positions</b>\n\n"
            
            for pos in positions:
                direction_emoji = "🟢" if pos["direction"] == "LONG" else "🔴"
                pnl_emoji = "📈" if pos["unrealized_pnl"] >= 0 else "📉"
                
                # Calculate P&L percentage (leverage-adjusted = ROI on margin)
                position_value = pos["entry_price"] * pos["size"]
                leverage = pos.get("leverage", 1) or 1
                margin = position_value / leverage
                pnl_pct = (pos["unrealized_pnl"] / margin * 100) if margin > 0 else 0
                
                message += f"{direction_emoji} <b>{pos['coin']}</b> {pos['direction']}\n"
                message += f"   Size: {pos['size']:.6f}\n"
                message += f"   Entry: ${pos['entry_price']:,.2f}\n"
                message += f"   Leverage: {pos['leverage']}x\n"
                message += f"   {pnl_emoji} P&L: ${pos['unrealized_pnl']:.2f} ({pnl_pct:+.2f}%)\n\n"
            
            # Add account info (already fetched, no extra API call)
            if account:
                message += f"━━━━━━━━━━━━━━━━━━━━\n"
                message += f"<b>💰 Account:</b> ${account['account_value']:,.2f}\n"
                message += f"<b>Margin Used:</b> ${account['total_margin_used']:,.2f}"
            
            # Add close buttons for each position
            inline_keyboard = []
            for pos in positions:
                inline_keyboard.append([
                    {"text": f"🔒 Close {pos['coin']}", "callback_data": f"closepos:{pos['coin']}"}
                ])
            # Add refresh button
            inline_keyboard.append([
                {"text": "🔄 Refresh", "callback_data": "refresh:positions"}
            ])
            
            self.send_message(message, inline_keyboard=inline_keyboard)
            
        except ImportError:
            self.send_message("❌ Hyperliquid service not available")
        except Exception as e:
            self.send_message(f"❌ Error getting positions: {e}")
    
    def _show_sltp_levels(self):
        """Show SL/TP levels for all tracked positions"""
        try:
            from services.hyperliquid_service import get_position_monitor, get_mexc_price, get_open_positions
            
            monitor = get_position_monitor()
            tracked = monitor.get_tracked_positions()
            
            if not tracked:
                # Check if there are positions but not tracked
                positions = get_open_positions()
                if positions:
                    msg = "<b>🎯 SL/TP Levels</b>\n\n"
                    msg += "⚠️ <i>No SL/TP tracking active for current positions.</i>\n\n"
                    msg += "Positions without SL/TP tracking:\n"
                    for pos in positions:
                        msg += f"• {pos['coin']} ({pos['direction']})\n"
                    msg += "\n<i>SL/TP tracking is set up automatically when trades are opened via the trading assistant.</i>"
                else:
                    msg = "<b>🎯 SL/TP Levels</b>\n\n<i>No tracked positions.</i>"
                self.send_message(msg)
                return
            
            msg = "<b>🎯 SL/TP Levels</b>\n\n"
            
            for coin, data in tracked.items():
                direction = data.get("direction", "UNKNOWN")
                entry = data.get("entry_price", 0)
                stop_loss = data.get("stop_loss")
                take_profit = data.get("take_profit")
                
                direction_emoji = "🟢" if direction == "LONG" else "🔴"
                
                # Get current MEXC price
                mexc_price = get_mexc_price(f"{coin}USDT")
                
                msg += f"{direction_emoji} <b>{coin}</b> {direction}\n"
                msg += f"   <b>Entry:</b> ${entry:,.2f}\n"
                
                if mexc_price:
                    # Calculate current P&L
                    if direction == "LONG":
                        pnl_pct = ((mexc_price - entry) / entry) * 100
                    else:
                        pnl_pct = ((entry - mexc_price) / entry) * 100
                    pnl_emoji = "📈" if pnl_pct >= 0 else "📉"
                    msg += f"   <b>Current:</b> ${mexc_price:,.2f} ({pnl_emoji} {pnl_pct:+.2f}%)\n"
                
                if stop_loss:
                    sl_dist = abs((stop_loss - entry) / entry * 100)
                    if mexc_price:
                        if direction == "LONG":
                            sl_pct_away = ((mexc_price - stop_loss) / mexc_price * 100)
                        else:
                            sl_pct_away = ((stop_loss - mexc_price) / mexc_price * 100)
                        msg += f"   🛡️ <b>SL:</b> ${stop_loss:,.2f} ({sl_pct_away:.2f}% away)\n"
                    else:
                        msg += f"   🛡️ <b>SL:</b> ${stop_loss:,.2f} ({sl_dist:.2f}% from entry)\n"
                else:
                    msg += f"   🛡️ <b>SL:</b> Not set\n"
                
                if take_profit:
                    tp_dist = abs((take_profit - entry) / entry * 100)
                    if mexc_price:
                        if direction == "LONG":
                            tp_pct_away = ((take_profit - mexc_price) / mexc_price * 100)
                        else:
                            tp_pct_away = ((mexc_price - take_profit) / mexc_price * 100)
                        msg += f"   🎯 <b>TP:</b> ${take_profit:,.2f} ({tp_pct_away:.2f}% away)\n"
                    else:
                        msg += f"   🎯 <b>TP:</b> ${take_profit:,.2f} ({tp_dist:.2f}% from entry)\n"
                else:
                    msg += f"   🎯 <b>TP:</b> Not set\n"
                
                # R:R ratio
                if stop_loss and take_profit and entry:
                    if direction == "LONG":
                        risk = entry - stop_loss
                        reward = take_profit - entry
                    else:
                        risk = stop_loss - entry
                        reward = entry - take_profit
                    if risk > 0:
                        rr = reward / risk
                        msg += f"   📊 <b>R:R:</b> 1:{rr:.2f}\n"
                
                msg += "\n"
            
            # Monitor status
            if monitor.is_running():
                msg += "━━━━━━━━━━━━━━━━━━━━\n"
                msg += "✅ <b>Auto SL/TP monitoring:</b> Active\n"
                msg += "<i>Positions will auto-close when SL/TP hit on MEXC</i>"
            else:
                msg += "━━━━━━━━━━━━━━━━━━━━\n"
                msg += "⚠️ <b>Auto SL/TP monitoring:</b> Inactive"
            
            # Add refresh button
            inline_keyboard = [
                [{"text": "🔄 Refresh", "callback_data": "refresh:sltp"}]
            ]
            
            self.send_message(msg, inline_keyboard=inline_keyboard)
            
        except ImportError:
            self.send_message("❌ Hyperliquid service not available")
        except Exception as e:
            self.send_message(f"❌ Error getting SL/TP levels: {e}")
    
    def _close_all_positions(self):
        """Close all open positions"""
        try:
            from services.hyperliquid_service import get_open_positions, close_position_limit
            
            positions = get_open_positions()
            
            if not positions:
                self.send_message("<b>ℹ️ No Positions to Close</b>\n\nYou have no active trades.")
                return
            
            # Confirm before closing
            message = f"<b>⚠️ Close All Positions?</b>\n\nThis will close {len(positions)} position(s):\n\n"
            
            for pos in positions:
                direction_emoji = "🟢" if pos["direction"] == "LONG" else "🔴"
                message += f"{direction_emoji} {pos['coin']} {pos['direction']} ({pos['size']:.6f})\n"
            
            inline_keyboard = [
                [
                    {"text": "✅ Yes, Close All", "callback_data": "closeall:confirm"},
                    {"text": "❌ Cancel", "callback_data": "closeall:cancel"}
                ]
            ]
            
            self.send_message(message, inline_keyboard=inline_keyboard)
            
        except ImportError:
            self.send_message("❌ Hyperliquid service not available")
        except Exception as e:
            self.send_message(f"❌ Error: {e}")
    
    def handle_callback(self, callback_query: Dict[str, Any]):
        """Handle inline button callback"""
        callback_id = callback_query.get("id")
        data = callback_query.get("data", "")
        message = callback_query.get("message", {})
        message_id = message.get("message_id")
        
        # Parse callback data (format: action:param)
        parts = data.split(":", 1)
        action = parts[0]
        param = parts[1] if len(parts) > 1 else ""
        
        try:
            from services.hyperliquid_service import (
                get_position, get_open_positions, close_position_limit, 
                close_position_market, get_price_difference, get_hyperliquid_price
            )
            
            if action == "pnl":
                # Show P&L for a specific trade
                trade_data = self._active_trades.get(param, {})
                coin = trade_data.get("coin", param)
                position = get_position(coin)
                
                if position:
                    pnl = position["unrealized_pnl"]
                    position_value = position["entry_price"] * position["size"]
                    leverage = position.get("leverage", 1) or 1
                    margin = position_value / leverage
                    pnl_pct = (pnl / margin * 100) if margin > 0 else 0
                    text = f"{'📈' if pnl >= 0 else '📉'} {coin}: ${pnl:.2f} ({pnl_pct:+.2f}%)"
                else:
                    text = f"No position found for {coin}"
                
                self.answer_callback(callback_id, text)
            
            elif action == "prices":
                # Show price comparison
                trade_data = self._active_trades.get(param, {})
                coin = trade_data.get("coin", param)
                diff = get_price_difference(coin)
                
                if diff.get("hyperliquid_price") and diff.get("mexc_price"):
                    text = f"HL: ${diff['hyperliquid_price']:,.2f} | MEXC: ${diff['mexc_price']:,.2f} ({diff['difference_pct']:+.3f}%)"
                else:
                    text = "Could not fetch prices"
                
                self.answer_callback(callback_id, text)
            
            elif action == "sltp":
                # Show current SL/TP levels
                from services.hyperliquid_service import get_position_monitor, get_mexc_price
                
                trade_data = self._active_trades.get(param, {})
                coin = trade_data.get("coin", param)
                
                monitor = get_position_monitor()
                tracked = monitor.get_tracked_positions().get(coin, {})
                
                if tracked:
                    stop_loss = tracked.get("stop_loss")
                    take_profit = tracked.get("take_profit")
                    entry = tracked.get("entry_price", 0)
                    direction = tracked.get("direction", "UNKNOWN")
                    
                    # Get current MEXC price for distance calculation
                    mexc_price = get_mexc_price(f"{coin}USDT")
                    
                    msg = f"<b>🎯 SL/TP Levels for {coin}</b>\n\n"
                    msg += f"<b>Direction:</b> {direction}\n"
                    msg += f"<b>Entry:</b> ${entry:,.2f}\n"
                    
                    if mexc_price:
                        msg += f"<b>Current (MEXC):</b> ${mexc_price:,.2f}\n"
                    
                    msg += "\n"
                    
                    if stop_loss:
                        sl_distance_pct = abs((stop_loss - entry) / entry * 100)
                        sl_status = "🔴" if (direction == "LONG" and mexc_price and mexc_price <= stop_loss) or \
                                           (direction == "SHORT" and mexc_price and mexc_price >= stop_loss) else "⚪"
                        msg += f"{sl_status} <b>Stop Loss:</b> ${stop_loss:,.2f} ({sl_distance_pct:.2f}% from entry)\n"
                        if mexc_price:
                            if direction == "LONG":
                                sl_current_dist = ((mexc_price - stop_loss) / stop_loss * 100)
                                msg += f"   └ {sl_current_dist:.2f}% above SL\n"
                            else:
                                sl_current_dist = ((stop_loss - mexc_price) / mexc_price * 100)
                                msg += f"   └ {sl_current_dist:.2f}% below SL\n"
                    else:
                        msg += "⚪ <b>Stop Loss:</b> Not set\n"
                    
                    if take_profit:
                        tp_distance_pct = abs((take_profit - entry) / entry * 100)
                        tp_status = "🟢" if (direction == "LONG" and mexc_price and mexc_price >= take_profit) or \
                                           (direction == "SHORT" and mexc_price and mexc_price <= take_profit) else "⚪"
                        msg += f"{tp_status} <b>Take Profit:</b> ${take_profit:,.2f} ({tp_distance_pct:.2f}% from entry)\n"
                        if mexc_price:
                            if direction == "LONG":
                                tp_current_dist = ((take_profit - mexc_price) / mexc_price * 100)
                                msg += f"   └ {tp_current_dist:.2f}% to TP\n"
                            else:
                                tp_current_dist = ((mexc_price - take_profit) / take_profit * 100)
                                msg += f"   └ {tp_current_dist:.2f}% to TP\n"
                    else:
                        msg += "⚪ <b>Take Profit:</b> Not set\n"
                    
                    # Calculate R:R if both set
                    if stop_loss and take_profit and entry:
                        if direction == "LONG":
                            risk = entry - stop_loss
                            reward = take_profit - entry
                        else:
                            risk = stop_loss - entry
                            reward = entry - take_profit
                        
                        if risk > 0:
                            rr_ratio = reward / risk
                            msg += f"\n<b>Risk:Reward:</b> 1:{rr_ratio:.2f}"
                    
                    # Show if monitoring is active
                    if monitor.is_running():
                        msg += f"\n\n✅ <i>Auto SL/TP monitoring active</i>"
                    else:
                        msg += f"\n\n⚠️ <i>Monitor not running</i>"
                    
                    self.send_message(msg)
                    self.answer_callback(callback_id, "SL/TP levels shown below ⬇️")
                else:
                    self.answer_callback(callback_id, f"No SL/TP tracking for {coin}", show_alert=True)
            
            elif action == "close50":
                # Close 50% of position
                trade_data = self._active_trades.get(param, {})
                coin = trade_data.get("coin", param)
                position = get_position(coin)
                
                if position:
                    size_to_close = position["size"] * 0.5
                    result = close_position_limit(coin, size=size_to_close)
                    
                    if result.get("ok"):
                        self.answer_callback(callback_id, f"✅ Closing 50% of {coin} position...")
                        self.send_message(f"<b>🔒 Partial Close Order Placed</b>\n\nClosing 50% ({size_to_close:.6f}) of {coin} position via limit order.")
                    else:
                        self.answer_callback(callback_id, f"❌ Failed: {result.get('error')}", show_alert=True)
                else:
                    self.answer_callback(callback_id, f"No position for {coin}", show_alert=True)
            
            elif action == "close100":
                # Close 100% of position (limit order)
                trade_data = self._active_trades.get(param, {})
                coin = trade_data.get("coin", param)
                
                print(f"📤 Telegram: Close 100% for {coin} (trade_id: {param})")
                result = close_position_limit(coin)
                print(f"📤 Close result: {result}")
                
                if result.get("ok"):
                    close_details = result.get("close_details", {})
                    self.answer_callback(callback_id, f"✅ Closing {coin} position...")
                    
                    # Calculate P&L percentage (leverage-adjusted = ROI on margin)
                    entry = close_details.get('entry_price', 0)
                    size = close_details.get('size', 0)
                    pnl = close_details.get('unrealized_pnl', 0)
                    leverage = close_details.get('leverage', 1) or 1
                    position_value = entry * size
                    margin = position_value / leverage
                    pnl_pct = (pnl / margin * 100) if margin > 0 else 0
                    pnl_emoji = "📈" if pnl >= 0 else "📉"
                    
                    msg = f"<b>🔒 Close Order Placed</b>\n\n"
                    msg += f"<b>Coin:</b> {coin}\n"
                    msg += f"<b>Size:</b> {size:.6f}\n"
                    msg += f"<b>Limit Price:</b> ${close_details.get('limit_price', 0):,.2f}\n"
                    msg += f"<b>Entry was:</b> ${entry:,.2f}\n"
                    msg += f"<b>{pnl_emoji} P&L:</b> ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
                    
                    if result.get("filled"):
                        msg += "\n✅ <b>Filled immediately!</b>"
                    
                    self.send_message(msg)
                    
                    # Remove from active trades
                    if param in self._active_trades:
                        del self._active_trades[param]
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"❌ Close failed: {error}")
                    self.answer_callback(callback_id, f"❌ {error}", show_alert=True)
                    self.send_message(f"<b>❌ Close Failed</b>\n\n{coin}: {error}")
            
            elif action == "emergency":
                # Emergency market close
                trade_data = self._active_trades.get(param, {})
                coin = trade_data.get("coin", param)
                
                print(f"🚨 Telegram: Emergency close for {coin}")
                self.answer_callback(callback_id, "⚠️ Executing emergency market close...")
                
                result = close_position_market(coin)
                print(f"🚨 Emergency close result: {result}")
                
                if result.get("ok"):
                    close_details = result.get("close_details", {})
                    msg = f"<b>🚨 Emergency Close Executed</b>\n\n"
                    msg += f"<b>Coin:</b> {coin}\n"
                    msg += f"<b>Size:</b> {close_details.get('size', 0):.6f}\n"
                    msg += f"<b>Method:</b> Market Order\n"
                    
                    if result.get("filled"):
                        msg += "\n✅ <b>Position closed!</b>"
                    
                    self.send_message(msg)
                    if param in self._active_trades:
                        del self._active_trades[param]
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"❌ Emergency close failed: {error}")
                    self.send_message(f"<b>❌ Emergency Close Failed</b>\n\n{coin}: {error}")
            
            elif action == "closepos":
                # Close a specific position from /positions view
                coin = param
                print(f"📤 Telegram: Closing position for {coin}")
                
                result = close_position_limit(coin)
                print(f"📤 Close result: {result}")
                
                if result.get("ok"):
                    close_details = result.get("close_details", {})
                    self.answer_callback(callback_id, f"✅ Closing {coin}...")
                    
                    msg = f"<b>🔒 Close Order Placed</b>\n\n"
                    msg += f"<b>Coin:</b> {coin}\n"
                    msg += f"<b>Direction:</b> {close_details.get('direction', 'N/A')}\n"
                    msg += f"<b>Size:</b> {close_details.get('size', 0):.6f}\n"
                    msg += f"<b>Limit Price:</b> ${close_details.get('limit_price', 0):,.2f}\n"
                    
                    if result.get("filled"):
                        msg += "\n<b>Status:</b> ✅ Filled immediately"
                    elif result.get("order_id"):
                        msg += f"\n<b>Status:</b> ⏳ Resting (OID: {result['order_id']})"
                    
                    self.send_message(msg)
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"❌ Close failed: {error}")
                    self.answer_callback(callback_id, f"❌ {error}", show_alert=True)
                    self.send_message(f"<b>❌ Close Failed</b>\n\n{coin}: {error}")
            
            elif action == "refresh":
                # Handle refresh callbacks
                if param == "positions":
                    self.answer_callback(callback_id, "Refreshing...")
                    self._show_positions()
                elif param == "balance":
                    self.answer_callback(callback_id, "Refreshing...")
                    self._show_balance()
                elif param == "sltp":
                    self.answer_callback(callback_id, "Refreshing...")
                    self._show_sltp_levels()
                else:
                    self.answer_callback(callback_id, "Refreshed")
            
            elif action == "closeall":
                if param == "confirm":
                    positions = get_open_positions()
                    results = []
                    
                    for pos in positions:
                        result = close_position_limit(pos["coin"])
                        results.append({
                            "coin": pos["coin"],
                            "ok": result.get("ok"),
                            "error": result.get("error")
                        })
                    
                    success = sum(1 for r in results if r["ok"])
                    self.answer_callback(callback_id, f"Closing {success}/{len(positions)} positions...")
                    
                    message = "<b>🔒 Closing All Positions</b>\n\n"
                    for r in results:
                        emoji = "✅" if r["ok"] else "❌"
                        message += f"{emoji} {r['coin']}: {'Order placed' if r['ok'] else r['error']}\n"
                    
                    self.send_message(message)
                else:
                    self.answer_callback(callback_id, "Cancelled")
            
            else:
                # Check for registered custom handlers
                handler = get_callback_handler(action)
                if handler:
                    handler(param, callback_query, self)
                else:
                    self.answer_callback(callback_id, f"Unknown action: {action}")
                    
        except ImportError as e:
            self.answer_callback(callback_id, "Hyperliquid service not available", show_alert=True)
        except Exception as e:
            self.answer_callback(callback_id, f"Error: {str(e)[:50]}", show_alert=True)
            print(f"❌ Callback error: {e}")
    
    def process_updates(self):
        """Process incoming updates (messages and callback queries)"""
        updates = self.get_updates(self._last_update_id + 1 if self._last_update_id else None)
        
        for update in updates:
            self._last_update_id = update.get("update_id", 0)
            
            # Handle callback queries (inline button presses)
            callback_query = update.get("callback_query")
            if callback_query:
                # Verify it's from the correct chat
                callback_chat_id = str(callback_query.get("message", {}).get("chat", {}).get("id"))
                if callback_chat_id == str(self.chat_id):
                    self.handle_callback(callback_query)
                continue
            
            # Handle regular messages
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

def send_trade_with_buttons(trade_data: Dict[str, Any], trade_id: str) -> bool:
    """
    Send a trade notification with management buttons.
    
    Args:
        trade_data: Trade information including:
            - symbol: Trading symbol
            - coin: Hyperliquid coin name
            - direction: LONG or SHORT
            - entry_price: Entry price
            - size: Position size
            - leverage: Leverage used
        trade_id: Unique ID for this trade (used for button callbacks)
    
    Returns:
        True if message sent successfully
    """
    bot = get_bot()
    result = bot.send_trade_notification(trade_data, trade_id)
    return result is not None

def send_position_closed(coin: str, pnl: float, method: str = "limit"):
    """Send notification that a position was closed"""
    bot = get_bot()
    pnl_emoji = "📈" if pnl >= 0 else "📉"
    
    message = f"""
<b>🔒 Position Closed</b>

<b>Coin:</b> {coin}
<b>Method:</b> {method.upper()} order
<b>{pnl_emoji} Realized P&L:</b> ${pnl:.2f}
    """.strip()
    
    bot.send_message(message)

def send_custom_buttons(message: str, buttons: List[List[Dict[str, str]]]) -> bool:
    """
    Send a message with custom inline buttons.
    
    Args:
        message: Message text (HTML)
        buttons: List of rows of buttons. Each button is {"text": "...", "callback_data": "action:param"}
    
    Returns:
        True if sent successfully
    """
    bot = get_bot()
    result = bot.send_message(message, inline_keyboard=buttons)
    return result is not None


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

