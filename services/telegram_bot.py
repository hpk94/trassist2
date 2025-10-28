"""Telegram bot service for trading analysis and image uploads."""

import os
import asyncio
import json
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from io import BytesIO

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes, CallbackQueryHandler
)
from telegram.constants import ParseMode
from dotenv import load_dotenv

from .trading_service import TradingService
from .notification_service import NotificationService
from .status_service import StatusService, TradeAnalysis

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TradingTelegramBot:
    """Telegram bot for trading analysis with image upload functionality."""
    
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.authorized_users = self._get_authorized_users()
        self.trading_service = TradingService()
        self.notification_service = NotificationService()
        self.status_service = StatusService()
        
        # In-memory cache for recent analyses (persistent storage in StatusService)
        self.active_analyses = {}
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
    
    def _get_authorized_users(self) -> set:
        """Get list of authorized user IDs from environment."""
        users_str = os.getenv("TELEGRAM_AUTHORIZED_USERS", "")
        if not users_str:
            logger.warning("No authorized users configured. Bot will accept requests from anyone.")
            return set()
        
        try:
            return set(int(user_id.strip()) for user_id in users_str.split(",") if user_id.strip())
        except ValueError:
            logger.error("Invalid TELEGRAM_AUTHORIZED_USERS format. Should be comma-separated user IDs.")
            return set()
    
    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized to use the bot."""
        if not self.authorized_users:
            return True  # If no users configured, allow everyone
        return user_id in self.authorized_users
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        user = update.effective_user
        
        if not self._is_authorized(user.id):
            await update.message.reply_text(
                "❌ You are not authorized to use this bot. Please contact the administrator."
            )
            return
        
        welcome_message = f"""
🤖 **Trading Assistant Bot**

Welcome {user.first_name}! I can help you analyze trading charts and manage your trading signals.

**Available Commands:**
/help - Show this help message
/upload - Upload a chart image for analysis
/status - Check current trading status
/recent - Show recent analyses
/test - Test notification system

**Image Upload:**
Simply send me a chart image and I'll analyze it automatically!

**Features:**
📊 AI-powered chart analysis
📈 Technical indicator validation
🎯 Trade signal recommendations
📱 Real-time notifications
        """
        
        await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access.")
            return
        
        help_text = """
🔧 **Bot Commands & Usage**

**Commands:**
• `/start` - Welcome message and overview
• `/help` - Show this help message
• `/upload` - Upload chart for analysis
• `/status` - Current trading status
• `/recent` - Recent analyses (last 5)
• `/test` - Test notification systems

**Image Analysis:**
1. Send any chart image directly to the bot
2. Supported formats: PNG, JPG, JPEG, GIF, BMP, WEBP
3. Maximum file size: 20MB
4. The bot will automatically analyze technical indicators

**Analysis Results:**
• Chart pattern recognition
• Technical indicator values (RSI, MACD, etc.)
• Trading signal validation
• Entry/exit recommendations
• Risk management suggestions

**Status Tracking:**
• View active trading signals
• Monitor invalidation conditions
• Track analysis history

**Notifications:**
• Pushover notifications for valid trades
• Email alerts for important updates
• Telegram messages for status changes

Need help? Contact the administrator.
        """
        
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def upload_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /upload command."""
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access.")
            return
        
        await update.message.reply_text(
            "📸 Please send me a chart image to analyze.\n\n"
            "Supported formats: PNG, JPG, JPEG, GIF, BMP, WEBP\n"
            "Maximum size: 20MB"
        )
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access.")
            return
        
        user_id = update.effective_user.id
        
        # Get active analyses from persistent storage
        active_analyses = self.status_service.get_active_analyses(user_id)
        
        if not active_analyses:
            await update.message.reply_text("📊 No active trading analyses found.")
            return
        
        status_message = "📊 **Current Trading Status**\n\n"
        
        for analysis in active_analyses[:3]:  # Show last 3
            status_emoji = {
                "valid": "✅",
                "invalidated": "❌", 
                "pending": "⏳",
                "error": "🚫"
            }.get(analysis.signal_status, "❓")
            
            status_message += f"{status_emoji} **{analysis.symbol}**\n"
            status_message += f"Status: {analysis.signal_status.upper()}\n"
            status_message += f"Price: ${analysis.current_price:.4f}\n"
            status_message += f"Time: {analysis.timestamp}\n\n"
        
        # Add action buttons
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh Status", callback_data="refresh_status")],
            [InlineKeyboardButton("📈 Recent Analyses", callback_data="show_recent")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            status_message, 
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    async def recent_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /recent command to show recent analyses."""
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access.")
            return
        
        user_id = update.effective_user.id
        
        # Get recent analyses from persistent storage
        recent_analyses = self.status_service.get_user_analyses(user_id, limit=5)
        
        if not recent_analyses:
            await update.message.reply_text("📊 No recent analyses found.")
            return
        
        recent_message = "📈 **Recent Analyses**\n\n"
        
        for analysis in recent_analyses:
            status_emoji = {
                "valid": "✅",
                "invalidated": "❌",
                "pending": "⏳", 
                "error": "🚫"
            }.get(analysis.signal_status, "❓")
            
            recent_message += f"{status_emoji} **{analysis.symbol}** - {analysis.signal_status.upper()}\n"
            recent_message += f"💰 ${analysis.current_price:.4f} | ⏰ {analysis.timestamp}\n\n"
        
        await update.message.reply_text(recent_message, parse_mode=ParseMode.MARKDOWN)
    
    async def test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /test command to test notification systems."""
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access.")
            return
        
        await update.message.reply_text("🧪 Testing notification systems...")
        
        # Test notification service
        results = self.notification_service.test_notifications()
        
        test_message = "🧪 **Notification Test Results**\n\n"
        test_message += f"📱 Pushover: {'✅ Success' if results.get('pushover') else '❌ Failed'}\n"
        test_message += f"📧 Email: {'✅ Success' if results.get('email') else '❌ Failed'}\n"
        test_message += f"💬 Telegram: ✅ Success (you received this message!)\n"
        
        await update.message.reply_text(test_message, parse_mode=ParseMode.MARKDOWN)
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle image uploads for trading analysis."""
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access.")
            return
        
        user = update.effective_user
        message = update.message
        
        # Check if message contains an image
        if not message.photo and not message.document:
            await message.reply_text("❌ Please send an image file for analysis.")
            return
        
        try:
            # Send initial processing message
            processing_msg = await message.reply_text(
                "🔄 **Processing your chart...**\n\n"
                "⏳ Downloading image...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Download the image
            if message.photo:
                # Get the highest resolution photo
                photo = message.photo[-1]
                file = await context.bot.get_file(photo.file_id)
            else:
                # Handle document (image file)
                document = message.document
                if not document.mime_type or not document.mime_type.startswith('image/'):
                    await processing_msg.edit_text("❌ Please send a valid image file.")
                    return
                file = await context.bot.get_file(document.file_id)
            
            # Download image data
            image_data = BytesIO()
            await file.download_to_memory(image_data)
            image_bytes = image_data.getvalue()
            
            # Update processing message
            await processing_msg.edit_text(
                "🔄 **Processing your chart...**\n\n"
                "✅ Image downloaded\n"
                "🤖 Analyzing with AI...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Run analysis
            analysis_result = self.trading_service.execute_full_analysis(image_data=image_bytes)
            
            # Update processing message
            await processing_msg.edit_text(
                "🔄 **Processing your chart...**\n\n"
                "✅ Image downloaded\n"
                "✅ AI analysis complete\n"
                "📊 Preparing results...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Store analysis result
            analysis_id = f"{user.id}_{int(datetime.now().timestamp())}"
            analysis_result.update({
                "analysis_id": analysis_id,
                "user_id": user.id,
                "username": user.username or user.first_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Save to persistent storage
            self.status_service.save_analysis_from_dict(analysis_result)
            
            # Also keep in memory cache for quick access
            self.active_analyses[analysis_id] = analysis_result
            
            # Format and send results
            result_message = self._format_analysis_result(analysis_result)
            
            # Create action buttons
            keyboard = []
            if analysis_result.get("signal_status") == "valid":
                keyboard.append([InlineKeyboardButton("📊 View Details", callback_data=f"details_{analysis_id}")])
            keyboard.append([InlineKeyboardButton("🔄 Refresh Status", callback_data="refresh_status")])
            
            reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
            
            # Delete processing message and send results
            await processing_msg.delete()
            await message.reply_text(
                result_message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await processing_msg.edit_text(
                f"❌ **Error processing image**\n\n"
                f"Error: {str(e)}\n\n"
                f"Please try again or contact support.",
                parse_mode=ParseMode.MARKDOWN
            )
    
    def _format_analysis_result(self, result: Dict[str, Any]) -> str:
        """Format analysis result for Telegram message."""
        if result.get("status") == "error":
            return f"❌ **Analysis Error**\n\n{result.get('message', 'Unknown error')}"
        
        symbol = result.get("symbol", "Unknown")
        signal_status = result.get("signal_status", "unknown")
        current_price = result.get("current_price", 0)
        current_rsi = result.get("current_rsi", 0)
        
        status_emoji = {
            "valid": "✅",
            "invalidated": "❌",
            "pending": "⏳",
            "error": "🚫"
        }.get(signal_status, "❓")
        
        message = f"{status_emoji} **Trading Analysis Complete**\n\n"
        message += f"📊 **Symbol:** {symbol}\n"
        message += f"📈 **Status:** {signal_status.upper()}\n"
        message += f"💰 **Price:** ${current_price:.4f}\n"
        
        if current_rsi:
            message += f"📊 **RSI:** {current_rsi:.2f}\n"
        
        # Add specific status information
        if signal_status == "valid":
            gate_result = result.get("gate_result", {})
            if gate_result.get("should_open"):
                direction = gate_result.get("direction", "unknown")
                confidence = gate_result.get("confidence", 0)
                message += f"\n🎯 **Trade Signal:** {direction.upper()}\n"
                message += f"🎲 **Confidence:** {confidence:.1%}\n"
                
                execution = gate_result.get("execution", {})
                if execution.get("stop_loss"):
                    message += f"🛑 **Stop Loss:** ${execution['stop_loss']:.4f}\n"
                if execution.get("risk_reward"):
                    message += f"⚖️ **Risk/Reward:** {execution['risk_reward']:.2f}\n"
            else:
                reasons = gate_result.get("reasons", [])
                message += f"\n⚠️ **Trade Not Approved**\n"
                if reasons:
                    message += f"Reasons: {', '.join(reasons[:2])}\n"
        
        elif signal_status == "invalidated":
            triggered = result.get("triggered_conditions", [])
            if triggered:
                message += f"\n⚠️ **Invalidated by:** {', '.join(triggered[:2])}\n"
        
        elif signal_status == "pending":
            message += f"\n⏳ **Waiting for conditions to be met**\n"
        
        message += f"\n⏰ **Time:** {result.get('timestamp', 'Unknown')}"
        
        return message
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline keyboard callbacks."""
        if not self._is_authorized(update.effective_user.id):
            await update.callback_query.answer("❌ Unauthorized access.")
            return
        
        query = update.callback_query
        await query.answer()
        
        if query.data == "refresh_status":
            await self._handle_refresh_status(query, context)
        elif query.data == "show_recent":
            await self._handle_show_recent(query, context)
        elif query.data.startswith("details_"):
            analysis_id = query.data.replace("details_", "")
            await self._handle_show_details(query, context, analysis_id)
    
    async def _handle_refresh_status(self, query, context) -> None:
        """Handle refresh status callback."""
        user_id = query.from_user.id
        user_analyses = {k: v for k, v in self.active_analyses.items() if v.get("user_id") == user_id}
        
        if not user_analyses:
            await query.edit_message_text("📊 No active trading analyses found.")
            return
        
        status_message = "📊 **Current Trading Status** (Refreshed)\n\n"
        
        for analysis_id, analysis in list(user_analyses.items())[-3:]:
            symbol = analysis.get("symbol", "Unknown")
            signal_status = analysis.get("signal_status", "unknown")
            timestamp = analysis.get("timestamp", "Unknown")
            current_price = analysis.get("current_price", 0)
            
            status_emoji = {
                "valid": "✅",
                "invalidated": "❌",
                "pending": "⏳",
                "error": "🚫"
            }.get(signal_status, "❓")
            
            status_message += f"{status_emoji} **{symbol}**\n"
            status_message += f"Status: {signal_status.upper()}\n"
            status_message += f"Price: ${current_price:.4f}\n"
            status_message += f"Time: {timestamp}\n\n"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh Again", callback_data="refresh_status")],
            [InlineKeyboardButton("📈 Recent Analyses", callback_data="show_recent")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            status_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    async def _handle_show_recent(self, query, context) -> None:
        """Handle show recent callback."""
        user_id = query.from_user.id
        user_analyses = {k: v for k, v in self.active_analyses.items() if v.get("user_id") == user_id}
        
        if not user_analyses:
            await query.edit_message_text("📊 No recent analyses found.")
            return
        
        recent_message = "📈 **Recent Analyses**\n\n"
        
        sorted_analyses = sorted(user_analyses.items(),
                               key=lambda x: x[1].get("timestamp", ""),
                               reverse=True)[:5]
        
        for analysis_id, analysis in sorted_analyses:
            symbol = analysis.get("symbol", "Unknown")
            signal_status = analysis.get("signal_status", "unknown")
            timestamp = analysis.get("timestamp", "Unknown")
            current_price = analysis.get("current_price", 0)
            
            status_emoji = {
                "valid": "✅",
                "invalidated": "❌",
                "pending": "⏳",
                "error": "🚫"
            }.get(signal_status, "❓")
            
            recent_message += f"{status_emoji} **{symbol}** - {signal_status.upper()}\n"
            recent_message += f"💰 ${current_price:.4f} | ⏰ {timestamp}\n\n"
        
        keyboard = [[InlineKeyboardButton("🔄 Back to Status", callback_data="refresh_status")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            recent_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    async def _handle_show_details(self, query, context, analysis_id: str) -> None:
        """Handle show details callback."""
        analysis = self.active_analyses.get(analysis_id)
        if not analysis:
            await query.edit_message_text("❌ Analysis not found.")
            return
        
        # Format detailed analysis information
        llm_output = analysis.get("llm_output", {})
        symbol = analysis.get("symbol", "Unknown")
        
        details_message = f"📊 **Detailed Analysis: {symbol}**\n\n"
        
        # Technical indicators
        tech_indicators = llm_output.get("technical_indicators", {})
        if tech_indicators:
            details_message += "📈 **Technical Indicators:**\n"
            
            rsi = tech_indicators.get("RSI14", {})
            if rsi:
                details_message += f"• RSI14: {rsi.get('value', 'N/A')} ({rsi.get('status', 'N/A')})\n"
            
            macd = tech_indicators.get("MACD12_26_9", {})
            if macd:
                details_message += f"• MACD: {macd.get('signal', 'N/A')}\n"
            
            bb = tech_indicators.get("BB20_2", {})
            if bb:
                details_message += f"• Bollinger: {bb.get('price_position', 'N/A')}\n"
            
            details_message += "\n"
        
        # Risk management
        risk_mgmt = llm_output.get("risk_management", {})
        if risk_mgmt:
            details_message += "⚖️ **Risk Management:**\n"
            
            stop_loss = risk_mgmt.get("stop_loss", {})
            if stop_loss:
                details_message += f"• Stop Loss: ${stop_loss.get('price', 'N/A')}\n"
            
            take_profits = risk_mgmt.get("take_profit", [])
            if take_profits:
                for i, tp in enumerate(take_profits[:2], 1):
                    details_message += f"• TP{i}: ${tp.get('price', 'N/A')} (RR: {tp.get('rr', 'N/A')})\n"
        
        keyboard = [[InlineKeyboardButton("🔄 Back to Status", callback_data="refresh_status")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            details_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    async def handle_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors."""
        logger.error(f"Update {update} caused error {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An error occurred. Please try again or contact support."
            )
    
    def run(self) -> None:
        """Run the Telegram bot."""
        # Create application
        application = Application.builder().token(self.bot_token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("upload", self.upload_command))
        application.add_handler(CommandHandler("status", self.status_command))
        application.add_handler(CommandHandler("recent", self.recent_command))
        application.add_handler(CommandHandler("test", self.test_command))
        
        # Image handler (photos and documents)
        application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, self.handle_image))
        
        # Callback query handler
        application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Error handler
        application.add_error_handler(self.handle_error)
        
        # Start the bot
        logger.info("Starting Telegram bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """Main function to run the Telegram bot."""
    try:
        bot = TradingTelegramBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise

if __name__ == "__main__":
    main()