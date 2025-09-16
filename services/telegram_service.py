import os
import tempfile
from typing import Optional, Callable

from dotenv import load_dotenv

# We use python-telegram-bot v20+ (async API)
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext, filters

# We will reuse the web_app pipeline and its progress fan-out
from web_app import run_trading_analysis, add_progress_listener, remove_progress_listener


load_dotenv()


class TelegramTradingBot:
    def __init__(self, token: str, allowed_chat_id: Optional[int] = None):
        self.token = token
        self.allowed_chat_id = allowed_chat_id
        self.application: Optional[Application] = None

    def _is_authorized(self, update: Update) -> bool:
        if self.allowed_chat_id is None:
            return True
        chat = update.effective_chat
        return chat and chat.id == self.allowed_chat_id

    async def _start(self, update: Update, context: CallbackContext):
        if not self._is_authorized(update):
            return
        await update.message.reply_text("Send me a chart image and I will analyze it. Use /status to see progress.")

    async def _status(self, update: Update, context: CallbackContext):
        if not self._is_authorized(update):
            return
        await update.message.reply_text("Waiting for an image. I will stream progress as I work.")

    async def _handle_photo(self, update: Update, context: CallbackContext):
        if not self._is_authorized(update):
            return

        if not update.message or not update.message.photo:
            await update.message.reply_text("Please send an image (photo) of the chart.")
            return

        # Get highest resolution photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)

        # Save to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            temp_path = tmp.name
        await file.download_to_drive(temp_path)

        # Stream progress back to this chat using the progress listener
        async def _send_progress(progress):
            try:
                message = progress.get('message', '')
                # squash too-verbose JSON payloads
                if isinstance(message, str) and len(message) > 1800:
                    message = message[:1800] + '…'
                await context.bot.send_message(chat_id=update.effective_chat.id, text=str(message))
            except Exception:
                pass

        # Wrap async sender into a sync callable expected by add_progress_listener
        def progress_listener(progress):
            # Fire-and-forget via application.create_task
            try:
                self.application.create_task(_send_progress(progress))
            except Exception:
                pass

        add_progress_listener(progress_listener)
        try:
            await update.message.reply_text("Analyzing your chart. This may take a couple of minutes…")
            # Run analysis in a background thread to avoid blocking the event loop
            import asyncio
            result = await asyncio.to_thread(run_trading_analysis, temp_path)

            # Send a concise summary
            success = result.get('success') if isinstance(result, dict) else False
            if success:
                signal_status = result.get('signal_status')
                gate_result = result.get('gate_result') or {}
                should_open = gate_result.get('should_open', False)
                direction = gate_result.get('direction', 'unknown')
                confidence = gate_result.get('confidence', 0.0)
                await update.message.reply_text(
                    f"Done. Signal: {signal_status}. Gate: {'OPEN' if should_open else 'NO-OPEN'} "
                    f"({direction}, conf {confidence:.2f})."
                )
            else:
                await update.message.reply_text("Analysis failed. Check server logs for details.")
        finally:
            remove_progress_listener(progress_listener)
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass

    def build(self) -> Application:
        self.application = Application.builder().token(self.token).build()
        self.application.add_handler(CommandHandler("start", self._start))
        self.application.add_handler(CommandHandler("status", self._status))
        self.application.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        return self.application


def get_bot_from_env() -> TelegramTradingBot:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    allowed_chat = os.getenv("TELEGRAM_ALLOWED_CHAT_ID")
    allowed_chat_id = int(allowed_chat) if allowed_chat and allowed_chat.isdigit() else None
    return TelegramTradingBot(token, allowed_chat_id)

