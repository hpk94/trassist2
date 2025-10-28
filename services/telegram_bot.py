import os
import asyncio
import tempfile
from typing import Callable, Optional

from dotenv import load_dotenv

# Lazy import telegram to avoid mandatory dependency when not used
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
except Exception:
    Update = object  # type: ignore
    Application = None  # type: ignore
    CommandHandler = object  # type: ignore
    MessageHandler = object  # type: ignore
    ContextTypes = object  # type: ignore
    filters = None  # type: ignore

# Import analysis and progress hooks from the web app module
from web_app import run_trading_analysis, add_progress_listener, remove_progress_listener, get_progress_history


load_dotenv()


class TelegramAnalysisBot:
    """Telegram bot for uploading images and streaming analysis status."""

    def __init__(self, token: str, chat_whitelist: Optional[str] = None):
        if Application is None:
            raise RuntimeError("python-telegram-bot is not installed. Please add it to requirements.")

        self.token = token
        self.chat_whitelist = set()
        if chat_whitelist:
            # Comma-separated chat IDs or usernames (e.g., "12345,@myuser")
            for item in chat_whitelist.split(','):
                val = item.strip()
                if val:
                    self.chat_whitelist.add(val)

        self.app = Application.builder().token(self.token).build()

        # Handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))

    def _is_allowed(self, update: Update) -> bool:
        if not self.chat_whitelist:
            return True
        chat = update.effective_chat
        user = update.effective_user
        allowed = False
        if chat:
            if str(chat.id) in self.chat_whitelist or (chat.username and f"@{chat.username}" in self.chat_whitelist):
                allowed = True
        if not allowed and user:
            if str(user.id) in self.chat_whitelist or (user.username and f"@{user.username}" in self.chat_whitelist):
                allowed = True
        return allowed

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return
        await update.message.reply_text(
            "Send me a chart screenshot as a photo. I will analyze it and stream progress here.\n"
            "Commands: /status (last 20 logs), /help"
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return
        await update.message.reply_text(
            "Usage:\n"
            "- Send a photo of a trading chart to start analysis.\n"
            "- /status to see recent logs."
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return
        logs = get_progress_history(20)
        if not logs:
            await update.message.reply_text("No recent activity.")
            return
        text_lines = [f"[{m['timestamp']}] {m['message']}" for m in logs]
        text = "\n".join(text_lines)
        # Telegram messages are limited; truncate if necessary
        if len(text) > 3500:
            text = text[-3500:]
        await update.message.reply_text(text)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return

        message = update.message
        if not message or not message.photo:
            return

        await message.reply_text("📥 Received image. Starting analysis... You will receive live updates here.")

        # Get the highest-resolution photo
        photo = message.photo[-1]
        file = await photo.get_file()

        # Download to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            temp_path = tmp.name
        await file.download_to_drive(temp_path)

        chat_id = message.chat_id

        # Define a listener that forwards progress to Telegram
        async def forward(progress: dict):
            try:
                msg = f"[{progress.get('timestamp','')}] {progress.get('message','')}"
                # Keep messages concise to avoid flooding
                if len(msg) > 1000:
                    msg = msg[:1000] + '...'
                await context.bot.send_message(chat_id=chat_id, text=msg)
            except Exception:
                pass

        # Bridge sync emit_progress -> async telegram send
        def sync_listener(progress: dict):
            asyncio.create_task(forward(progress))

        # Register listener and run analysis in a thread to avoid blocking the event loop
        add_progress_listener(sync_listener)

        loop = asyncio.get_running_loop()

        def run_and_cleanup():
            try:
                run_trading_analysis(temp_path)
            finally:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

        await loop.run_in_executor(None, run_and_cleanup)

        # Remove listener after completion
        remove_progress_listener(sync_listener)
        await message.reply_text("✅ Analysis complete. Use /status to see recent logs or send another image.")

    def run(self):
        self.app.run_polling()


def run_from_env():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment.")
    whitelist = os.getenv("TELEGRAM_WHITELIST")  # optional
    bot = TelegramAnalysisBot(token=token, chat_whitelist=whitelist)
    bot.run()


if __name__ == "__main__":
    run_from_env()

