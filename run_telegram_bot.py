from services.telegram_service import get_bot_from_env


def main():
    bot = get_bot_from_env()
    app = bot.build()
    app.run_polling()


if __name__ == "__main__":
    main()

