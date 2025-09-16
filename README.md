# Trassist2 - AI-Powered Trading Analysis System

An intelligent trading analysis system that combines OpenAI Vision API for chart analysis with MEXC API integration for real-time market data and comprehensive signal validation. Now includes **Telegram Bot Integration** for seamless chart uploads and status monitoring!

## 🚀 Features

### Core Analysis
- **AI Chart Analysis**: Uses OpenAI Vision API to analyze trading charts and extract technical indicators
- **Real-time Market Data**: Integrates with MEXC API for live market data and klines
- **Technical Indicators**: Calculates RSI14, MACD, Bollinger Bands, Stochastic, and ATR
- **Signal Validation**: Comprehensive checklist and invalidation system for trading signals
- **LLM Trade Gate**: Final AI decision layer for trade execution approval

### Telegram Bot Integration 🤖
- **📸 Image Upload**: Send chart screenshots directly to the bot for instant analysis
- **📊 Status Tracking**: Monitor active trading signals and analysis history
- **📱 Real-time Notifications**: Get notified of valid trades and signal changes
- **🔐 User Authorization**: Secure access control with authorized user management
- **💾 Persistent Storage**: SQLite database for analysis history and status tracking

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- MEXC API credentials
- Telegram Bot Token (for bot integration)
- Optional: Pushover/Email credentials for notifications

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/hpk94/trassist2.git
cd trassist2
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and bot token
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
MEXC_API_KEY=your_mexc_api_key_here
MEXC_API_SECRET=your_mexc_api_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Optional but recommended
TELEGRAM_AUTHORIZED_USERS=123456789,987654321
PUSHOVER_TOKEN=your_pushover_token_here
PUSHOVER_USER=your_pushover_user_key_here
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
EMAIL_TO=recipient@gmail.com
```

### Setting up the Telegram Bot

1. **Create a Telegram Bot**:
   - Message [@BotFather](https://t.me/BotFather) on Telegram
   - Use `/newbot` command and follow instructions
   - Copy the bot token to your `.env` file

2. **Get Your User ID**:
   - Message [@userinfobot](https://t.me/userinfobot) to get your Telegram user ID
   - Add your user ID to `TELEGRAM_AUTHORIZED_USERS` in `.env`

3. **Optional Notifications**:
   - **Pushover**: Sign up at [pushover.net](https://pushover.net) for mobile notifications
   - **Email**: Use Gmail app passwords for email notifications

## 📊 Usage

### Option 1: Telegram Bot (Recommended)
```bash
python3 telegram_bot_main.py
```

Then interact with your bot on Telegram:
- Send chart images directly to the bot
- Use `/status` to check active signals
- Use `/recent` to see analysis history
- Use `/help` for available commands

### Option 2: Direct Script (Legacy)
```bash
python3 app.py  # Analyze predefined test image
python3 web_app.py  # Run web interface
```

## 🤖 Telegram Bot Commands

- `/start` - Welcome message and bot overview
- `/help` - Show available commands and usage
- `/upload` - Instructions for uploading charts
- `/status` - View active trading signals
- `/recent` - Show recent analysis history
- `/test` - Test notification systems

**Image Analysis**: Simply send any chart image to the bot for automatic analysis!

## 🏗️ Architecture

### Core Files
- **`telegram_bot_main.py`**: Main launcher for Telegram bot
- **`app.py`**: Legacy standalone analysis script
- **`web_app.py`**: Web interface for chart uploads
- **`prompt.py`**: AI prompts for chart analysis and trade decisions

### Services
- **`services/telegram_bot.py`**: Telegram bot implementation with commands and handlers
- **`services/trading_service.py`**: Core trading analysis logic extracted from app.py
- **`services/status_service.py`**: Persistent storage for analysis history and status
- **`services/notification_service.py`**: Multi-channel notification system (Pushover, Email)
- **`services/indicator_service.py`**: Technical indicator calculations

### Storage
- **`llm_outputs/`**: JSON files with detailed analysis results
- **`trade_status.db`**: SQLite database for persistent analysis storage
- **`.env`**: Environment configuration (not tracked in git)

## 🔒 Security

- API keys are stored in `.env` file (excluded from version control)
- Sensitive data is never committed to the repository
- GitHub's push protection prevents accidental secret exposure

## 📈 Output

The system provides:
- Final signal status (valid/invalidated/pending)
- Essential error messages for debugging
- JSON output files for analysis results
- Clean, production-ready logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves risk, and past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making trading decisions.
