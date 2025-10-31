# Trassist2 - AI-Powered Trading Analysis System

An intelligent trading analysis system that combines LLM vision capabilities for chart analysis with MEXC API integration for real-time market data and comprehensive signal validation.

## 🚀 Features

- **Multi-Model AI Support**: Uses LiteLLM to support OpenAI, Anthropic Claude, Google Gemini, and more
- **AI Chart Analysis**: Vision-capable LLMs analyze trading charts and extract technical indicators
- **Real-time Market Data**: Integrates with MEXC API for live market data and klines
- **Technical Indicators**: Calculates RSI14, MACD, Stochastic, Bollinger Bands, and ATR
- **Signal Validation**: Comprehensive checklist and invalidation system for trading signals
- **LLM Trade Gate**: Final AI decision layer for trade execution approval
- **iPhone Notifications**: Pushover and email notifications for valid and invalidated trades
- **Flexible Model Selection**: Easily switch between different LLM providers via environment variable

## 📋 Requirements

- Python 3.8+
- LLM API key (OpenAI, Anthropic, Google, or others)
- MEXC API credentials

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
# Edit .env with your API keys
```

## 🔧 Configuration

Create a `.env` file with the following variables:
```bash
# LiteLLM Model Selection (choose one)
LITELLM_MODEL=gpt-4o  # Default: OpenAI GPT-4o

# API Keys (provide the key for your chosen model provider)
OPENAI_API_KEY=your_openai_api_key_here          # For OpenAI models
# ANTHROPIC_API_KEY=your_anthropic_api_key_here  # For Claude models
# GEMINI_API_KEY=your_gemini_api_key_here        # For Gemini models

# MEXC Exchange Configuration
MEXC_API_KEY=your_mexc_api_key_here
MEXC_API_SECRET=your_mexc_api_secret_here
MEXC_DEFAULT_VOL=0.001
MEXC_OPEN_TYPE=2
```

### Supported LLM Models

**Vision-capable models** (required for chart analysis):
- `gpt-4o` (OpenAI) - **Recommended**
- `gpt-4-turbo` (OpenAI)
- `claude-3-5-sonnet-20241022` (Anthropic)
- `claude-3-opus-20240229` (Anthropic)
- `gemini/gemini-1.5-pro` (Google)

**Text-only models** (for trade gate decisions):
- `deepseek/deepseek-chat` (DeepSeek) - **Cost-effective**
- All vision models above also work for text

### Dual-Model Setup (Save Costs!)

Use different models for different tasks:
```bash
LITELLM_VISION_MODEL=gpt-4o              # For chart analysis
LITELLM_TEXT_MODEL=deepseek/deepseek-chat # For trade gates (50% cost savings!)
```

See [`DEEPSEEK_GUIDE.md`](DEEPSEEK_GUIDE.md) for DeepSeek setup and [`LITELLM_CONFIGURATION.md`](LITELLM_CONFIGURATION.md) for all options.

## 📊 Usage

1. Place your trading chart image in the project directory
2. Update the `test_image` variable in `app.py` with your image filename
3. Run the analysis:
```bash
python3 app.py
```

The system will:
- Analyze the trading chart using AI vision
- Fetch real-time market data from MEXC
- Calculate technical indicators (RSI, MACD, Stochastic, BB, ATR)
- Validate trading signals against checklist and invalidation conditions
- Run LLM trade gate for final approval
- Send notifications for valid/invalidated trades
- Display the final signal status (valid/invalidated/pending)

### Switching LLM Models

You can easily switch between different models by changing the `LITELLM_MODEL` environment variable:

```bash
# Use OpenAI GPT-4o (default)
LITELLM_MODEL=gpt-4o python3 app.py

# Use Anthropic Claude 3.5 Sonnet
LITELLM_MODEL=claude-3-5-sonnet-20241022 python3 app.py

# Use Google Gemini 1.5 Pro
LITELLM_MODEL=gemini/gemini-1.5-pro python3 app.py
```

Or set it permanently in your `.env` file. See [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md) for more details.

## 🏗️ Architecture

### Core Files
- **`app.py`**: Main application with trading analysis logic (uses LiteLLM)
- **`prompt.py`**: AI prompts for chart analysis and trade decisions
- **`services/`**: Notification and indicator calculation services
  - `notification_service.py`: Pushover and email notifications
  - `indicator_service.py`: Technical indicator calculations

### Data & Output
- **`llm_outputs/`**: Directory for storing AI analysis results (JSON)
- **`trades.db`**: SQLite database for trade history

### Documentation
- **`README.md`**: This file
- **`LITELLM_CONFIGURATION.md`**: Complete LiteLLM setup and model reference
- **`MIGRATION_GUIDE.md`**: Migration guide from OpenAI to LiteLLM
- **`NOTIFICATION_SETUP.md`**: iPhone notification configuration guide

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
