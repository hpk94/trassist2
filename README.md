# Trassist2 - AI-Powered Trading Analysis System

An intelligent trading analysis system that combines OpenAI Vision API for chart analysis with MEXC API integration for real-time market data and comprehensive signal validation.

## 🚀 Features

- **AI Chart Analysis**: Uses OpenAI Vision API to analyze trading charts and extract technical indicators
- **Real-time Market Data**: Integrates with MEXC API for live market data and klines
- **Technical Indicators**: Calculates RSI14 and other technical indicators
- **Signal Validation**: Comprehensive checklist and invalidation system for trading signals
- **LLM Trade Gate**: Final AI decision layer for trade execution approval
- **Clean Output**: Minimal, essential logging with final signal status display

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- MEXC API credentials

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/hpk94/trassist2.git
cd trassist2
```

2. Create a virtual environment:
```bash
python -m venv venv
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
```
OPENAI_API_KEY=your_openai_api_key_here
MEXC_API_KEY=your_mexc_api_key_here
MEXC_API_SECRET=your_mexc_api_secret_here
```

## 📊 Usage

1. Place your trading chart image in the project directory
2. Update the `test_image` variable in `app.py` with your image filename
3. Run the analysis:
```bash
python app.py
```

The system will:
- Analyze the trading chart using AI
- Fetch real-time market data from MEXC
- Validate trading signals against technical indicators
- Display the final signal status (valid/invalidated/pending)

## 🏗️ Architecture

- **`app.py`**: Main application with trading analysis logic
- **`prompt.py`**: AI prompts for chart analysis and trade decisions
- **`llm_outputs/`**: Directory for storing AI analysis results
- **`.gitignore`**: Properly configured to exclude sensitive files

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
