# Trading Analysis Services Architecture

This document describes the refactored service architecture for the trading analysis application.

## Overview

The application has been split into several focused services, each handling a specific aspect of the trading analysis workflow:

- **LLMService**: Handles chart analysis and trade gate decisions using OpenAI
- **MarketDataService**: Manages MEXC API interactions and market data fetching
- **TechnicalAnalysisService**: Performs technical indicator calculations
- **SignalValidationService**: Validates trading signals against checklist and invalidation conditions
- **TradingOrchestrator**: Coordinates all services and manages the complete workflow

## Service Details

### LLMService (`services/llm_service.py`)

Handles all LLM interactions:
- `analyze_trading_chart(image_path)`: Analyzes chart images using OpenAI Vision
- `llm_trade_gate_decision(...)`: Makes final trade decisions after programmatic validation

### MarketDataService (`services/market_data_service.py`)

Manages market data operations:
- `fetch_market_data(symbol, timeframe, limit)`: Fetches raw klines data
- `fetch_market_dataframe(symbol, timeframe, limit)`: Returns processed DataFrame
- `validate_timeframe(timeframe)`: Validates timeframe against MEXC intervals
- `get_current_market_values(df)`: Extracts current market values

### TechnicalAnalysisService (`services/technical_analysis_service.py`)

Performs technical analysis calculations:
- `calculate_rsi14(df)`: Calculates RSI14 indicator
- `calculate_timeframe_seconds(interval)`: Converts timeframe to seconds
- `calculate_time_difference(time_of_screenshot, df)`: Calculates time differences

### SignalValidationService (`services/signal_validation_service.py`)

Validates trading signals:
- `evaluate_comparison(current_value, comparator, target_value)`: Evaluates comparisons
- `check_indicator_threshold(df, condition)`: Checks indicator threshold conditions
- `check_price_level(df, condition, llm_output)`: Checks price level conditions
- `indicator_checker(df, llm_output)`: Checks all technical indicator conditions
- `invalidation_checker(df, llm_output)`: Checks invalidation conditions
- `validate_trading_signal(df, llm_output)`: Comprehensive signal validation

### TradingOrchestrator (`services/trading_orchestrator.py`)

Main orchestrator that coordinates all services:
- `analyze_chart(image_path)`: Analyzes chart using LLM service
- `save_llm_output(llm_output)`: Saves LLM output to file
- `process_trading_signal(llm_output)`: Processes a single trading signal
- `poll_until_decision(llm_output, max_cycles)`: Polls until decision is made
- `make_trade_decision(...)`: Makes final trade decision using gate
- `run_complete_analysis(image_path, max_cycles)`: Runs complete workflow

## Usage Examples

### Using the Orchestrator (Recommended)

```python
from services.trading_orchestrator import TradingOrchestrator

# Initialize orchestrator
orchestrator = TradingOrchestrator()

# Run complete analysis
results = orchestrator.run_complete_analysis("chart.png", max_cycles=5)

# Access results
print(f"Signal Status: {results['signal_status']}")
print(f"Gate Decision: {results['gate_result']}")
```

### Using Individual Services

```python
from services.market_data_service import MarketDataService
from services.technical_analysis_service import TechnicalAnalysisService

# Fetch market data
market_service = MarketDataService()
df = market_service.fetch_market_dataframe("BTCUSDT", "1m")

# Calculate indicators
tech_service = TechnicalAnalysisService()
df_with_rsi = tech_service.calculate_rsi14(df)
```

## Migration from Original Code

The original `app.py` contained all functionality in a single file. The refactored version:

1. **Separates concerns**: Each service handles a specific domain
2. **Improves testability**: Services can be tested independently
3. **Enhances maintainability**: Changes to one service don't affect others
4. **Enables reusability**: Services can be used in different contexts
5. **Simplifies debugging**: Issues can be isolated to specific services

## File Structure

```
services/
├── __init__.py
├── llm_service.py
├── market_data_service.py
├── technical_analysis_service.py
├── signal_validation_service.py
└── trading_orchestrator.py

app_refactored.py          # New main application
example_usage.py           # Example usage of individual services
app.py                     # Original application (preserved)
app copy.py                # Original backup (preserved)
```

## Benefits of the New Architecture

1. **Modularity**: Each service has a single responsibility
2. **Scalability**: Easy to add new services or modify existing ones
3. **Testing**: Services can be unit tested independently
4. **Configuration**: Services can be configured separately
5. **Error Handling**: Errors can be isolated to specific services
6. **Documentation**: Each service is self-documenting with clear interfaces

## Next Steps

1. Add unit tests for each service
2. Add configuration management
3. Add logging to each service
4. Add error handling and retry logic
5. Consider adding a service registry for dependency injection
