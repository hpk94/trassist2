"""Example usage of individual services."""

from services.llm_service import LLMService
from services.market_data_service import MarketDataService
from services.technical_analysis_service import TechnicalAnalysisService
from services.signal_validation_service import SignalValidationService


def example_individual_services():
    """Example of using individual services."""
    
    # Example 1: LLM Service
    print("=== LLM Service Example ===")
    llm_service = LLMService()
    # llm_output = llm_service.analyze_trading_chart("test_image.png")
    # print("LLM Analysis completed")
    
    # Example 2: Market Data Service
    print("\n=== Market Data Service Example ===")
    market_service = MarketDataService()
    df = market_service.fetch_market_dataframe("BTCUSDT", "1m", limit=50)
    print(f"Fetched {len(df)} candles for BTCUSDT")
    if not df.empty:
        print(f"Latest price: {df['Close'].iloc[-1]}")
        print(f"Latest time: {df['Open_time'].iloc[-1]}")
    
    # Example 3: Technical Analysis Service
    print("\n=== Technical Analysis Service Example ===")
    tech_service = TechnicalAnalysisService()
    if not df.empty:
        df_with_rsi = tech_service.calculate_rsi14(df.copy())
        if 'RSI14' in df_with_rsi.columns:
            latest_rsi = df_with_rsi['RSI14'].iloc[-1]
            print(f"Latest RSI14: {latest_rsi:.2f}")
    
    # Example 4: Signal Validation Service
    print("\n=== Signal Validation Service Example ===")
    validation_service = SignalValidationService()
    
    # Example comparison
    result = validation_service.evaluate_comparison(45.5, ">=", 30.0)
    print(f"RSI 45.5 >= 30.0: {result}")
    
    # Example timeframe conversion
    seconds = tech_service.calculate_timeframe_seconds("5m")
    print(f"5 minutes = {seconds} seconds")


if __name__ == "__main__":
    example_individual_services()
