"""Market data service for MEXC API interactions."""

import os
import pandas as pd
from typing import List, Dict, Any
from pymexc import spot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MarketDataService:
    """Service for fetching and processing market data from MEXC API."""
    
    def __init__(self):
        """Initialize the market data service with MEXC clients."""
        # Initialize HTTP client for authenticated endpoints
        self.spot_client = spot.HTTP(
            api_key=os.getenv("MEXC_API_KEY"), 
            api_secret=os.getenv("MEXC_API_SECRET")
        )
        # Initialize public HTTP client for market data (no auth required)
        self.public_spot_client = spot.HTTP()
        # Initialize WebSocket client
        self.ws_spot_client = spot.WebSocket(
            api_key=os.getenv("MEXC_API_KEY"), 
            api_secret=os.getenv("MEXC_API_SECRET")
        )
    
    def fetch_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> List[List]:
        """
        Fetch raw klines data from MEXC API.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Time interval (e.g., '1m', '5m', '1h')
            limit: Number of klines to fetch (default: 100)
            
        Returns:
            List of kline data
        """
        try:
            klines = self.public_spot_client.klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            return klines
        except Exception as e:
            print(f"Error retrieving klines: {e}")
            return []
    
    def fetch_market_dataframe(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch market data and return as processed DataFrame.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Time interval (e.g., '1m', '5m', '1h')
            limit: Number of klines to fetch (default: 100)
            
        Returns:
            Processed DataFrame with market data
        """
        klines = self.fetch_market_data(symbol, timeframe, limit)
        
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines)
        df.columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume']
        
        # Convert timestamps to readable datetime
        df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
        df['Close_time'] = pd.to_datetime(df['Close_time'], unit='ms')
        
        # Convert price columns to numeric
        df['Open'] = pd.to_numeric(df['Open'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['Close'] = pd.to_numeric(df['Close'])
        df['Volume'] = pd.to_numeric(df['Volume'])
        
        return df
    
    def validate_timeframe(self, timeframe: str) -> str:
        """
        Validate timeframe against MEXC supported intervals.
        
        Args:
            timeframe: Time interval to validate
            
        Returns:
            Validated timeframe (defaults to '1m' if invalid)
        """
        valid_intervals = ['1m', '5m', '15m', '30m', '60m', '4h', '1d', '1W', '1M']
        if timeframe not in valid_intervals:
            print(f"Warning: {timeframe} is not a valid MEXC interval. Using 1m as default.")
            return '1m'
        return timeframe
    
    def get_current_market_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract current market values from DataFrame.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            Dictionary with current market values
        """
        if df.empty:
            return {
                'current_price': 0,
                'current_time': None,
                'current_rsi': None
            }
        
        current_price = df['Close'].iloc[-1]
        current_time = df['Open_time'].iloc[-1]
        current_rsi = df['RSI14'].iloc[-1] if 'RSI14' in df.columns else None
        
        return {
            'current_price': current_price,
            'current_time': current_time,
            'current_rsi': current_rsi
        }
