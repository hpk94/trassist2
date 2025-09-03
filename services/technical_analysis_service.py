"""Technical analysis service for indicators and calculations."""

import pandas as pd
from typing import Dict, Any


class TechnicalAnalysisService:
    """Service for technical analysis calculations and indicators."""
    
    def __init__(self):
        """Initialize the technical analysis service."""
        pass
    
    def calculate_rsi14(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI14 indicator manually using pandas.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with RSI14 column added
        """
        if df.empty or len(df) < 15:  # Need 15 candles to calculate 14-period RSI
            print("Warning: Not enough data points for RSI14 calculation (need at least 15 candles)")
            return df
        
        # Calculate price changes
        df['Price_Change'] = df['Close'].diff()
        
        # Separate gains and losses
        df['Gain'] = df['Price_Change'].where(df['Price_Change'] > 0, 0)
        df['Loss'] = -df['Price_Change'].where(df['Price_Change'] < 0, 0)
        
        # Calculate initial average gain and loss (first 14 periods)
        initial_avg_gain = df['Gain'].iloc[1:15].mean()
        initial_avg_loss = df['Loss'].iloc[1:15].mean()
        
        # Initialize RSI array
        rsi_values = [None] * len(df)
        
        # Calculate RSI for the first valid period
        if initial_avg_loss != 0:
            rs = initial_avg_gain / initial_avg_loss
            rsi_values[14] = 100 - (100 / (1 + rs))
        else:
            rsi_values[14] = 100
        
        # Calculate RSI for remaining periods using Wilder's smoothing
        for i in range(15, len(df)):
            avg_gain = (initial_avg_gain * 13 + df['Gain'].iloc[i]) / 14
            avg_loss = (initial_avg_loss * 13 + df['Loss'].iloc[i]) / 14
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))
            else:
                rsi_values[i] = 100
            
            # Update averages for next iteration
            initial_avg_gain = avg_gain
            initial_avg_loss = avg_loss
        
        df['RSI14'] = rsi_values
        
        # Clean up temporary columns
        df.drop(['Price_Change', 'Gain', 'Loss'], axis=1, inplace=True)
        
        return df
    
    def calculate_timeframe_seconds(self, interval: str) -> int:
        """
        Convert timeframe interval to seconds.
        
        Args:
            interval: Timeframe string (e.g., '1m', '5m', '1h')
            
        Returns:
            Number of seconds in the interval
        """
        mapping = {
            '1m': 60,
            '5m': 5 * 60,
            '15m': 15 * 60,
            '30m': 30 * 60,
            '60m': 60 * 60,
            '4h': 4 * 60 * 60,
            '1d': 24 * 60 * 60,
            '1W': 7 * 24 * 60 * 60,
            '1M': 30 * 24 * 60 * 60,
        }
        return mapping.get(interval, 60)
    
    def calculate_time_difference(self, time_of_screenshot: str, df: pd.DataFrame) -> pd.Timedelta:
        """
        Calculate time difference between screenshot time and latest data.
        
        Args:
            time_of_screenshot: Screenshot timestamp string
            df: Market data DataFrame
            
        Returns:
            Time difference as pandas Timedelta
        """
        from datetime import datetime
        
        time_of_screenshot = datetime.strptime(time_of_screenshot, "%Y-%m-%d %H:%M")
        df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
        
        # Check if time_of_screenshot is within the dataframe's time range
        min_time = df['Open_time'].min()
        max_time = df['Open_time'].max()
        
        if time_of_screenshot > max_time:
            return None
        
        df['Time_difference'] = df['Open_time'] - time_of_screenshot
        
        # Find the closest row to the screenshot time
        df['Abs_time_difference'] = abs(df['Time_difference'])
        closest_idx = df['Abs_time_difference'].idxmin()
        rows_back = len(df) - 1 - closest_idx
        
        time_diff = df['Time_difference'].iloc[-1]
        
        return time_diff
