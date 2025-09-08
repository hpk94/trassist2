from typing import Optional

import pandas as pd


def calculate_rsi14(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI14 indicator using pandas and Wilder's smoothing.

    Requires at least 15 candles to produce the first value at index 14.
    Leaves rows before that as None.
    """
    if df is None or df.empty or len(df) < 15:
        return df

    df['Price_Change'] = df['Close'].diff()
    df['Gain'] = df['Price_Change'].where(df['Price_Change'] > 0, 0)
    df['Loss'] = -df['Price_Change'].where(df['Price_Change'] < 0, 0)

    initial_avg_gain = df['Gain'].iloc[1:15].mean()
    initial_avg_loss = df['Loss'].iloc[1:15].mean()

    rsi_values = [None] * len(df)

    if initial_avg_loss != 0:
        rs = initial_avg_gain / initial_avg_loss
        rsi_values[14] = 100 - (100 / (1 + rs))
    else:
        rsi_values[14] = 100

    for i in range(15, len(df)):
        avg_gain = (initial_avg_gain * 13 + df['Gain'].iloc[i]) / 14
        avg_loss = (initial_avg_loss * 13 + df['Loss'].iloc[i]) / 14

        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))
        else:
            rsi_values[i] = 100

        initial_avg_gain = avg_gain
        initial_avg_loss = avg_loss

    df['RSI14'] = rsi_values
    df.drop(['Price_Change', 'Gain', 'Loss'], axis=1, inplace=True)
    return df


def calculate_macd12_26_9(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate MACD (12, 26, 9) with EMA smoothing.

    Produces columns: MACD_Line, MACD_Signal, MACD_Histogram.
    """
    if df is None or df.empty or len(df) < 27:
        return df

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']
    df.drop(['EMA12', 'EMA26'], axis=1, inplace=True)
    return df


def calculate_stoch14_3_3(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Stochastic Oscillator (14, 3, 3).

    Produces columns: STOCH_K, STOCH_D.
    """
    if df is None or df.empty or len(df) < 17:
        return df

    df['Lowest_Low_14'] = df['Low'].rolling(window=14, min_periods=14).min()
    df['Highest_High_14'] = df['High'].rolling(window=14, min_periods=14).max()
    df['Stoch_K_Raw'] = 100 * (df['Close'] - df['Lowest_Low_14']) / (df['Highest_High_14'] - df['Lowest_Low_14'])
    df['STOCH_K'] = df['Stoch_K_Raw'].rolling(window=3, min_periods=3).mean()
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3, min_periods=3).mean()
    df.drop(['Lowest_Low_14', 'Highest_High_14', 'Stoch_K_Raw'], axis=1, inplace=True)
    return df


def calculate_bb20_2(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Bollinger Bands (20, 2).

    Produces columns: BB_Middle, BB_Upper, BB_Lower, BB_PercentB, BB_Bandwidth.
    """
    if df is None or df.empty or len(df) < 20:
        return df

    df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=20).mean()
    df['BB_StdDev'] = df['Close'].rolling(window=20, min_periods=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_StdDev'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_StdDev'])
    df['BB_PercentB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df.drop(['BB_StdDev'], axis=1, inplace=True)
    return df


def calculate_atr14(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ATR(14) using Wilder's method.

    Produces column: ATR14.
    """
    if df is None or df.empty or len(df) < 15:
        return df

    prev_close = df['Close'].shift(1)
    high_low = (df['High'] - df['Low']).abs()
    high_prev_close = (df['High'] - prev_close).abs()
    low_prev_close = (df['Low'] - prev_close).abs()

    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

    atr_values = [None] * len(df)
    initial_atr = tr.iloc[1:15].mean()
    atr_values[14] = initial_atr

    prev_atr = initial_atr
    for i in range(15, len(df)):
        current_tr = tr.iloc[i]
        atr = (prev_atr * 13 + current_tr) / 14
        atr_values[i] = atr
        prev_atr = atr

    df['ATR14'] = atr_values
    return df


__all__ = [
    'calculate_rsi14',
    'calculate_macd12_26_9',
    'calculate_stoch14_3_3',
    'calculate_bb20_2',
    'calculate_atr14',
]


