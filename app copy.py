import os
import json
import base64
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from prompt import OPENAI_VISION_PROMPT
import pandas as pd
from datetime import datetime

from pymexc import spot, futures

# Load environment variables
load_dotenv()

test_image = "BTCUSDT.P_2025-09-02_19-28-40_20635.png"

def analyze_trading_chart(image_path: str) -> dict:
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {"role": "system", "content": OPENAI_VISION_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": "Analyze this trading chart."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# llm_output = {'symbol': 'BTCUSDT', 'timeframe': '1m', 'time_of_screenshot': '2025-08-31 20:01', 'trend_direction': 'bearish', 'support_resistance': {'support': 108, 'resistance': 109240}, 'technical_indicators': {'rsi_14': {'value': 39.16, 'status': 'neutral', 'signal': 'no_signal'}, 'macd': {'macd_line': None, 'signal_line': None, 'histogram': None, 'signal': 'not_available'}, 'bollinger_bands': {'upper': None, 'middle': None, 'lower': None, 'price_position': 'not_available', 'bandwidth': 'not_available'}, 'stochastic': {'k_percent': None, 'd_percent': None, 'signal': 'not_available'}, 'volume': {'current': 19, 'average': None, 'ratio': 'not_available', 'trend': 'increasing'}, 'atr': {'value': None, 'period': None, 'volatility': 'not_available'}}, 'pattern_analysis': [{'pattern': 'bearish_channel', 'confidence': 0.75}], 'validity_assessment': {'alignment_score': 0.65, 'notes': 'Price is in a bearish channel and approaching the lower Fibonacci level.'}, 'opening_signal': {'direction': 'short', 'scope': {'candle_indices': [0, 1, 2], 'lookback_seconds': 180}, 'checklist': [{'id': 'rsi_below_50', 'type': 'indicator_threshold', 'indicator': 'RSI(14)', 'timeframe': '1m', 'comparator': '<', 'value': 50.0, 'observed_on_candle': 0}, {'id': 'price_below_bb_middle', 'type': 'price_level', 'level': 'bollinger_middle', 'comparator': '<', 'value': None, 'observed_on_candle': 0}, {'id': 'volume_above_average', 'type': 'volume_threshold', 'lookback_candles': 3, 'comparator': '>', 'value': 1.0, 'baseline': 'average_volume'}, {'id': 'current_price_below_fib_0_236', 'type': 'price_level', 'level': 'fib_0_236', 'comparator': '<', 'value': 109103.1, 'observed_on_candle': 0}, {'id': 'bearish_pattern', 'type': 'candle_pattern', 'pattern': 'bearish_channel', 'candle_index': 0}], 'invalidation': [{'id': 'close_above_fib_0_382', 'type': 'price_breach', 'level': 109044.6, 'comparator': '>'}, {'id': 'rsi_breach_70', 'type': 'indicator_threshold', 'indicator': 'RSI(14)', 'timeframe': '1m', 'comparator': '>=', 'value': 70.0}], 'is_met': False}, 'risk_management': {'stop_loss': {'price': 109, 'basis': 'above_fib_0_382', 'distance_ticks': 120}, 'take_profit': [{'price': 108000, 'basis': 'previous_support', 'rr': 1.5}, {'price': 107800, 'basis': 'next_fibonacci_support', 'rr': 2.0}]}, 'summary_actions': ['Watch for confirmation below the 0.236 Fibonacci level.', 'Monitor RSI to remain below 50 for potential short entry.', 'Ensure volume is sustaining the bearish move.', 'Invalidation if price closes above the 0.382 Fibonacci level.'], 'improvements': 'Consider adding MACD for additional confirmation on momentum.'}
# llm_output = {'symbol': 'BTCUSDT', 'timeframe': '1m', 'time_of_screenshot': '2025-09-01 20:01', 'trend_direction': 'bearish', 'support_resistance': {'support': 108, 'technical_indicators': {'RSI14': {'value': 33.11, 'status': 'oversold', 'signal': 'oversold'}, 'MACD12_26_9': {'macd_line': -0.5, 'signal_line': -0.3, 'histogram': -0.2, 'signal': 'bearish_crossover'}, 'BB20_2': {'upper': 109, 'middle': 108.5, 'lower': 108, 'price_position': 'below', 'bandwidth': 'narrow'}, 'STOCH14_3_3': {'k_percent': 20, 'd_percent': 22, 'signal': 'oversold'}, 'VOLUME': {'current': 19, 'average': 25, 'ratio': 0.76, 'trend': 'decreasing'}, 'ATR14': {'value': 1.5, 'period': 14, 'volatility': 'low'}}, 'pattern_analysis': [{'pattern': 'bearish_engulfing', 'candle_index': 0, 'confidence': 0.85}], 'validity_assessment': {'alignment_score': 0.72, 'notes': 'Overall bearish movement supported by bearish engulfing pattern, RSI indicating oversold conditions.'}, 'opening_signal': {'direction': 'short', 'scope': {'candle_indices': [0, 1, 2], 'lookback_seconds': 180}, 'checklist': [{'id': 'rsi_oversold', 'type': 'indicator_threshold', 'indicator': 'RSI14', 'timeframe': '1m', 'comparator': '<=', 'value': 30.0, 'observed_on_candle': 0, 'technical_indicator': True, 'category': 'technical_indicator'}, {'id': 'macd_bearish', 'type': 'indicator_crossover', 'indicator': 'MACD12_26_9', 'timeframe': '1m', 'condition': 'macd_line < signal_line', 'observed_on_candle': 0, 'technical_indicator': True, 'category': 'technical_indicator'}, {'id': 'price_below_bb_middle', 'type': 'price_level', 'level': 'bollinger_middle', 'comparator': '<', 'value': 108.5, 'observed_on_candle': 0, 'technical_indicator': False, 'category': 'price_level'}, {'id': 'volume_below_average', 'type': 'volume_threshold', 'lookback_candles': 3, 'comparator': '<', 'value': 1.0, 'baseline': 'average_volume', 'technical_indicator': False, 'category': 'volume_analysis'}, {'id': 'bearish_engulfing', 'type': 'candle_pattern', 'pattern': 'bearish_engulfing', 'candle_index': 0, 'technical_indicator': False, 'category': 'candle_pattern'}, {'id': 'stochastic_oversold', 'type': 'indicator_threshold', 'indicator': 'STOCH14_3_3', 'timeframe': '1m', 'comparator': '<=', 'value': 20.0, 'observed_on_candle': 0, 'technical_indicator': True, 'category': 'technical_indicator'}], 'invalidation': [{'id': 'close_above_bollinger_middle', 'type': 'price_breach', 'level': 'bollinger_middle', 'comparator': '>'}, {'id': 'rsi_overbought', 'type': 'indicator_threshold', 'indicator': 'RSI14', 'timeframe': '1m', 'comparator': '>=', 'value': 70.0}, {'id': 'macd_bullish_crossover', 'type': 'indicator_crossover', 'indicator': 'MACD12_26_9', 'timeframe': '1m', 'condition': 'macd_line > signal_line'}, {'id': 'price_above_bb_upper', 'type': 'price_level', 'level': 'bollinger_upper', 'comparator': '>=', 'value': 109}, {'id': 'two_bull_candles_large', 'type': 'sequence', 'count': 2, 'direction': 'bullish', 'atr_multiple': 1.0}], 'is_met': False}, 'risk_management': {'stop_loss': {'price': 109.0, 'basis': 'above_bearish_engulfing', 'distance_ticks': 15}, 'take_profit': [{'price': 108.5, 'basis': 'near_support', 'rr': 1.5}, {'price': 108.2, 'basis': 'near_fib_0.5', 'rr': 2.0}]}, 'summary_actions': ['Wait for confirmation of bearish sentiment before entering', 'Look for MACD bearish crossover for further confirmation', 'Enter short if all checklist items are true', 'Invalidate on RSI14 >= 70 or close above middle Bollinger Band'], 'improvements': 'Consider adding volume confirmation and ensuring adherence to stop-loss placement below recent highs.'}, 'resistance': 109240}
llm_output = {
    'symbol': 'BTCUSDT',
    'timeframe': '1m',
    'time_of_screenshot': '2025-09-01 20:01',
    'trend_direction': 'bearish',

    'support_resistance': {
        'support': 108,
    },
    'resistance': 109240,  # <- note: placed outside support_resistance, may be a nesting issue

    'technical_indicators': {
        'RSI14': {
            'value': 33.11,
            'status': 'oversold',
            'signal': 'oversold'
        },
        'MACD12_26_9': {
            'macd_line': -0.5,
            'signal_line': -0.3,
            'histogram': -0.2,
            'signal': 'bearish_crossover'
        },
        'BB20_2': {
            'upper': 109,
            'middle': 108.5,
            'lower': 108,
            'price_position': 'below',
            'bandwidth': 'narrow'
        },
        'STOCH14_3_3': {
            'k_percent': 20,
            'd_percent': 22,
            'signal': 'oversold'
        },
        'VOLUME': {
            'current': 19,
            'average': 25,
            'ratio': 0.76,
            'trend': 'decreasing'
        },
        'ATR14': {
            'value': 1.5,
            'period': 14,
            'volatility': 'low'
        }
    },

    'pattern_analysis': [
        {
            'pattern': 'bearish_engulfing',
            'candle_index': 0,
            'confidence': 0.85
        }
    ],

    'validity_assessment': {
        'alignment_score': 0.72,
        'notes': 'Overall bearish movement supported by bearish engulfing pattern, RSI indicating oversold conditions.'
    },

    'opening_signal': {
        'direction': 'short',
        'scope': {
            'candle_indices': [0, 1, 2],
            'lookback_seconds': 180
        },
        'checklist': [
            {
                'id': 'rsi_oversold',
                'type': 'indicator_threshold',
                'indicator': 'RSI14',
                'timeframe': '1m',
                'comparator': '<=',
                'value': 30.0,
                'observed_on_candle': 0,
                'technical_indicator': True,
                'category': 'technical_indicator'
            },
            {
                'id': 'macd_bearish',
                'type': 'indicator_crossover',
                'indicator': 'MACD12_26_9',
                'timeframe': '1m',
                'condition': 'macd_line < signal_line',
                'observed_on_candle': 0,
                'technical_indicator': True,
                'category': 'technical_indicator'
            },
            {
                'id': 'price_below_bb_middle',
                'type': 'price_level',
                'level': 'bollinger_middle',
                'comparator': '<',
                'value': 108.5,
                'observed_on_candle': 0,
                'technical_indicator': False,
                'category': 'price_level'
            },
            {
                'id': 'volume_below_average',
                'type': 'volume_threshold',
                'lookback_candles': 3,
                'comparator': '<',
                'value': 1.0,
                'baseline': 'average_volume',
                'technical_indicator': False,
                'category': 'volume_analysis'
            },
            {
                'id': 'bearish_engulfing',
                'type': 'candle_pattern',
                'pattern': 'bearish_engulfing',
                'candle_index': 0,
                'technical_indicator': False,
                'category': 'candle_pattern'
            },
            {
                'id': 'stochastic_oversold',
                'type': 'indicator_threshold',
                'indicator': 'STOCH14_3_3',
                'timeframe': '1m',
                'comparator': '<=',
                'value': 20.0,
                'observed_on_candle': 0,
                'technical_indicator': True,
                'category': 'technical_indicator'
            }
        ],
        'invalidation': [
            {
                'id': 'close_above_bollinger_middle',
                'type': 'price_breach',
                'level': 'bollinger_middle',
                'comparator': '>'
            },
            {
                'id': 'rsi_overbought',
                'type': 'indicator_threshold',
                'indicator': 'RSI14',
                'timeframe': '1m',
                'comparator': '>=',
                'value': 70.0
            },
            {
                'id': 'macd_bullish_crossover',
                'type': 'indicator_crossover',
                'indicator': 'MACD12_26_9',
                'timeframe': '1m',
                'condition': 'macd_line > signal_line'
            },
            {
                'id': 'price_above_bb_upper',
                'type': 'price_level',
                'level': 'bollinger_upper',
                'comparator': '>=',
                'value': 109
            },
            {
                'id': 'two_bull_candles_large',
                'type': 'sequence',
                'count': 2,
                'direction': 'bullish',
                'atr_multiple': 1.0
            }
        ],
        'is_met': False
    },

    'risk_management': {
        'stop_loss': {
            'price': 109.0,
            'basis': 'above_bearish_engulfing',
            'distance_ticks': 15
        },
        'take_profit': [
            {
                'price': 108.5,
                'basis': 'near_support',
                'rr': 1.5
            },
            {
                'price': 108.2,
                'basis': 'near_fib_0.5',
                'rr': 2.0
            }
        ]
    },

    'summary_actions': [
        'Wait for confirmation of bearish sentiment before entering',
        'Look for MACD bearish crossover for further confirmation',
        'Enter short if all checklist items are true',
        'Invalidate on RSI14 >= 70 or close above middle Bollinger Band'
    ],

    'improvements': 'Consider adding volume confirmation and ensuring adherence to stop-loss placement below recent highs.'
}

# print(analyze_trading_chart(test_image))

# llm_output = analyze_trading_chart(test_image)

# Create output directory if it doesn't exist
output_dir = "llm_outputs"
os.makedirs(output_dir, exist_ok=True)

# Save LLM output to JSON file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"llm_output_{timestamp}.json"
output_path = os.path.join(output_dir, output_filename)

with open(output_path, 'w') as f:
    json.dump(llm_output, f, indent=2, default=str)

symbol = llm_output['symbol']
timeframe = llm_output['timeframe']
time_of_screenshot = llm_output['time_of_screenshot']

# Validate timeframe against MEXC supported intervals
valid_intervals = ['1m', '5m', '15m', '30m', '60m', '4h', '1d', '1W', '1M']
if timeframe not in valid_intervals:
    print(f"Warning: {timeframe} is not a valid MEXC interval. Using 1m as default.")
    timeframe = '1m'

print(f"symbol: {symbol}, timeframe: {timeframe}, time_of_screenshot: {time_of_screenshot}")

# initialize HTTP client for authenticated endpoints
spot_client = spot.HTTP(api_key = os.getenv("MEXC_API_KEY"), api_secret = os.getenv("MEXC_API_SECRET"))
# initialize public HTTP client for market data (no auth required)
public_spot_client = spot.HTTP()
# initialize WebSocket client
ws_spot_client = spot.WebSocket(api_key = os.getenv("MEXC_API_KEY"), api_secret = os.getenv("MEXC_API_SECRET"))

# make http request to api

def fetch_market_data(symbol, timeframe):
    """Fetch raw klines data from MEXC API"""
    try:
        klines = public_spot_client.klines(
            symbol=symbol,
            interval=timeframe,
            limit=100
        )

        # print(f"Successfully retrieved {len(klines)} klines for {symbol}")
        # print(f"Latest kline: {klines[-1] if klines else 'No data'}")
    
    except Exception as e:
        print(f"Error retrieving klines: {e}")
        klines = []

    return klines

def fetch_market_dataframe(symbol, timeframe):
    """Fetch market data and return as processed DataFrame"""
    klines = fetch_market_data(symbol, timeframe)
    
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

def calculate_rsi14(df):
    """Calculate RSI14 indicator manually using pandas"""
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

# Use the new function for cleaner code
df = fetch_market_dataframe(symbol, timeframe)

# Calculate RSI14
df = calculate_rsi14(df)

# Display the DataFrame with RSI14
print(df[['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI14']].tail(10))

# Get the latest RSI14 value and signal
if not df.empty and 'RSI14' in df.columns:
    latest_rsi = df['RSI14'].iloc[-1]
    
    print(f"\nLatest RSI14 Analysis:")
    print(f"RSI14 Value: {latest_rsi:.2f}")
    
else:
    print("No RSI14 data available")


def list_of_indicator_checker():
    technical_indicator_list=[]
    for i in llm_output['opening_signal']['checklist']:
        if i['technical_indicator'] == True:
            technical_indicator_list.append(i)
    return technical_indicator_list
  

def indicator_checker(df):
    technical_indicator_list = list_of_indicator_checker()
    all_conditions_met = True
    conditions_met_count = 0
    total_conditions = len(technical_indicator_list)

    for i in technical_indicator_list:
        indicator_name = i['indicator'] 
        indicator_type = i['type']
        condition_met = False
        
        if indicator_type == 'indicator_threshold':
            # Handle threshold comparisons (RSI, Stochastic, etc.)
            indicator_comparator = i['comparator']
            indicator_value = i['value']
            
            # Check if the indicator column exists in the DataFrame
            if indicator_name not in df.columns:
                print(f"{indicator_name}: Column not found in DataFrame - skipping")
                condition_met = False
            else:
                current_indicator_value = df[indicator_name].iloc[-1]
                
                # Perform the comparison based on the comparator
                if indicator_comparator == '<=':
                    condition_met = current_indicator_value <= indicator_value
                elif indicator_comparator == '>=':
                    condition_met = current_indicator_value >= indicator_value
                elif indicator_comparator == '<':
                    condition_met = current_indicator_value < indicator_value
                elif indicator_comparator == '>':
                    condition_met = current_indicator_value > indicator_value
                elif indicator_comparator == '==':
                    condition_met = current_indicator_value == indicator_value
                elif indicator_comparator == '!=':
                    condition_met = current_indicator_value != indicator_value
                else:
                    print(f"Unknown comparator: {indicator_comparator}")
                    condition_met = False
                
                # Print the comparison for debugging
                print("--------------------------------")
                print(f"indicator_name: {indicator_name}, indicator_comparator: {indicator_comparator}, indicator_value: {indicator_value}, current_indicator_value: {current_indicator_value}, condition_met: {condition_met}")
                print(f"{indicator_name}: {current_indicator_value:.2f} {indicator_comparator} {indicator_value} = {condition_met}")
                print("--------------------------------")
            
        elif indicator_type == 'indicator_crossover':
            # Handle crossover conditions (MACD, etc.)
            condition = i['condition']
            print(f"{indicator_name}: Crossover condition '{condition}' - Not implemented yet")
            # For now, skip crossover conditions as they need more complex logic
            condition_met = True  # Skip for now
            
        else:
            print(f"Unknown indicator type: {indicator_type}")
            condition_met = False
        
        # Count conditions that are met
        if condition_met:
            conditions_met_count += 1
        else:
            all_conditions_met = False
        
    # Print the score
    print(f"Score: {conditions_met_count}/{total_conditions}")
    
    return all_conditions_met

print(indicator_checker(df))

def invalidation_checker(df):
    """Check invalidation conditions from the LLM output"""
    invalidation_conditions = llm_output['opening_signal']['invalidation']
    invalidation_triggered = False
    triggered_conditions = []
    
    print("\n=== INVALIDATION CHECK ===")
    
    for condition in invalidation_conditions:
        condition_id = condition['id']
        condition_type = condition['type']
        condition_met = False
        
        if condition_type == 'price_breach':
            # Handle price breach conditions
            level = condition.get('level')
            comparator = condition.get('comparator', '>')
            
            if level == 'bollinger_middle':
                # Get Bollinger Band middle value from technical indicators
                bb_middle = llm_output['technical_indicators']['BB20_2']['middle']
                current_price = df['Close'].iloc[-1]
                
                if comparator == '>':
                    condition_met = current_price > bb_middle
                elif comparator == '>=':
                    condition_met = current_price >= bb_middle
                elif comparator == '<':
                    condition_met = current_price < bb_middle
                elif comparator == '<=':
                    condition_met = current_price <= bb_middle
                
                print(f"Price Breach Check - {condition_id}:")
                print(f"  Current Price: {current_price:.2f}")
                print(f"  Bollinger Middle: {bb_middle:.2f}")
                print(f"  Condition: {current_price:.2f} {comparator} {bb_middle:.2f} = {condition_met}")
                
            elif isinstance(level, (int, float)):
                # Direct price level comparison
                current_price = df['Close'].iloc[-1]
                
                if comparator == '>':
                    condition_met = current_price > level
                elif comparator == '>=':
                    condition_met = current_price >= level
                elif comparator == '<':
                    condition_met = current_price < level
                elif comparator == '<=':
                    condition_met = current_price <= level
                
                print(f"Price Breach Check - {condition_id}:")
                print(f"  Current Price: {current_price:.2f}")
                print(f"  Level: {level:.2f}")
                print(f"  Condition: {current_price:.2f} {comparator} {level:.2f} = {condition_met}")
        
        elif condition_type == 'indicator_threshold':
            # Handle indicator threshold conditions
            indicator_name = condition['indicator']
            comparator = condition['comparator']
            threshold_value = condition['value']
            
            if indicator_name in df.columns:
                current_value = df[indicator_name].iloc[-1]
                
                if comparator == '>=':
                    condition_met = current_value >= threshold_value
                elif comparator == '>':
                    condition_met = current_value > threshold_value
                elif comparator == '<=':
                    condition_met = current_value <= threshold_value
                elif comparator == '<':
                    condition_met = current_value < threshold_value
                elif comparator == '==':
                    condition_met = current_value == threshold_value
                elif comparator == '!=':
                    condition_met = current_value != threshold_value
                
                print(f"Indicator Threshold Check - {condition_id}:")
                print(f"  {indicator_name}: {current_value:.2f}")
                print(f"  Threshold: {threshold_value:.2f}")
                print(f"  Condition: {current_value:.2f} {comparator} {threshold_value:.2f} = {condition_met}")
            else:
                print(f"Indicator Threshold Check - {condition_id}:")
                print(f"  {indicator_name}: Column not found in DataFrame")
                condition_met = False
        
        elif condition_type == 'indicator_crossover':
            # Handle indicator crossover conditions
            indicator_name = condition['indicator']
            crossover_condition = condition['condition']
            
            print(f"Indicator Crossover Check - {condition_id}:")
            print(f"  {indicator_name}: Crossover condition '{crossover_condition}' - Not implemented yet")
            # For now, skip crossover conditions as they need more complex logic
            condition_met = False
        
        elif condition_type == 'price_level':
            # Handle price level conditions
            level_type = condition['level']
            comparator = condition['comparator']
            level_value = condition.get('value')
            
            current_price = df['Close'].iloc[-1]
            
            if level_type == 'bollinger_upper':
                bb_upper = llm_output['technical_indicators']['BB20_2']['upper']
                if comparator == '>=':
                    condition_met = current_price >= bb_upper
                elif comparator == '>':
                    condition_met = current_price > bb_upper
                elif comparator == '<=':
                    condition_met = current_price <= bb_upper
                elif comparator == '<':
                    condition_met = current_price < bb_upper
                
                print(f"Price Level Check - {condition_id}:")
                print(f"  Current Price: {current_price:.2f}")
                print(f"  Bollinger Upper: {bb_upper:.2f}")
                print(f"  Condition: {current_price:.2f} {comparator} {bb_upper:.2f} = {condition_met}")
            
            elif level_value is not None:
                if comparator == '>=':
                    condition_met = current_price >= level_value
                elif comparator == '>':
                    condition_met = current_price > level_value
                elif comparator == '<=':
                    condition_met = current_price <= level_value
                elif comparator == '<':
                    condition_met = current_price < level_value
                
                print(f"Price Level Check - {condition_id}:")
                print(f"  Current Price: {current_price:.2f}")
                print(f"  Level: {level_value:.2f}")
                print(f"  Condition: {current_price:.2f} {comparator} {level_value:.2f} = {condition_met}")
        
        elif condition_type == 'sequence':
            # Handle sequence conditions (e.g., two bullish candles)
            count = condition.get('count', 2)
            direction = condition.get('direction', 'bullish')
            atr_multiple = condition.get('atr_multiple', 1.0)
            
            print(f"Sequence Check - {condition_id}:")
            print(f"  Looking for {count} consecutive {direction} candles with ATR multiple {atr_multiple}")
            print(f"  Sequence conditions - Not fully implemented yet")
            # This would require more complex logic to check candle patterns
            condition_met = False
        
        else:
            print(f"Unknown invalidation condition type: {condition_type}")
            condition_met = False
        
        # Check if this invalidation condition is triggered
        if condition_met:
            invalidation_triggered = True
            triggered_conditions.append(condition_id)
            print(f"  ⚠️  INVALIDATION TRIGGERED: {condition_id}")
        else:
            print(f"  ✅ Invalidation condition not met: {condition_id}")
        
        print("  " + "-" * 50)
    
    print(f"\n=== INVALIDATION SUMMARY ===")
    if invalidation_triggered:
        print(f"🚨 SIGNAL INVALIDATED! Triggered conditions: {', '.join(triggered_conditions)}")
        print("The trading signal should be considered invalid and no trade should be taken.")
    else:
        print("✅ No invalidation conditions triggered. Signal remains valid.")
    
    return invalidation_triggered, triggered_conditions

# Test the invalidation checker
invalidation_result = invalidation_checker(df)

def validate_trading_signal(df):
    """Comprehensive trading signal validation combining checklist and invalidation checks"""
    print("\n" + "="*60)
    print("COMPREHENSIVE TRADING SIGNAL VALIDATION")
    print("="*60)
    
    # Check if all checklist conditions are met
    checklist_passed = indicator_checker(df)
    
    # Check if any invalidation conditions are triggered
    invalidation_triggered, triggered_conditions = invalidation_checker(df)
    
    print("\n" + "="*60)
    print("FINAL TRADING DECISION")
    print("="*60)
    
    if invalidation_triggered:
        print("🚨 TRADE SIGNAL: INVALIDATED")
        print(f"   Reason: Invalidation conditions triggered: {', '.join(triggered_conditions)}")
        print("   Action: DO NOT ENTER TRADE")
        return False, "invalidated", triggered_conditions
    elif checklist_passed:
        print("✅ TRADE SIGNAL: VALID")
        print("   All checklist conditions met and no invalidation triggered")
        print("   Action: CONSIDER ENTERING TRADE")
        return True, "valid", []
    else:
        print("⏳ TRADE SIGNAL: PENDING")
        print("   Checklist conditions not yet met")
        print("   Action: WAIT FOR CONDITIONS TO BE MET")
        return False, "pending", []

# Test the comprehensive validation
signal_valid, signal_status, triggered_conditions = validate_trading_signal(df)

def calculate_time_difference(time_of_screenshot, df):
    time_of_screenshot = datetime.strptime(time_of_screenshot, "%Y-%m-%d %H:%M")
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
    
    # Check if time_of_screenshot is within the dataframe's time range
    min_time = df['Open_time'].min()
    max_time = df['Open_time'].max()
    
    if time_of_screenshot < min_time:
        print(f"Warning: time_of_screenshot ({time_of_screenshot}) is before the earliest data point ({min_time})")
    elif time_of_screenshot > max_time:
        print(f"ERROR: time_of_screenshot ({time_of_screenshot}) is in the future from the latest data point ({max_time})")
        print("This is invalid - screenshots cannot be from the future!")
        return None
    else:
        print(f"Info: time_of_screenshot ({time_of_screenshot}) is within the dataframe's time range")
    
    df['Time_difference'] = df['Open_time'] - time_of_screenshot
    
    # Find the closest row to the screenshot time
    df['Abs_time_difference'] = abs(df['Time_difference'])
    closest_idx = df['Abs_time_difference'].idxmin()
    rows_back = len(df) - 1 - closest_idx
    
    time_diff = df['Time_difference'].iloc[-1]
    
    # Make the time difference more readable
    if time_diff.total_seconds() < 0:
        # Screenshot is in the future
        future_diff = abs(time_diff)
        print(f"Screenshot is {future_diff} in the future from latest data")
    else:
        # Screenshot is in the past
        print(f"Screenshot is {time_diff} in the past from latest data")
    
    print(f"Closest data point is {rows_back} rows back from the latest")
    print(f"Closest data point time: {df['Open_time'].iloc[closest_idx]}")
        
    return time_diff


print(calculate_time_difference(llm_output['time_of_screenshot'], df))

print(df)