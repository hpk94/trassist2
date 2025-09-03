import os
import json
import base64
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from prompt import OPENAI_VISION_PROMPT, TRADE_GATE_PROMPT
import pandas as pd
from datetime import datetime
import time

from pymexc import spot, futures

# Load environment variables
load_dotenv()

test_image = "BTCUSDT.P_2025-09-02_22-55-40_545b4.png"

def analyze_trading_chart(image_path: str) -> dict:
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": OPENAI_VISION_PROMPT},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

def llm_trade_gate_decision(
    base_llm_output: Dict[str, Any],
    market_values: Dict[str, Any],
    checklist_passed: bool,
    invalidation_triggered: bool,
    triggered_conditions: list
) -> Dict[str, Any]:
    client = OpenAI()

    # Prepare concise context for the gate
    gate_context = {
        "llm_snapshot": {
            "symbol": base_llm_output.get("symbol"),
            "timeframe": base_llm_output.get("timeframe"),
            "opening_signal": base_llm_output.get("opening_signal"),
            "risk_management": base_llm_output.get("risk_management"),
        },
        "market_values": {
            "current_price": float(market_values.get("current_price", 0) or 0),
            "current_rsi": float(market_values.get("current_rsi", 0) or 0),
            "current_time": str(market_values.get("current_time")),
        },
        "program_checks": {
            "checklist_passed": bool(checklist_passed),
            "invalidation_triggered": bool(invalidation_triggered),
            "triggered_conditions": triggered_conditions,
        },
    }

    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": TRADE_GATE_PROMPT},
            {"role": "user", "content": json.dumps(gate_context)},
        ],
        response_format={"type": "json_object"},
    )

    try:
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {
            "should_open": False,
            "direction": base_llm_output.get("opening_signal", {}).get("direction", "unknown"),
            "confidence": 0.0,
            "reasons": [f"Gate parsing error: {str(e)}"],
            "warnings": [],
            "execution": {
                "entry_type": "market",
                "entry_price": float(market_values.get("current_price", 0) or 0),
                "stop_loss": 0.0,
                "take_profits": [],
                "risk_reward": 0.0,
                "position_size_note": "n/a"
            },
            "checks": {
                "invalidation_triggered": bool(invalidation_triggered),
                "checklist_score": {"met": 0, "total": 0},
                "context_alignment": "weak"
            }
        }

print("Step 1: Analyzing trading chart with LLM...")
llm_output = analyze_trading_chart(test_image)
print(f"Step 1 Complete: Chart analysis finished for {test_image}")

print("Step 2: Setting up output directory...")
# Create output directory if it doesn't exist
output_dir = "llm_outputs"
os.makedirs(output_dir, exist_ok=True)
print(f"Step 2 Complete: Output directory ready at {output_dir}")

print("Step 3: Saving LLM output to JSON file...")
# Save LLM output to JSON file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"llm_output_{timestamp}.json"
output_path = os.path.join(output_dir, output_filename)

with open(output_path, 'w') as f:
    json.dump(llm_output, f, indent=2, default=str)
print(f"Step 3 Complete: LLM output saved to {output_path}")

print("Step 4: Extracting trading parameters from LLM output...")
symbol = llm_output['symbol']
timeframe = llm_output['timeframe']
time_of_screenshot = llm_output['time_of_screenshot']
print(f"Step 4 Complete: Symbol={symbol}, Timeframe={timeframe}, Screenshot time={time_of_screenshot}")

print("Step 5: Validating timeframe against MEXC supported intervals...")
# Validate timeframe against MEXC supported intervals
valid_intervals = ['1m', '5m', '15m', '30m', '60m', '4h', '1d', '1W', '1M']
if timeframe not in valid_intervals:
    print(f"Warning: {timeframe} is not a valid MEXC interval. Using 1m as default.")
    timeframe = '1m'
print(f"Step 5 Complete: Using timeframe {timeframe}")



print("Step 6: Initializing MEXC API clients...")
# initialize HTTP client for authenticated endpoints
spot_client = spot.HTTP(api_key = os.getenv("MEXC_API_KEY"), api_secret = os.getenv("MEXC_API_SECRET"))
# initialize public HTTP client for market data (no auth required)
public_spot_client = spot.HTTP()
# initialize WebSocket client
ws_spot_client = spot.WebSocket(api_key = os.getenv("MEXC_API_KEY"), api_secret = os.getenv("MEXC_API_SECRET"))
print("Step 6 Complete: MEXC API clients initialized")

# make http request to api

def fetch_market_data(symbol, timeframe):
    """Fetch raw klines data from MEXC API"""
    print(f"      Fetching market data for {symbol} on {timeframe} timeframe...")
    try:
        klines = public_spot_client.klines(
            symbol=symbol,
            interval=timeframe,
            limit=100
        )
        print(f"      Successfully fetched {len(klines)} klines")
    
    except Exception as e:
        print(f"      Error retrieving klines: {e}")
        klines = []

    return klines

def fetch_market_dataframe(symbol, timeframe):
    """Fetch market data and return as processed DataFrame"""
    print(f"      Processing market data into DataFrame...")
    klines = fetch_market_data(symbol, timeframe)
    
    if not klines:
        print("      No klines data available, returning empty DataFrame")
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
    
    print(f"      DataFrame created with {len(df)} rows")
    return df

def calculate_rsi14(df):
    """Calculate RSI14 indicator manually using pandas"""
    print(f"      Calculating RSI14 indicator...")
    if df.empty or len(df) < 15:  # Need 15 candles to calculate 14-period RSI
        print("      Warning: Not enough data points for RSI14 calculation (need at least 15 candles)")
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
    
    print(f"      RSI14 calculation complete")
    return df

print("Step 7: Fetching market data...")
# Use the new function for cleaner code
df = fetch_market_dataframe(symbol, timeframe)
print(f"Step 7 Complete: Market data fetched, {len(df)} candles retrieved")

print("Step 8: Calculating RSI14 indicator...")
# Calculate RSI14
df = calculate_rsi14(df)
print("Step 8 Complete: RSI14 calculation finished")

print("Step 9: Extracting latest market indicators...")
# Get the latest RSI14 value and signal
if not df.empty and 'RSI14' in df.columns:
    latest_rsi = df['RSI14'].iloc[-1]
    print(f"Step 9 Complete: Latest RSI14 = {latest_rsi}")
else:
    print("Step 9 Complete: No RSI14 data available")

# Common utility functions for condition checking
def evaluate_comparison(current_value, comparator, target_value):
    """Evaluate a comparison between current value and target value"""
    if comparator == '<=':
        return current_value <= target_value
    elif comparator == '>=':
        return current_value >= target_value
    elif comparator == '<':
        return current_value < target_value
    elif comparator == '>':
        return current_value > target_value
    elif comparator == '==':
        return current_value == target_value
    elif comparator == '!=':
        return current_value != target_value
    else:
        return False

def check_indicator_threshold(df, condition):
    """Check indicator threshold conditions"""
    indicator_name = condition['indicator']
    comparator = condition['comparator']
    threshold_value = condition['value']
    
    if indicator_name not in df.columns:

        return False
    
    current_value = df[indicator_name].iloc[-1]
    condition_met = evaluate_comparison(current_value, comparator, threshold_value)
    

    return condition_met

def check_price_level(df, condition):
    """Check price level conditions"""
    level_type = condition.get('level')
    comparator = condition.get('comparator', '>')
    level_value = condition.get('value')
    
    current_price = df['Close'].iloc[-1]
    
    if level_type == 'bollinger_middle':
        bb_middle = llm_output['technical_indicators']['BB20_2']['middle']
        condition_met = evaluate_comparison(current_price, comparator, bb_middle)

        return condition_met
    elif level_type == 'bollinger_upper':
        bb_upper = llm_output['technical_indicators']['BB20_2']['upper']
        condition_met = evaluate_comparison(current_price, comparator, bb_upper)

        return condition_met
    elif level_type == 'direct' or level_value is not None:
        # Handle direct price level comparison
        if level_type == 'direct':
            target_value = level_value
        else:
            target_value = level_value
        condition_met = evaluate_comparison(current_price, comparator, target_value)

        return condition_met
    else:

        return False

def check_indicator_crossover(df, condition):
    """Check indicator crossover conditions"""
    indicator_name = condition['indicator']
    crossover_condition = condition['condition']
    

    # For now, skip crossover conditions as they need more complex logic
    return False

def check_sequence_condition(df, condition):
    """Check sequence conditions (e.g., consecutive candles)"""
    count = condition.get('count', 2)
    direction = condition.get('direction', 'bullish')
    atr_multiple = condition.get('atr_multiple', 1.0)
    

    return False

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
            condition_met = check_indicator_threshold(df, i)
        elif indicator_type == 'indicator_crossover':
            condition_met = check_indicator_crossover(df, i)
        else:

            condition_met = False
        

        
        # Count conditions that are met
        if condition_met:
            conditions_met_count += 1
        else:
            all_conditions_met = False
        

    
    return all_conditions_met



def invalidation_checker(df):
    """Check invalidation conditions from the LLM output"""
    invalidation_conditions = llm_output['opening_signal']['invalidation']
    invalidation_triggered = False
    triggered_conditions = []
    

    
    for condition in invalidation_conditions:
        condition_id = condition['id']
        condition_type = condition['type']
        condition_met = False
        

        
        if condition_type == 'price_breach':
            # Handle price breach conditions using shared function
            level = condition.get('level')
            comparator = condition.get('comparator', '>')
            
            if level == 'bollinger_middle':
                # Convert to price_level format for shared function
                price_condition = {
                    'level': 'bollinger_middle',
                    'comparator': comparator
                }
                condition_met = check_price_level(df, price_condition)
            elif isinstance(level, (int, float)):
                # Direct price level comparison
                price_condition = {
                    'level': 'direct',
                    'comparator': comparator,
                    'value': level
                }
                condition_met = check_price_level(df, price_condition)
        
        elif condition_type == 'indicator_threshold':
            # Use shared indicator threshold function
            condition_met = check_indicator_threshold(df, condition)
        
        elif condition_type == 'indicator_crossover':
            # Use shared crossover function
            condition_met = check_indicator_crossover(df, condition)
        
        elif condition_type == 'price_level':
            # Use shared price level function
            condition_met = check_price_level(df, condition)
        
        elif condition_type == 'sequence':
            # Use shared sequence function
            condition_met = check_sequence_condition(df, condition)
        
        else:
            print(f"Unknown invalidation condition type: {condition_type}")
            condition_met = False
        
        # Check if this invalidation condition is triggered
        if condition_met:
            invalidation_triggered = True
            triggered_conditions.append(condition_id)
    
    return invalidation_triggered, triggered_conditions



def validate_trading_signal(df):
    """Comprehensive trading signal validation combining checklist and invalidation checks"""
    print("    Validating trading signal...")
    
    # Check if all checklist conditions are met
    print("    Checking indicator checklist...")
    checklist_passed = indicator_checker(df)
    print(f"    Checklist passed: {checklist_passed}")
    
    # Check if any invalidation conditions are triggered
    print("    Checking invalidation conditions...")
    invalidation_triggered, triggered_conditions = invalidation_checker(df)
    print(f"    Invalidation triggered: {invalidation_triggered}")
    if triggered_conditions:
        print(f"    Triggered conditions: {triggered_conditions}")
    
    # Get current market values for the return
    current_price = df['Close'].iloc[-1]
    current_time = df['Open_time'].iloc[-1]
    current_rsi = df['RSI14'].iloc[-1] if 'RSI14' in df.columns else None
    
    market_values = {
        'current_price': current_price,
        'current_time': current_time,
        'current_rsi': current_rsi
    }
    
    if invalidation_triggered:
        print("    Signal validation result: INVALIDATED")
        return False, "invalidated", triggered_conditions, market_values
    elif checklist_passed:
        print("    Signal validation result: VALID")
        return True, "valid", [], market_values
    else:
        print("    Signal validation result: PENDING")
        return False, "pending", [], market_values

def _timeframe_seconds(interval):
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

def poll_until_decision(symbol, timeframe, max_cycles=None):
    cycles = 0
    wait_seconds = _timeframe_seconds(timeframe)
    print(f"  Polling started: waiting {wait_seconds} seconds between checks")
    while True:
        print(f"  Polling cycle {cycles + 1}: fetching fresh market data...")
        current_df = fetch_market_dataframe(symbol, timeframe)
        current_df = calculate_rsi14(current_df)
        print(f"  Polling cycle {cycles + 1}: validating trading signal...")
        signal_valid, signal_status, triggered_conditions, market_values = validate_trading_signal(current_df)
        print(f"  Polling cycle {cycles + 1}: signal status = {signal_status}")

        if signal_status != "pending":
            print(f"  Polling complete: final status = {signal_status}")
            return signal_valid, signal_status, triggered_conditions, market_values
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            print(f"  Polling complete: max cycles ({max_cycles}) reached")
            return signal_valid, signal_status, triggered_conditions, market_values

        print(f"  Polling cycle {cycles + 1}: waiting {wait_seconds} seconds...")
        time.sleep(wait_seconds)

print("Step 10: Starting signal validation polling...")
# Run polling until signal is validated or invalidated
signal_valid, signal_status, triggered_conditions, market_values = poll_until_decision(symbol, timeframe)
print(f"Step 10 Complete: Final Signal Status: {signal_status}")

print("Step 11: Checking if signal is valid for trade gate...")
# If programmatic signal is valid, run LLM trade gate before opening a position
if signal_valid and signal_status == "valid":
    print("Step 11 Complete: Signal is valid, proceeding to trade gate")
    print("Step 12: Running LLM trade gate decision...")
    # Derive a simple checklist score for the gate from the last indicator check run
    checklist_passed = True
    invalidation_triggered_recent = len(triggered_conditions) > 0
    gate_result = llm_trade_gate_decision(
        base_llm_output=llm_output,
        market_values=market_values,
        checklist_passed=checklist_passed,
        invalidation_triggered=invalidation_triggered_recent,
        triggered_conditions=triggered_conditions,
    )
    print("Step 12 Complete: LLM trade gate decision finished")
    
    print("Step 13: Saving gate result...")
    # Save gate result
    gate_filename = f"llm_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    gate_path = os.path.join(output_dir, gate_filename)
    with open(gate_path, 'w') as f:
        json.dump(gate_result, f, indent=2, default=str)
    print(f"Step 13 Complete: Gate result saved to {gate_path}")

    print("Step 14: Checking if trade should be opened...")
    if gate_result.get("should_open") is True:
        print("Step 14 Complete: Trade approved for opening")
        # Place order integration would go here
        pass
    else:
        print("Step 14 Complete: Trade not approved by gate")
else:
    print(f"Step 11 Complete: Signal not valid (status: {signal_status}), skipping trade gate")

def calculate_time_difference(time_of_screenshot, df):
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




