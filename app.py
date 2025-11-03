import os
import json
import base64
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import litellm
from prompt import OPENAI_VISION_PROMPT, TRADE_GATE_PROMPT
import pandas as pd
from datetime import datetime
import time

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from eth_account import Account

# Import notification service
from services.notification_service import notify_valid_trade, notify_invalidated_trade

# Import Telegram bot for command control
from telegram_bot import (
    get_analysis_state, 
    should_stop_analysis, 
    set_analysis_running, 
    set_analysis_info,
    send_telegram_status,
    start_bot
)

# Load environment variables
load_dotenv()

# Start Telegram bot for command control
try:
    start_bot()
    print("✅ Telegram bot started for command control")
except Exception as e:
    print(f"⚠️  Could not start Telegram bot: {e}")

# Configure LiteLLM models (can be changed via environment variable)
# Examples: "gpt-4o", "claude-3-5-sonnet-20241022", "gemini/gemini-1.5-pro", "deepseek/deepseek-chat"
# Vision model for chart analysis (must support vision)
LITELLM_VISION_MODEL = os.getenv("LITELLM_VISION_MODEL", os.getenv("LITELLM_MODEL", "gpt-4o"))
# Text model for trade gate decisions (can be any model, including non-vision models like DeepSeek)
LITELLM_TEXT_MODEL = os.getenv("LITELLM_TEXT_MODEL", os.getenv("LITELLM_MODEL", "gpt-4o"))

test_image = "BTCUSDT.P_2025-09-02_22-55-40_545b4.png"

def analyze_trading_chart(image_path: str) -> dict:
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    response = litellm.completion(
        model=LITELLM_VISION_MODEL,  # Use vision model for image analysis
        messages=[
            {"role": "system", "content": OPENAI_VISION_PROMPT},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
        ],
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content
    
    if content is None:
        raise ValueError("LLM API returned None content")
    
    return json.loads(content)

def llm_trade_gate_decision(
    base_llm_output: Dict[str, Any],
    market_values: Dict[str, Any],
    checklist_passed: bool,
    invalidation_triggered: bool,
    triggered_conditions: list
) -> Dict[str, Any]:
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

    response = litellm.completion(
        model=LITELLM_TEXT_MODEL,  # Use text model for trade gate (can be DeepSeek)
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

def main():
    """Main execution function - only runs when script is called directly"""
    print("Step 1: Analyzing trading chart with LLM...")
    # Mark analysis as running
    set_analysis_running(True)
    send_telegram_status("🚀 <b>Analysis Started</b>\n\nAnalyzing trading chart...")

    llm_output = analyze_trading_chart(test_image)
    print(f"Step 1 Complete: Chart analysis finished for {test_image}")

    # Check if stop was requested
    if should_stop_analysis():
        print("⏹️  Analysis stopped by user request")
        set_analysis_running(False)
        send_telegram_status("⏹️ <b>Analysis Stopped</b>\n\nAnalysis was cancelled by user.")
        return  # Exit main() function, not the entire process

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

    # Get direction and update analysis state
    direction = llm_output.get('opening_signal', {}).get('direction', 'Unknown')
    set_analysis_info(symbol, direction)
    send_telegram_status(f"📊 <b>Chart Analyzed</b>\n\n<b>Symbol:</b> {symbol}\n<b>Direction:</b> {direction}\n<b>Timeframe:</b> {timeframe}\n\nFetching market data...")

    print("Step 5: Validating timeframe against Hyperliquid supported intervals...")
    # Validate timeframe against Hyperliquid supported intervals
    # Hyperliquid uses: 1m, 5m, 15m, 1h, 4h, 1d
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    timeframe_mapping = {
        '60m': '1h',
        '1W': '1d',
        '1M': '1d'
    }
    if timeframe in timeframe_mapping:
        print(f"Warning: {timeframe} is not directly supported. Using {timeframe_mapping[timeframe]} as equivalent.")
        timeframe = timeframe_mapping[timeframe]
    elif timeframe not in valid_intervals:
        print(f"Warning: {timeframe} is not a valid Hyperliquid interval. Using 1m as default.")
        timeframe = '1m'
    print(f"Step 5 Complete: Using timeframe {timeframe}")



    print("Step 6: Initializing Hyperliquid API clients...")
    # Initialize Hyperliquid clients
    # For market data (public)
    info_client = Info(skip_ws=True)
    # For trading (requires private key)
    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    if private_key:
        account = Account.from_key(private_key)
        exchange_client = Exchange(account, skip_ws=True)
        print(f"Step 6 Complete: Hyperliquid clients initialized (wallet: {account.address})")
    else:
        exchange_client = None
        print("Step 6 Complete: Hyperliquid info client initialized (no private key for trading)")

    # make http request to api

def _convert_symbol_to_hyperliquid(symbol: str) -> str:
        """Convert symbols like BTCUSDT or BTCUSDT.P to Hyperliquid format (e.g., BTC)."""
        if not symbol:
            return symbol
        # Remove .P suffix if present
        cleaned = symbol.replace(".P", "").replace("USDT", "").replace("USDC", "")
        # Remove any slashes
        cleaned = cleaned.replace("/", "")
        return cleaned.upper()

def fetch_market_data(symbol, timeframe):
        """Fetch raw klines data from Hyperliquid API"""
        print(f"      Fetching market data for {symbol} on {timeframe} timeframe...")
        try:
            # Convert symbol to Hyperliquid format
            hl_symbol = _convert_symbol_to_hyperliquid(symbol)
        
            # Hyperliquid expects intervals like "1m", "5m", "15m", "1h", "4h", "1d"
            # Fetch candles using Hyperliquid's API
            candles = info_client.candles_snapshot(
                coin=hl_symbol,
                interval=timeframe,
                startTime=int((datetime.now().timestamp() - 400 * 60) * 1000),  # rough estimate
                endTime=int(datetime.now().timestamp() * 1000)
            )
        
            if candles:
                print(f"      Successfully fetched {len(candles)} klines")
                # Convert Hyperliquid candles format to MEXC-like format for compatibility
                # Hyperliquid format: [timestamp, open, high, low, close, volume]
                klines = []
                for candle in candles:
                    klines.append([
                        candle['t'],  # timestamp in ms
                        candle['o'],  # open
                        candle['h'],  # high
                        candle['l'],  # low
                        candle['c'],  # close
                        candle['v'],  # volume
                        candle['T'],  # close time
                        0  # quote asset volume (not provided by Hyperliquid)
                    ])
            else:
                print(f"      Warning: No klines data returned for {symbol} on {timeframe}")
                klines = []
    
        except Exception as e:
            print(f"      Error retrieving klines for {symbol} on {timeframe}: {e}")
            print(f"      This might be due to invalid symbol, timeframe, or API issues")
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

from services.indicator_service import (
    calculate_rsi14,
    calculate_macd12_26_9,
    calculate_stoch14_3_3,
    calculate_bb20_2,
    calculate_atr14,
)


def open_trade_with_hyperliquid(symbol: str, gate_result: Dict[str, Any]) -> Dict[str, Any]:
    """Place a perpetual order on Hyperliquid based on the gate result.

    Environment overrides:
    - HYPERLIQUID_DEFAULT_SIZE: float size in USD (default 10.0)
    - HYPERLIQUID_LEVERAGE: int leverage (default 1, max depends on asset)
    """
    if not exchange_client:
        return {"ok": False, "error": "No exchange client initialized (missing HYPERLIQUID_PRIVATE_KEY)"}
    
    hl_symbol = _convert_symbol_to_hyperliquid(symbol)
    direction = (gate_result.get("direction") or "").lower()
    execution = gate_result.get("execution", {})
    entry_type = (execution.get("entry_type") or "market").lower()
    entry_price = execution.get("entry_price")
    stop_loss = execution.get("stop_loss")
    take_profits = execution.get("take_profits") or []

    # Defaults and envs
    try:
        default_size = float(os.getenv("HYPERLIQUID_DEFAULT_SIZE", "10.0"))
    except Exception:
        default_size = 10.0
    
    try:
        leverage = int(os.getenv("HYPERLIQUID_LEVERAGE", "1"))
    except Exception:
        leverage = 1

    # Map direction to Hyperliquid side (true = buy/long, false = sell/short)
    is_buy = (direction == "long")

    # Determine order type
    if entry_type == "market":
        limit_price = None  # Market order
    else:
        limit_price = float(entry_price) if entry_price else None

    try:
        # Set leverage first
        exchange_client.update_leverage(leverage, hl_symbol)
        
        # Place the order
        order_result = exchange_client.market_open(
            coin=hl_symbol,
            is_buy=is_buy,
            sz=default_size,
            px=limit_price
        )
        
        # Note: Hyperliquid doesn't support stop-loss/take-profit in the same order
        # You would need to place separate orders for those
        # For now, we'll just place the main order
        
        return {"ok": True, "response": order_result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

    print("Step 7: Fetching market data...")
    # Use the new function for cleaner code
    df = fetch_market_dataframe(symbol, timeframe)
    print(f"Step 7 Complete: Market data fetched, {len(df)} candles retrieved")

    print("Step 8: Calculating technical indicators...")
    # Calculate RSI14
    df = calculate_rsi14(df)
    # Calculate MACD12_26_9
    df = calculate_macd12_26_9(df)
    # Calculate STOCH14_3_3
    df = calculate_stoch14_3_3(df)
    # Calculate BB20_2
    df = calculate_bb20_2(df)
    # Calculate ATR14
    df = calculate_atr14(df)
    print("Step 8 Complete: Technical indicators calculation finished")

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
        # Ensure both values are numeric for comparison
        try:
            current_num = float(current_value) if current_value is not None else 0.0
            target_num = float(target_value) if target_value is not None else 0.0
        except (ValueError, TypeError):
            return False
    
        if comparator == '<=':
            return current_num <= target_num
        elif comparator == '>=':
            return current_num >= target_num
        elif comparator == '<':
            return current_num < target_num
        elif comparator == '>':
            return current_num > target_num
        elif comparator == '==':
            return current_num == target_num
        elif comparator == '!=':
            return current_num != target_num
        else:
            return False

def check_indicator_threshold(df, condition):
        """Check indicator threshold conditions"""
        indicator_name = condition['indicator']
        comparator = condition['comparator']
        threshold_value = condition['value']
    
        # Check if DataFrame is empty or missing required columns
        if df.empty or indicator_name not in df.columns:
            print(f"      Warning: DataFrame empty or missing indicator '{indicator_name}'")
            return False
    
        # Check if the indicator value is null/NaN
        current_value = df[indicator_name].iloc[-1]
        if pd.isna(current_value):
            print(f"      Warning: Indicator '{indicator_name}' value is null/NaN")
            return False
    
        condition_met = evaluate_comparison(current_value, comparator, threshold_value)
        print(f"      Indicator '{indicator_name}': {current_value} {comparator} {threshold_value} = {condition_met}")
        return condition_met

def check_price_level(df, condition):
        """Check price level conditions"""
        level_type = condition.get('level')
        comparator = condition.get('comparator', '>')
        level_value = condition.get('value')
    
        # Check if DataFrame is empty or missing required columns
        if df.empty or 'Close' not in df.columns:
            print(f"      Warning: DataFrame empty or missing 'Close' column")
            return False
    
        current_price = df['Close'].iloc[-1]
    
        # Check if price value is null/NaN
        if pd.isna(current_price):
            print(f"      Warning: Current price is null/NaN")
            return False
    
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

def check_macd_condition(df, condition):
        """Check MACD-specific conditions"""
        if df.empty or len(df) < 2:
            return False
    
        condition_text = condition.get('condition', '').lower()
    
        # Check if required MACD columns exist
        if 'MACD_Line' not in df.columns or 'MACD_Signal' not in df.columns or 'MACD_Histogram' not in df.columns:
            return False
    
        # Get the last few values for trend analysis
        macd_line = df['MACD_Line'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        macd_histogram = df['MACD_Histogram'].iloc[-1]
    
        # Check for null values
        if pd.isna(macd_line) or pd.isna(macd_signal) or pd.isna(macd_histogram):
            return False
    
        if 'histogram increasing' in condition_text:
            # Check if histogram is increasing over last 2 candles
            if len(df) >= 2:
                prev_histogram = df['MACD_Histogram'].iloc[-2]
                if not pd.isna(prev_histogram):
                    return macd_histogram > prev_histogram
            return False
    
        elif 'histogram decreasing' in condition_text:
            # Check if histogram is decreasing over last 2 candles
            if len(df) >= 2:
                prev_histogram = df['MACD_Histogram'].iloc[-2]
                if not pd.isna(prev_histogram):
                    return macd_histogram < prev_histogram
            return False
    
        elif 'macd_line < signal_line' in condition_text:
            return macd_line < macd_signal
    
        elif 'macd_line > signal_line' in condition_text:
            return macd_line > macd_signal
    
        # Default case - check if histogram is positive
        return macd_histogram > 0

def check_stoch_condition(df, condition):
        """Check Stochastic-specific conditions"""
        if df.empty:
            return False
    
        condition_text = condition.get('condition', '').lower()
    
        # Check if required STOCH columns exist
        if 'STOCH_K' not in df.columns or 'STOCH_D' not in df.columns:
            return False
    
        k_percent = df['STOCH_K'].iloc[-1]
        d_percent = df['STOCH_D'].iloc[-1]
    
        # Check for null values
        if pd.isna(k_percent) or pd.isna(d_percent):
            return False
    
        if 'k_percent < d_percent' in condition_text:
            return k_percent < d_percent
    
        elif 'k_percent > d_percent' in condition_text:
            return k_percent > d_percent
    
        # Default case - check if K is above D
        return k_percent > d_percent

def list_of_indicator_checker():
        """Return list of indicator items, supporting new and legacy schemas.

        - New: concatenates `opening_signal.core_checklist` and `opening_signal.secondary_checklist`.
        - Legacy: filters `opening_signal.checklist` by `technical_indicator == True`.
        """
        opening = llm_output.get('opening_signal', {}) if isinstance(llm_output, dict) else {}
        if not isinstance(opening, dict):
            opening = {}
        if 'core_checklist' in opening or 'secondary_checklist' in opening:
            core = opening.get('core_checklist', []) or []
            secondary = opening.get('secondary_checklist', []) or []
            return list(core) + list(secondary)
        technical_indicator_list = []
        for i in opening.get('checklist', []) or []:
            if i.get('technical_indicator') is True:
                technical_indicator_list.append(i)
        return technical_indicator_list

def indicator_checker(df):
        """Return aggregate results following new pass rule components.

        Returns: (all_core_met, num_core_met, total_core)
        """
        opening = llm_output.get('opening_signal', {}) if isinstance(llm_output, dict) else {}
        core_items = opening.get('core_checklist') if isinstance(opening, dict) else None
        legacy_items = None
        if core_items is None:
            # Legacy: treat filtered items as core
            legacy_items = list_of_indicator_checker()

        def eval_one(i):
            indicator_name = i.get('indicator')
            indicator_type = i.get('type') or 'indicator_threshold'
            if indicator_type == 'indicator_threshold':
                return check_indicator_threshold(df, i)
            if indicator_type == 'indicator_crossover':
                return check_indicator_crossover(df, i)
            if indicator_type == 'indicator_condition':
                if indicator_name == 'MACD12_26_9':
                    return check_macd_condition(df, i)
                if indicator_name == 'STOCH14_3_3':
                    return check_stoch_condition(df, i)
            return False

        num_core_met = 0
        total_core = 0
        if core_items is not None:
            total_core = len(core_items)
            for it in core_items:
                if eval_one(it):
                    num_core_met += 1
        elif legacy_items is not None:
            total_core = len(legacy_items)
            for it in legacy_items:
                if eval_one(it):
                    num_core_met += 1

        all_core_met = (total_core > 0 and num_core_met == total_core)
        return all_core_met, num_core_met, total_core



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
        """Comprehensive trading signal validation combining checklist and invalidation checks

        New rule: valid if all core met OR (>=2 core met AND strong pattern).
        Strong pattern: any pattern with confidence >= 0.75.
        """
        print("    Validating trading signal...")
        print("    Checking indicator checklist (core/secondary)...")
        all_core_met, num_core_met, total_core = indicator_checker(df)
        print(f"    Core met: {num_core_met}/{total_core} (all_core_met={all_core_met})")
    
        # Check if any invalidation conditions are triggered
        print("    Checking invalidation conditions...")
        invalidation_triggered, triggered_conditions = invalidation_checker(df)
        print(f"    Invalidation triggered: {invalidation_triggered}")
        if triggered_conditions:
            print(f"    Triggered conditions: {triggered_conditions}")
    
        # Get current market values for the return
        if df.empty:
            print("    Warning: DataFrame is empty, cannot get market values")
            current_price = 0
            current_time = None
            current_rsi = None
        else:
            current_price = df['Close'].iloc[-1] if 'Close' in df.columns else 0
            current_time = df['Open_time'].iloc[-1] if 'Open_time' in df.columns else None
            current_rsi = df['RSI14'].iloc[-1] if 'RSI14' in df.columns else None
    
        market_values = {
            'current_price': current_price,
            'current_time': current_time,
            'current_rsi': current_rsi
        }
    
        if invalidation_triggered:
            print("    Signal validation result: INVALIDATED")
        
            # Send iPhone notification for invalidated trade
            try:
                trade_data = {
                    "symbol": llm_output.get("symbol", "Unknown"),
                    "current_price": market_values.get("current_price", 0),
                    "triggered_conditions": triggered_conditions,
                    "current_rsi": market_values.get("current_rsi", 0)
                }
                notification_results = notify_invalidated_trade(trade_data)
                print(f"📱 Invalidation notifications sent - Pushover: {'✅' if notification_results.get('pushover') else '❌'}, Email: {'✅' if notification_results.get('email') else '❌'}, Telegram: {'✅' if notification_results.get('telegram') else '❌'}")
            except Exception as e:
                print(f"⚠️ Invalidation notification failed: {str(e)}")
        
            return False, "invalidated", triggered_conditions, market_values
        else:
            # Evaluate new pass rule
            patterns = llm_output.get('pattern_analysis', []) if isinstance(llm_output, dict) else []
            strong_pattern = any((p.get('confidence') or 0) >= 0.75 for p in (patterns or []))
            if all_core_met or (num_core_met >= 2 and strong_pattern):
                print("    Signal validation result: VALID")
                return True, "valid", [], market_values
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
        # Check if stop was requested
        if should_stop_analysis():
            print("⏹️  Polling stopped by user request")
            set_analysis_running(False)
            send_telegram_status("⏹️ <b>Analysis Stopped</b>\n\nPolling was cancelled by user.")
            return False, "stopped", [], {}
        
        print(f"  Polling cycle {cycles + 1}: fetching fresh market data...")
        current_df = fetch_market_dataframe(symbol, timeframe)
        current_df = calculate_rsi14(current_df)
        current_df = calculate_macd12_26_9(current_df)
        current_df = calculate_stoch14_3_3(current_df)
        current_df = calculate_bb20_2(current_df)
        current_df = calculate_atr14(current_df)
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

        # Send periodic status updates
        if cycles % 5 == 0:  # Every 5 cycles
            elapsed_min = int((cycles * wait_seconds) / 60)
            send_telegram_status(f"⏳ <b>Still Polling...</b>\n\nCycle {cycles}\nElapsed: {elapsed_min}m\nStatus: {signal_status}")

        print(f"  Polling cycle {cycles + 1}: waiting {wait_seconds} seconds...")
        time.sleep(wait_seconds)

    print("Step 10: Starting signal validation polling...")
    # Run polling until signal is validated or invalidated
    signal_valid, signal_status, triggered_conditions, market_values = poll_until_decision(symbol, timeframe)
    print(f"Step 10 Complete: Final Signal Status: {signal_status}")

    print("Step 11: Checking if signal is valid for trade gate...")

    # Check if analysis was stopped
    if signal_status == "stopped":
        print("Step 11 Complete: Analysis was stopped, exiting")
        return  # Exit main() function, not the entire process

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
        
            # Send iPhone notification for valid trade
            try:
                trade_data = {
                    "symbol": llm_output.get("symbol", "Unknown"),
                    "direction": gate_result.get("direction", "unknown"),
                    "current_price": gate_result.get("execution", {}).get("entry_price", 0),
                    "confidence": gate_result.get("confidence", 0),
                    "current_rsi": market_values.get("current_rsi", 0),
                    "stop_loss": gate_result.get("execution", {}).get("stop_loss", 0),
                    "risk_reward": gate_result.get("execution", {}).get("risk_reward", 0),
                    "take_profits": gate_result.get("execution", {}).get("take_profits", [])
                }
                notification_results = notify_valid_trade(trade_data)
                print(f"📱 Notifications sent - Pushover: {'✅' if notification_results.get('pushover') else '❌'}, Email: {'✅' if notification_results.get('email') else '❌'}, Telegram: {'✅' if notification_results.get('telegram') else '❌'}")
            except Exception as e:
                print(f"⚠️ Notification failed: {str(e)}")
        
            # Place order on Hyperliquid
            print("Step 15: Placing Hyperliquid perpetual order...")
            order_outcome = open_trade_with_hyperliquid(llm_output.get("symbol", ""), gate_result)
            if order_outcome.get("ok"):
                print("Step 15 Complete: Order placed successfully")
                # Save order response
                try:
                    order_filename = f"hyperliquid_order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    order_path = os.path.join(output_dir, order_filename)
                    with open(order_path, 'w') as f:
                        json.dump(order_outcome.get("response"), f, indent=2, default=str)
                    print(f"Saved order response to {order_path}")
                except Exception as e:
                    print(f"⚠️ Failed to save order response: {str(e)}")
            else:
                print(f"Step 15 Failed: Order placement error - {order_outcome.get('error')}")
        else:
            print("Step 14 Complete: Trade not approved by gate")
    else:
        print(f"Step 11 Complete: Signal not valid (status: {signal_status}), skipping trade gate")

    # Mark analysis as complete
    print("Analysis complete!")
    set_analysis_running(False)
    send_telegram_status("✅ <b>Analysis Complete</b>\n\nThe analysis has finished.")

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


if __name__ == "__main__":
    # Only run analysis when script is called directly
    # Not when imported as a module (e.g., by web_app.py)
    main()

