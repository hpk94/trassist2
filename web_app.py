import os
import json
import base64
import tempfile
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from prompt import OPENAI_VISION_PROMPT, TRADE_GATE_PROMPT
import pandas as pd
from datetime import datetime
import time
from flask import Flask, request, jsonify, render_template_string, Response
from werkzeug.utils import secure_filename
import subprocess
import sys
import threading
import queue
import time
from database import create_trade, get_active_trade, cancel_active_trade, update_trade_status, get_trade_history, get_all_trades

# pymexc will be imported when needed to avoid compatibility issues

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Global progress queue for real-time updates
progress_queue = queue.Queue()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def emit_progress(message, step=None, total_steps=None):
    """Emit a progress message to the queue"""
    progress_data = {
        'message': message,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'step': step,
        'total_steps': total_steps
    }
    progress_queue.put(progress_data)
    print(message)  # Also print to console

def analyze_trading_chart(image_path: str) -> dict:
    """Analyze trading chart using OpenAI Vision API"""
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
    """Make trade gate decision using LLM"""
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

def fetch_market_data(symbol, timeframe):
    """Fetch raw klines data from MEXC API"""
    try:
        from pymexc import spot
        public_spot_client = spot.HTTP()
        klines = public_spot_client.klines(
            symbol=symbol,
            interval=timeframe,
            limit=100
        )
        return klines
    except Exception as e:
        print(f"Error retrieving klines: {e}")
        return []

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

def check_price_level(df, condition, llm_output):
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
    # For now, skip crossover conditions as they need more complex logic
    return False

def check_sequence_condition(df, condition):
    """Check sequence conditions (e.g., consecutive candles)"""
    return False

def list_of_indicator_checker(llm_output):
    """Get list of technical indicators from LLM output"""
    technical_indicator_list = []
    for i in llm_output['opening_signal']['checklist']:
        if i['technical_indicator'] == True:
            technical_indicator_list.append(i)
    return technical_indicator_list

def indicator_checker(df, llm_output):
    """Check if all technical indicator conditions are met"""
    technical_indicator_list = list_of_indicator_checker(llm_output)
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

def invalidation_checker(df, llm_output):
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
                condition_met = check_price_level(df, price_condition, llm_output)
            elif isinstance(level, (int, float)):
                # Direct price level comparison
                price_condition = {
                    'level': 'direct',
                    'comparator': comparator,
                    'value': level
                }
                condition_met = check_price_level(df, price_condition, llm_output)
        
        elif condition_type == 'indicator_threshold':
            # Use shared indicator threshold function
            condition_met = check_indicator_threshold(df, condition)
        
        elif condition_type == 'indicator_crossover':
            # Use shared crossover function
            condition_met = check_indicator_crossover(df, condition)
        
        elif condition_type == 'price_level':
            # Use shared price level function
            condition_met = check_price_level(df, condition, llm_output)
        
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

def validate_trading_signal(df, llm_output):
    """Comprehensive trading signal validation combining checklist and invalidation checks"""
    # Check if all checklist conditions are met
    checklist_passed = indicator_checker(df, llm_output)
    
    # Check if any invalidation conditions are triggered
    invalidation_triggered, triggered_conditions = invalidation_checker(df, llm_output)
    
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
        return False, "invalidated", triggered_conditions, market_values
    elif checklist_passed:
        return True, "valid", [], market_values
    else:
        return False, "pending", [], market_values

def _timeframe_seconds(interval):
    """Convert timeframe to seconds"""
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

def poll_until_decision(symbol, timeframe, llm_output, max_cycles=None):
    """Poll market data until trading decision is made"""
    cycles = 0
    wait_seconds = _timeframe_seconds(timeframe)
    
    while True:
        current_df = fetch_market_dataframe(symbol, timeframe)
        current_df = calculate_rsi14(current_df)
        signal_valid, signal_status, triggered_conditions, market_values = validate_trading_signal(current_df, llm_output)

        if signal_status != "pending":
            return signal_valid, signal_status, triggered_conditions, market_values
        
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            return signal_valid, signal_status, triggered_conditions, market_values

        time.sleep(wait_seconds)

def run_trading_analysis(image_path: str) -> dict:
    """Run the complete trading analysis pipeline"""
    try:
        # Step 1: Analyze trading chart with LLM
        emit_progress("Step 1: Analyzing trading chart with LLM...", 1, 14)
        llm_output = analyze_trading_chart(image_path)
        emit_progress("Step 1 Complete: Chart analysis finished", 1, 14)

        # Step 2: Create output directory
        emit_progress("Step 2: Setting up output directory...", 2, 14)
        output_dir = "llm_outputs"
        os.makedirs(output_dir, exist_ok=True)
        emit_progress(f"Step 2 Complete: Output directory ready at {output_dir}", 2, 14)

        # Step 3: Save LLM output to JSON file
        emit_progress("Step 3: Saving LLM output to JSON file...", 3, 14)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"llm_output_{timestamp}.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(llm_output, f, indent=2, default=str)
        emit_progress(f"Step 3 Complete: LLM output saved to {output_path}", 3, 14)

        # Step 4: Extract trading parameters
        emit_progress("Step 4: Extracting trading parameters from LLM output...", 4, 14)
        symbol = llm_output['symbol']
        timeframe = llm_output['timeframe']
        time_of_screenshot = llm_output['time_of_screenshot']
        emit_progress(f"Step 4 Complete: Symbol={symbol}, Timeframe={timeframe}, Screenshot time={time_of_screenshot}", 4, 14)

        # Step 5: Validate timeframe
        emit_progress("Step 5: Validating timeframe against MEXC supported intervals...", 5, 14)
        valid_intervals = ['1m', '5m', '15m', '30m', '60m', '4h', '1d', '1W', '1M']
        if timeframe not in valid_intervals:
            emit_progress(f"Warning: {timeframe} is not a valid MEXC interval. Using 1m as default.", 5, 14)
            timeframe = '1m'
        emit_progress(f"Step 5 Complete: Using timeframe {timeframe}", 5, 14)

        # Step 6: Initialize MEXC API clients
        emit_progress("Step 6: Initializing MEXC API clients...", 6, 14)
        try:
            from pymexc import spot
            spot_client = spot.HTTP(api_key=os.getenv("MEXC_API_KEY"), api_secret=os.getenv("MEXC_API_SECRET"))
            public_spot_client = spot.HTTP()
            ws_spot_client = spot.WebSocket(api_key=os.getenv("MEXC_API_KEY"), api_secret=os.getenv("MEXC_API_SECRET"))
            emit_progress("Step 6 Complete: MEXC API clients initialized", 6, 14)
        except Exception as e:
            emit_progress(f"Step 6 Warning: MEXC API initialization failed: {e}", 6, 14)
            emit_progress("Step 6 Complete: Continuing without MEXC API clients", 6, 14)

        # Step 7: Fetch market data
        emit_progress("Step 7: Fetching market data...", 7, 14)
        df = fetch_market_dataframe(symbol, timeframe)
        emit_progress(f"Step 7 Complete: Market data fetched, {len(df)} candles retrieved", 7, 14)

        # Step 8: Calculate RSI14
        emit_progress("Step 8: Calculating RSI14 indicator...", 8, 14)
        df = calculate_rsi14(df)
        emit_progress("Step 8 Complete: RSI14 calculation finished", 8, 14)

        # Step 9: Extract latest market indicators
        emit_progress("Step 9: Extracting latest market indicators...", 9, 14)
        if not df.empty and 'RSI14' in df.columns:
            latest_rsi = df['RSI14'].iloc[-1]
            emit_progress(f"Step 9 Complete: Latest RSI14 = {latest_rsi}", 9, 14)
        else:
            emit_progress("Step 9 Complete: No RSI14 data available", 9, 14)

        # Step 10: Start signal validation polling
        emit_progress("Step 10: Starting signal validation polling...", 10, 14)
        signal_valid, signal_status, triggered_conditions, market_values = poll_until_decision(symbol, timeframe, llm_output)
        emit_progress(f"Step 10 Complete: Final Signal Status: {signal_status}", 10, 14)

        # Step 11: Check if signal is valid for trade gate
        emit_progress("Step 11: Checking if signal is valid for trade gate...", 11, 14)
        gate_result = None
        
        if signal_valid and signal_status == "valid":
            emit_progress("Step 11 Complete: Signal is valid, proceeding to trade gate", 11, 14)
            emit_progress("Step 12: Running LLM trade gate decision...", 12, 14)
            
            checklist_passed = True
            invalidation_triggered_recent = len(triggered_conditions) > 0
            gate_result = llm_trade_gate_decision(
                base_llm_output=llm_output,
                market_values=market_values,
                checklist_passed=checklist_passed,
                invalidation_triggered=invalidation_triggered_recent,
                triggered_conditions=triggered_conditions,
            )
            emit_progress("Step 12 Complete: LLM trade gate decision finished", 12, 14)
            
            emit_progress("Step 13: Saving gate result...", 13, 14)
            gate_filename = f"llm_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            gate_path = os.path.join(output_dir, gate_filename)
            with open(gate_path, 'w') as f:
                json.dump(gate_result, f, indent=2, default=str)
            emit_progress(f"Step 13 Complete: Gate result saved to {gate_path}", 13, 14)

            emit_progress("Step 14: Checking if trade should be opened...", 14, 14)
            if gate_result.get("should_open") is True:
                emit_progress("Step 14 Complete: Trade approved for opening", 14, 14)
            else:
                emit_progress("Step 14 Complete: Trade not approved by gate", 14, 14)
        else:
            emit_progress(f"Step 11 Complete: Signal not valid (status: {signal_status}), skipping trade gate", 11, 14)

        # Create trade record in database
        trade_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": llm_output.get("opening_signal", {}).get("direction", "unknown"),
            "llm_output": llm_output,
            "signal_status": signal_status,
            "signal_valid": signal_valid,
            "market_values": market_values,
            "gate_result": gate_result,
            "output_files": {
                "llm_output": output_path,
                "gate_result": gate_path if gate_result else None
            },
            "notes": f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
        
        trade_id = create_trade(trade_data)
        emit_progress(f"Trade saved to database with ID: {trade_id}", 14, 14)

        return {
            "success": True,
            "trade_id": trade_id,
            "llm_output": llm_output,
            "signal_status": signal_status,
            "signal_valid": signal_valid,
            "market_values": market_values,
            "gate_result": gate_result,
            "output_files": {
                "llm_output": output_path,
                "gate_result": gate_path if gate_result else None
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.route('/')
def index():
    """Simple HTML form for image upload"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Chart Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .loading { background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            button:disabled { background-color: #6c757d; cursor: not-allowed; }
        </style>
    </head>
    <body>
        <h1>Trading Chart Analysis</h1>
        <p>Upload a trading chart image to analyze it using AI and get trading signals.</p>
        
        <!-- Active Trade Status -->
        <div id="activeTradeStatus" class="result" style="display: none;">
            <h3>Active Trade Status</h3>
            <div id="activeTradeInfo"></div>
            <button id="cancelTradeBtn" class="cancel-btn" style="background-color: #dc3545; margin-top: 10px;">Cancel Active Trade</button>
        </div>
        
        <form id="uploadForm" class="upload-form" enctype="multipart/form-data">
            <input type="file" id="imageFile" name="image" accept="image/*" required>
            <br><br>
            <button type="submit" id="submitBtn">Analyze Chart</button>
        </form>
        
        <div id="result"></div>
        <div id="progress" style="display: none;">
            <h3>Analysis Progress</h3>
            <div id="progressBar" style="width: 100%; background-color: #f0f0f0; border-radius: 5px; margin: 10px 0;">
                <div id="progressFill" style="width: 0%; height: 20px; background-color: #007bff; border-radius: 5px; transition: width 0.3s;"></div>
            </div>
            <div id="progressMessages" style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
                <!-- Progress messages will appear here -->
            </div>
        </div>
        
        <script>
            let eventSource = null;
            let analysisComplete = false;
            
            // Load active trade status on page load
            async function loadActiveTradeStatus() {
                try {
                    const response = await fetch('/api/active-trade');
                    const data = await response.json();
                    
                    const activeTradeStatus = document.getElementById('activeTradeStatus');
                    const activeTradeInfo = document.getElementById('activeTradeInfo');
                    
                    if (data.success && data.trade) {
                        const trade = data.trade;
                        activeTradeInfo.innerHTML = `
                            <p><strong>Trade ID:</strong> ${trade.trade_id}</p>
                            <p><strong>Symbol:</strong> ${trade.symbol}</p>
                            <p><strong>Timeframe:</strong> ${trade.timeframe}</p>
                            <p><strong>Direction:</strong> ${trade.direction}</p>
                            <p><strong>Status:</strong> ${trade.status}</p>
                            <p><strong>Signal Status:</strong> ${trade.signal_status}</p>
                            <p><strong>Created:</strong> ${new Date(trade.created_at).toLocaleString()}</p>
                            <p><strong>Updated:</strong> ${new Date(trade.updated_at).toLocaleString()}</p>
                        `;
                        activeTradeStatus.style.display = 'block';
                    } else {
                        activeTradeStatus.style.display = 'none';
                    }
                } catch (error) {
                    console.error('Error loading active trade status:', error);
                }
            }
            
            // Cancel active trade
            async function cancelActiveTrade() {
                if (!confirm('Are you sure you want to cancel the active trade?')) {
                    return;
                }
                
                try {
                    const response = await fetch('/api/cancel-trade', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            notes: 'Canceled by user from UI'
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        alert('Active trade canceled successfully!');
                        loadActiveTradeStatus(); // Refresh the status
                    } else {
                        alert('Error canceling trade: ' + data.message);
                    }
                } catch (error) {
                    alert('Error canceling trade: ' + error.message);
                }
            }
            
            // Load active trade status when page loads
            document.addEventListener('DOMContentLoaded', function() {
                loadActiveTradeStatus();
                
                // Set up cancel button event listener
                document.getElementById('cancelTradeBtn').addEventListener('click', cancelActiveTrade);
            });
            
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('imageFile');
                const submitBtn = document.getElementById('submitBtn');
                const resultDiv = document.getElementById('result');
                const progressDiv = document.getElementById('progress');
                const progressMessages = document.getElementById('progressMessages');
                const progressFill = document.getElementById('progressFill');
                
                if (!fileInput.files[0]) {
                    alert('Please select an image file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                
                submitBtn.disabled = true;
                submitBtn.textContent = 'Analyzing...';
                resultDiv.innerHTML = '';
                progressDiv.style.display = 'block';
                progressMessages.innerHTML = '';
                progressFill.style.width = '0%';
                analysisComplete = false;
                
                // Start listening for progress updates
                if (eventSource) {
                    eventSource.close();
                }
                
                eventSource = new EventSource('/progress');
                eventSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.message === 'keep-alive') {
                        return; // Ignore keep-alive messages
                    }
                    
                    // Add progress message
                    const messageDiv = document.createElement('div');
                    messageDiv.style.marginBottom = '5px';
                    messageDiv.style.padding = '5px';
                    messageDiv.style.backgroundColor = '#fff';
                    messageDiv.style.borderRadius = '3px';
                    messageDiv.style.borderLeft = '3px solid #007bff';
                    messageDiv.innerHTML = `<strong>[${data.timestamp}]</strong> ${data.message}`;
                    
                    progressMessages.appendChild(messageDiv);
                    progressMessages.scrollTop = progressMessages.scrollHeight;
                    
                    // Update progress bar
                    if (data.step && data.total_steps) {
                        const percentage = (data.step / data.total_steps) * 100;
                        progressFill.style.width = percentage + '%';
                    }
                    
                    // Check if analysis is complete
                    if (data.message.includes('Complete') && data.step === data.total_steps) {
                        analysisComplete = true;
                        setTimeout(() => {
                            if (eventSource) {
                                eventSource.close();
                            }
                            progressDiv.style.display = 'none';
                            submitBtn.disabled = false;
                            submitBtn.textContent = 'Analyze Chart';
                        }, 2000);
                    }
                };
                
                eventSource.onerror = function(event) {
                    console.error('EventSource failed:', event);
                    if (eventSource) {
                        eventSource.close();
                    }
                };
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        // Wait for analysis to complete
                        const checkCompletion = setInterval(() => {
                            if (analysisComplete) {
                                clearInterval(checkCompletion);
                                resultDiv.innerHTML = `
                                    <div class="result success">
                                        <h3>Analysis Complete!</h3>
                                        <p><strong>Job ID:</strong> ${data.job_id}</p>
                                        <p>Check the progress messages above for detailed results.</p>
                                        <p>Results have been saved to the llm_outputs directory.</p>
                                    </div>
                                `;
                                // Refresh active trade status after new analysis
                                loadActiveTradeStatus();
                            }
                        }, 1000);
                    } else {
                        resultDiv.innerHTML = `
                            <div class="result error">
                                <h3>Upload Failed</h3>
                                <p><strong>Error:</strong> ${data.error}</p>
                            </div>
                        `;
                        if (eventSource) {
                            eventSource.close();
                        }
                        progressDiv.style.display = 'none';
                        submitBtn.disabled = false;
                        submitBtn.textContent = 'Analyze Chart';
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <h3>Upload Failed</h3>
                            <p>Error: ${error.message}</p>
                        </div>
                    `;
                    if (eventSource) {
                        eventSource.close();
                    }
                    progressDiv.style.display = 'none';
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Analyze Chart';
                }
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/progress')
def progress():
    """Server-Sent Events endpoint for real-time progress updates"""
    def generate():
        while True:
            try:
                # Get progress message from queue (blocking with timeout)
                progress_data = progress_queue.get(timeout=1)
                yield f"data: {json.dumps(progress_data)}\n\n"
            except queue.Empty:
                # Send keep-alive message
                yield f"data: {json.dumps({'message': 'keep-alive', 'timestamp': datetime.now().strftime('%H:%M:%S')})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and run trading analysis"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image file selected'})
    
    if file and allowed_file(file.filename):
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # Clear the progress queue before starting
            while not progress_queue.empty():
                progress_queue.get()
            
            # Run the trading analysis in a separate thread
            def run_analysis():
                return run_trading_analysis(temp_path)
            
            # Start analysis in background thread
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.start()
            
            # Return immediately with a job ID
            return jsonify({
                'success': True,
                'message': 'Analysis started',
                'job_id': 'analysis_' + str(int(time.time()))
            })
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    else:
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload an image file.'})

@app.route('/api/active-trade', methods=['GET'])
def get_active_trade_api():
    """Get the currently active trade"""
    try:
        active_trade = get_active_trade()
        if active_trade:
            return jsonify({
                'success': True,
                'trade': active_trade
            })
        else:
            return jsonify({
                'success': True,
                'trade': None,
                'message': 'No active trade found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/cancel-trade', methods=['POST'])
def cancel_trade_api():
    """Cancel the currently active trade"""
    try:
        data = request.get_json() or {}
        notes = data.get('notes', 'Canceled by user')
        
        success = cancel_active_trade(notes)
        if success:
            return jsonify({
                'success': True,
                'message': 'Active trade canceled successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No active trade to cancel'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/trade-history', methods=['GET'])
def get_trade_history_api():
    """Get trade history"""
    try:
        trade_id = request.args.get('trade_id')
        history = get_trade_history(trade_id)
        return jsonify({
            'success': True,
            'history': history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/all-trades', methods=['GET'])
def get_all_trades_api():
    """Get all trades with pagination"""
    try:
        limit = int(request.args.get('limit', 50))
        trades = get_all_trades(limit)
        return jsonify({
            'success': True,
            'trades': trades
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/update-trade-status', methods=['POST'])
def update_trade_status_api():
    """Update trade status"""
    try:
        data = request.get_json()
        if not data or 'trade_id' not in data or 'status' not in data:
            return jsonify({
                'success': False,
                'error': 'trade_id and status are required'
            })
        
        trade_id = data['trade_id']
        status = data['status']
        notes = data.get('notes', '')
        
        success = update_trade_status(trade_id, status, notes)
        if success:
            return jsonify({
                'success': True,
                'message': f'Trade {trade_id} status updated to {status}'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Trade {trade_id} not found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/run-app', methods=['POST'])
def run_app():
    """Alternative endpoint that runs the original app.py script"""
    try:
        # Run the original app.py script
        result = subprocess.run([sys.executable, 'app.py'], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        return jsonify({
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
