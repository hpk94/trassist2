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

# Import notification service
from services.notification_service import notify_valid_trade, notify_invalidated_trade

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

def make_json_serializable(obj):
    """Convert objects to JSON-serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif hasattr(obj, 'isoformat'):  # pandas Timestamp, datetime
        return obj.isoformat()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return str(obj)

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

def _convert_symbol_to_mexc_futures(symbol: str) -> str:
    """Convert symbols like BTCUSDT or BTC/USDT to MEXC futures format (e.g., BTC_USDT)."""
    if not symbol:
        return symbol
    cleaned = symbol.strip().upper()
    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
        return f"{base}_{quote}"
    return cleaned.replace("/", "_")

def open_trade_with_mexc(symbol: str, gate_result: Dict[str, Any]) -> Dict[str, Any]:
    """Place a futures order on MEXC based on the gate result.

    Environment overrides:
    - MEXC_DEFAULT_VOL: float volume in contracts/base units (default 0.001)
    - MEXC_OPEN_TYPE: 1 isolated, 2 cross (default 2)
    - MEXC_LEVERAGE: int leverage for isolated (optional)
    """
    try:
        from pymexc import futures
    except Exception as e:
        return {"ok": False, "error": f"Failed to import pymexc.futures: {e}"}

    mexc_symbol = _convert_symbol_to_mexc_futures(symbol)
    direction = (gate_result.get("direction") or "").lower()
    execution = gate_result.get("execution", {})
    entry_type = (execution.get("entry_type") or "market").lower()
    entry_price = execution.get("entry_price")
    stop_loss = execution.get("stop_loss")
    take_profits = execution.get("take_profits") or []

    # Defaults and envs
    try:
        default_vol = float(os.getenv("MEXC_DEFAULT_VOL", "0.001"))
    except Exception:
        default_vol = 0.001
    try:
        open_type = int(os.getenv("MEXC_OPEN_TYPE", "2"))  # 1 isolated, 2 cross
    except Exception:
        open_type = 2
    leverage_env = os.getenv("MEXC_LEVERAGE")
    leverage = int(leverage_env) if leverage_env and leverage_env.isdigit() else None

    # Map direction to MEXC side
    # 1 open long, 2 close short, 3 open short, 4 close long
    if direction == "long":
        side = 1
    elif direction == "short":
        side = 3
    else:
        return {"ok": False, "error": f"Unknown trade direction: {direction}"}

    # Order type: 5 = market, 1 = limit
    if entry_type == "market":
        order_type = 5
        price = 0
    else:
        order_type = 1
        price = float(entry_price or 0)

    # Use first TP as take-profit if provided
    tp_price = None
    if take_profits:
        first_tp = take_profits[0]
        tp_price = first_tp.get("price") if isinstance(first_tp, dict) else None

    try:
        futures_client = futures.HTTP(api_key=os.getenv("MEXC_API_KEY"), api_secret=os.getenv("MEXC_API_SECRET"))
    except Exception as e:
        return {"ok": False, "error": f"Failed to init futures client: {e}"}

    try:
        response = futures_client.order(
            symbol=mexc_symbol,
            price=price,
            vol=default_vol,
            side=side,
            type=order_type,
            open_type=open_type,
            leverage=leverage,
            stop_loss_price=stop_loss,
            take_profit_price=tp_price,
            reduce_only=False,
        )
        return {"ok": True, "response": response}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def analyze_trading_chart(image_path: str, symbol: str = None, timeframe: str = None, df: pd.DataFrame = None) -> dict:
    """Analyze trading chart using OpenAI Vision API with optional market data"""
    emit_progress("Chart: Reading image file...")
    
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            emit_progress(f"Chart: Image file size: {len(image_bytes)} bytes")
            image_data = base64.b64encode(image_bytes).decode("utf-8")
        emit_progress(f"Chart: Image loaded, base64 size: {len(image_data)} characters")
        
        # Validate that the image data is not empty
        if not image_data:
            raise ValueError("Image file appears to be empty")
            
    except Exception as e:
        emit_progress(f"Chart: Error reading image file: {str(e)}")
        raise
    
    emit_progress("Chart: Initializing OpenAI client...")
    client = OpenAI()
    
    # Prepare the prompt based on whether we have market data
    if df is not None and not df.empty:
        emit_progress("Chart: Preparing enhanced prompt with market data...")
        # Create a summary of the market data for the LLM
        market_data_summary = create_market_data_summary(df, symbol, timeframe)
        enhanced_prompt = OPENAI_VISION_PROMPT + f"\n\n## MARKET DATA CONTEXT\n{symbol} {timeframe} - Latest market data:\n{market_data_summary}\n\nUse this market data to enhance your analysis and provide more accurate technical indicator values."
    else:
        emit_progress("Chart: Using standard prompt without market data...")
        enhanced_prompt = OPENAI_VISION_PROMPT
    
    emit_progress("Chart: Sending request to OpenAI Vision API...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
            ],
            response_format={"type": "json_object"},
            timeout=60  # Add 60 second timeout
        )
        emit_progress("Chart: Received response from OpenAI Vision API")
        emit_progress(f"Chart: Response has {len(response.choices)} choices")
        if response.choices:
            emit_progress(f"Chart: First choice finish_reason: {response.choices[0].finish_reason}")
    except Exception as e:
        emit_progress(f"Chart: Error calling OpenAI API: {str(e)}")
        raise
    
    emit_progress("Chart: Parsing JSON response...")
    try:
        # Check if response content exists
        if not response.choices or not response.choices[0].message.content:
            emit_progress("Chart: Error - No content in API response")
            emit_progress(f"Chart: Response structure: {response}")
            
            # Check if this is a refusal response
            if (response.choices and 
                response.choices[0].message and 
                hasattr(response.choices[0].message, 'refusal') and 
                response.choices[0].message.refusal):
                emit_progress(f"Chart: LLM refused request: {response.choices[0].message.refusal}")
                emit_progress("Chart: Creating fallback response structure...")
                return create_fallback_llm_response(symbol, timeframe)
            
            emit_progress("Chart: Attempting retry with different parameters...")
            
            # Retry without response_format to see if that's the issue
            emit_progress("Chart: Retrying without response_format constraint...")
            retry_response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": enhanced_prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON. No additional text or formatting."},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
                ],
                timeout=60
            )
            
            if not retry_response.choices or not retry_response.choices[0].message.content:
                # Check if retry also resulted in refusal
                if (retry_response.choices and 
                    retry_response.choices[0].message and 
                    hasattr(retry_response.choices[0].message, 'refusal') and 
                    retry_response.choices[0].message.refusal):
                    emit_progress(f"Chart: Retry also refused: {retry_response.choices[0].message.refusal}")
                    emit_progress("Chart: Creating fallback response structure...")
                    return create_fallback_llm_response(symbol, timeframe)
                else:
                    raise ValueError("OpenAI API returned empty content on retry")
            
            content = retry_response.choices[0].message.content
            emit_progress(f"Chart: Retry successful, content length: {len(content)} characters")
        else:
            content = response.choices[0].message.content
            emit_progress(f"Chart: Content length: {len(content)} characters")
        
        # Try to clean the content if it has markdown formatting
        if content.strip().startswith('```json'):
            # Remove markdown code block formatting
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.endswith('```'):
                content = content[:-3]  # Remove ```
            content = content.strip()
            emit_progress("Chart: Cleaned markdown formatting from response")
        
        result = json.loads(content)
        emit_progress("Chart: JSON parsing successful")
        return result
    except Exception as e:
        emit_progress(f"Chart: Error parsing JSON response: {str(e)}")
        if 'content' in locals() and content:
            emit_progress(f"Chart: Raw response: {content[:200]}...")
        else:
            emit_progress(f"Chart: No content available in response")
        raise

def create_fallback_llm_response(symbol: str = None, timeframe: str = None) -> dict:
    """Create a fallback LLM response structure when the AI refuses to analyze the chart"""
    emit_progress("Chart: Creating fallback response due to LLM refusal...")
    
    # Use current timestamp for screenshot time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    fallback_response = {
        "symbol": symbol or "UNKNOWN",
        "timeframe": timeframe or "1m",
        "time_of_screenshot": current_time,
        "trend_direction": "unknown",
        "support_resistance": {
            "support": None,
            "resistance": None
        },
        "technical_indicators": {
            "RSI14": {
                "value": None,
                "status": "unknown",
                "signal": "none"
            },
            "MACD12_26_9": {
                "macd_line": None,
                "signal_line": None,
                "histogram": None,
                "signal": "none"
            },
            "BB20_2": {
                "upper": None,
                "middle": None,
                "lower": None,
                "price_position": "unknown",
                "bandwidth": "unknown"
            },
            "STOCH14_3_3": {
                "k_percent": None,
                "d_percent": None,
                "signal": "none"
            },
            "VOLUME": {
                "current": None,
                "average": None,
                "ratio": None,
                "trend": "unknown"
            },
            "ATR14": {
                "value": None,
                "period": 14,
                "volatility": "unknown"
            }
        },
        "pattern_analysis": [],
        "validity_assessment": {
            "alignment_score": 0.0,
            "notes": "Analysis unavailable due to LLM refusal. Manual review required."
        },
        "opening_signal": {
            "direction": "none",
            "scope": {
                "candle_indices": [],
                "lookback_seconds": 0
            },
            "checklist": [],
            "invalidation": [],
            "is_met": False
        },
        "risk_management": {
            "stop_loss": {
                "price": None,
                "basis": "manual_review_required",
                "distance_ticks": None
            },
            "take_profit": []
        },
        "summary_actions": [
            "Manual chart analysis required due to AI refusal",
            "Review chart patterns and technical indicators manually",
            "Consider alternative analysis methods"
        ],
        "improvements": "LLM analysis unavailable - requires manual review or alternative AI model",
        "fallback_reason": "LLM refused to analyze the chart image",
        "requires_manual_review": True
    }
    
    emit_progress("Chart: Fallback response structure created successfully")
    return fallback_response

def create_market_data_summary(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    """Create a summary of market data for the LLM"""
    if df.empty:
        return "No market data available"
    
    # Get the latest values
    latest = df.iloc[-1]
    latest_time = latest['Open_time'] if 'Open_time' in df.columns else "Unknown"
    
    summary = f"""
Symbol: {symbol}
Timeframe: {timeframe}
Latest Candle Time: {latest_time}
Latest Price Data:
- Open: {latest.get('Open', 'N/A')}
- High: {latest.get('High', 'N/A')}
- Low: {latest.get('Low', 'N/A')}
- Close: {latest.get('Close', 'N/A')}
- Volume: {latest.get('Volume', 'N/A')}
"""
    
    # Add technical indicators if available
    if 'RSI14' in df.columns and not pd.isna(latest['RSI14']):
        summary += f"- RSI14: {latest['RSI14']:.2f}\n"
    
    if 'MACD_Line' in df.columns and not pd.isna(latest['MACD_Line']):
        summary += f"- MACD Line: {latest['MACD_Line']:.4f}\n"
        summary += f"- MACD Signal: {latest['MACD_Signal']:.4f}\n"
        summary += f"- MACD Histogram: {latest['MACD_Histogram']:.4f}\n"

    if 'ATR14' in df.columns and not pd.isna(latest['ATR14']):
        summary += f"- ATR14: {latest['ATR14']:.4f}\n"
    
    # Add recent price action context (last 5 candles)
    if len(df) >= 5:
        recent_5 = df.tail(5)
        summary += f"\nRecent 5 Candles Context:\n"
        for i, (_, candle) in enumerate(recent_5.iterrows()):
            candle_time = candle.get('Open_time', 'Unknown')
            close_price = candle.get('Close', 'N/A')
            summary += f"- Candle {i+1}: {candle_time} - Close: {close_price}\n"
    
    return summary

def llm_trade_gate_decision(
    base_llm_output: Dict[str, Any],
    market_values: Dict[str, Any],
    checklist_passed: bool,
    invalidation_triggered: bool,
    triggered_conditions: list
) -> Dict[str, Any]:
    """Make trade gate decision using LLM"""
    emit_progress("Gate: Initializing LLM trade gate decision...")
    
    client = OpenAI()

    # Prepare concise context for the gate
    emit_progress("Gate: Preparing context for gate decision...")
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
    
    emit_progress(f"Gate: Context prepared - Symbol: {gate_context['llm_snapshot']['symbol']}, Price: ${gate_context['market_values']['current_price']:.2f}, RSI: {gate_context['market_values']['current_rsi']:.2f}")
    emit_progress(f"Gate: Checklist passed: {checklist_passed}, Invalidation triggered: {invalidation_triggered}")

    emit_progress("Gate: Sending request to LLM for trade gate decision...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": TRADE_GATE_PROMPT},
                {"role": "user", "content": json.dumps(gate_context)},
            ],
            response_format={"type": "json_object"},
            timeout=30  # Add 30 second timeout for gate decision
        )
        emit_progress("Gate: Received response from LLM")
    except Exception as e:
        emit_progress(f"Gate: Error calling LLM for gate decision: {str(e)}")
        raise

    emit_progress("Gate: Received response from LLM, parsing result...")
    try:
        content = response.choices[0].message.content
        gate_result = json.loads(content)
        
        # Log the gate decision details
        should_open = gate_result.get("should_open", False)
        confidence = gate_result.get("confidence", 0.0)
        direction = gate_result.get("direction", "unknown")
        reasons = gate_result.get("reasons", [])
        
        emit_progress(f"Gate: Decision made - Should open: {should_open}, Direction: {direction}, Confidence: {confidence:.2f}")
        
        if reasons:
            emit_progress(f"Gate: Reasons: {'; '.join(reasons[:3])}")  # Show first 3 reasons
        
        return gate_result
    except Exception as e:
        emit_progress(f"Gate: Error parsing LLM response: {str(e)}")
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
    print(f"Fetching market data for {symbol} {timeframe}")
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

from services.indicator_service import (
    calculate_rsi14,
    calculate_macd12_26_9,
    calculate_stoch14_3_3,
    calculate_bb20_2,
    calculate_atr14,
)

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
        # Handle special cases for indicators that use different column names
        if indicator_name == 'MACD12_26_9' and 'MACD_Line' in df.columns:
            current_value = df['MACD_Line'].iloc[-1]
        elif indicator_name == 'STOCH14_3_3' and 'STOCH_K' in df.columns:
            current_value = df['STOCH_K'].iloc[-1]
        elif indicator_name == 'BB20_2_PercentB' and 'BB_PercentB' in df.columns:
            current_value = df['BB_PercentB'].iloc[-1]
        elif indicator_name == 'BB20_2_Bandwidth' and 'BB_Bandwidth' in df.columns:
            current_value = df['BB_Bandwidth'].iloc[-1]
        else:
            return False
    else:
        current_value = df[indicator_name].iloc[-1]
    
    # Check if current_value is None or NaN before comparison
    if current_value is None or pd.isna(current_value):
        return False
    
    condition_met = evaluate_comparison(current_value, comparator, threshold_value)
    
    return condition_met

def check_price_level(df, condition, llm_output):
    """Check price level conditions"""
    level_type = condition.get('level')
    comparator = condition.get('comparator', '>')
    level_value = condition.get('value')
    
    # Guard against missing data
    if df is None or df.empty or 'Close' not in df.columns:
        return False
    
    current_price = df['Close'].iloc[-1]
    
    # Check if current_price is None or NaN before comparison
    if current_price is None or pd.isna(current_price):
        return False
    
    if level_type == 'bollinger_middle':
        # Try to get from dataframe first, fallback to LLM output
        if 'BB_Middle' in df.columns:
            bb_middle = df['BB_Middle'].iloc[-1]
        else:
            bb_middle = llm_output['technical_indicators']['BB20_2']['middle']
        condition_met = evaluate_comparison(current_price, comparator, bb_middle)
        return condition_met
    elif level_type == 'bollinger_upper':
        # Try to get from dataframe first, fallback to LLM output
        if 'BB_Upper' in df.columns:
            bb_upper = df['BB_Upper'].iloc[-1]
        else:
            bb_upper = llm_output['technical_indicators']['BB20_2']['upper']
        condition_met = evaluate_comparison(current_price, comparator, bb_upper)
        return condition_met
    elif level_type == 'bollinger_lower':
        # Try to get from dataframe first, fallback to LLM output
        if 'BB_Lower' in df.columns:
            bb_lower = df['BB_Lower'].iloc[-1]
        else:
            bb_lower = llm_output['technical_indicators']['BB20_2']['lower']
        condition_met = evaluate_comparison(current_price, comparator, bb_lower)
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
    
    if indicator_name == 'MACD12_26_9':
        if 'MACD_Line' not in df.columns or 'MACD_Signal' not in df.columns:
            return False
        
        current_macd_line = df['MACD_Line'].iloc[-1]
        current_macd_signal = df['MACD_Signal'].iloc[-1]
        prev_macd_line = df['MACD_Line'].iloc[-2] if len(df) > 1 else current_macd_line
        prev_macd_signal = df['MACD_Signal'].iloc[-2] if len(df) > 1 else current_macd_signal
        
        # Check if any values are None or NaN before comparison
        if (current_macd_line is None or pd.isna(current_macd_line) or
            current_macd_signal is None or pd.isna(current_macd_signal) or
            prev_macd_line is None or pd.isna(prev_macd_line) or
            prev_macd_signal is None or pd.isna(prev_macd_signal)):
            return False
        
        if crossover_condition == 'macd_line > signal_line':
            # Check if MACD line crossed above signal line
            return (current_macd_line > current_macd_signal) and (prev_macd_line <= prev_macd_signal)
        elif crossover_condition == 'macd_line < signal_line':
            # Check if MACD line crossed below signal line
            return (current_macd_line < current_macd_signal) and (prev_macd_line >= prev_macd_signal)
        elif crossover_condition == 'macd_line >= signal_line':
            # Check if MACD line is above or equal to signal line
            return current_macd_line >= current_macd_signal
        elif crossover_condition == 'macd_line <= signal_line':
            # Check if MACD line is below or equal to signal line
            return current_macd_line <= current_macd_signal
    
    elif indicator_name == 'STOCH14_3_3':
        if 'STOCH_K' not in df.columns or 'STOCH_D' not in df.columns:
            return False
        
        current_stoch_k = df['STOCH_K'].iloc[-1]
        current_stoch_d = df['STOCH_D'].iloc[-1]
        prev_stoch_k = df['STOCH_K'].iloc[-2] if len(df) > 1 else current_stoch_k
        prev_stoch_d = df['STOCH_D'].iloc[-2] if len(df) > 1 else current_stoch_d
        
        # Check if any values are None or NaN before comparison
        if (current_stoch_k is None or pd.isna(current_stoch_k) or
            current_stoch_d is None or pd.isna(current_stoch_d) or
            prev_stoch_k is None or pd.isna(prev_stoch_k) or
            prev_stoch_d is None or pd.isna(prev_stoch_d)):
            return False
        
        if crossover_condition == 'stoch_k > stoch_d':
            # Check if %K crossed above %D
            return (current_stoch_k > current_stoch_d) and (prev_stoch_k <= prev_stoch_d)
        elif crossover_condition == 'stoch_k < stoch_d':
            # Check if %K crossed below %D
            return (current_stoch_k < current_stoch_d) and (prev_stoch_k >= prev_stoch_d)
        elif crossover_condition == 'stoch_k >= stoch_d':
            # Check if %K is above or equal to %D
            return current_stoch_k >= current_stoch_d
        elif crossover_condition == 'stoch_k <= stoch_d':
            # Check if %K is below or equal to %D
            return current_stoch_k <= current_stoch_d
    
    # For other indicators, return False for now
    return False

def check_sequence_condition(df, condition):
    """Check sequence conditions (e.g., consecutive candles)"""
    return False

def check_pattern_breach(df, condition, llm_output):
    """Check pattern breach conditions (e.g., channel, triangle breaches)"""
    level = condition.get('level')
    comparator = condition.get('comparator', '>')
    
    # Guard against missing data
    if df is None or df.empty or 'Close' not in df.columns:
        return False
    
    current_price = df['Close'].iloc[-1]
    
    # Check if current_price is None or NaN before comparison
    if current_price is None or pd.isna(current_price):
        return False
    
    if level == 'channel_upper':
        # Check if price breaches above the upper channel boundary
        # This would typically come from the pattern analysis in llm_output
        # For now, we'll use a simple approach based on support/resistance levels
        resistance = llm_output.get('support_resistance', {}).get('resistance')
        if resistance is not None:
            condition_met = evaluate_comparison(current_price, comparator, resistance)
            return condition_met
    
    elif level == 'channel_lower':
        # Check if price breaches below the lower channel boundary
        support = llm_output.get('support_resistance', {}).get('support')
        if support is not None:
            condition_met = evaluate_comparison(current_price, comparator, support)
            return condition_met
    
    elif level == 'triangle_upper':
        # Check if price breaches above triangle upper boundary
        # This would need more complex pattern recognition
        # For now, use resistance level as proxy
        resistance = llm_output.get('support_resistance', {}).get('resistance')
        if resistance is not None:
            condition_met = evaluate_comparison(current_price, comparator, resistance)
            return condition_met
    
    elif level == 'triangle_lower':
        # Check if price breaches below triangle lower boundary
        support = llm_output.get('support_resistance', {}).get('support')
        if support is not None:
            condition_met = evaluate_comparison(current_price, comparator, support)
            return condition_met
    
    # For other pattern types, return False for now
    return False

def list_of_indicator_checker(llm_output):
    """Get list of technical indicators from LLM output"""
    technical_indicator_list = []
    for i in llm_output['opening_signal']['checklist']:
        if i['technical_indicator'] == True:
            technical_indicator_list.append(i)
    return technical_indicator_list

def indicator_checker(df, llm_output, emit_progress_fn=None):
    """Check if all technical indicator conditions are met"""
    technical_indicator_list = list_of_indicator_checker(llm_output)
    all_conditions_met = True
    conditions_met_count = 0
    total_conditions = len(technical_indicator_list)
    indicator_details = []

    for i in technical_indicator_list:
        indicator_name = i['indicator'] 
        indicator_type = i['type']
        condition_met = False
        current_value = None
        target_value = None
        condition_description = ""
        
        if indicator_type == 'indicator_threshold':
            condition_met = check_indicator_threshold(df, i)
            if indicator_name in df.columns:
                current_value = df[indicator_name].iloc[-1]
            elif indicator_name == 'MACD12_26_9' and 'MACD_Line' in df.columns:
                # Handle MACD12_26_9 as threshold - use MACD_Line value
                current_value = df['MACD_Line'].iloc[-1]
            elif indicator_name == 'STOCH14_3_3' and 'STOCH_K' in df.columns:
                # Handle STOCH14_3_3 as threshold - use STOCH_K value
                current_value = df['STOCH_K'].iloc[-1]
            elif indicator_name == 'BB20_2_PercentB' and 'BB_PercentB' in df.columns:
                # Handle BB20_2_PercentB as threshold - use BB_PercentB value
                current_value = df['BB_PercentB'].iloc[-1]
            elif indicator_name == 'BB20_2_Bandwidth' and 'BB_Bandwidth' in df.columns:
                # Handle BB20_2_Bandwidth as threshold - use BB_Bandwidth value
                current_value = df['BB_Bandwidth'].iloc[-1]
            target_value = i.get('value')
            comparator = i.get('comparator', '==')
            condition_description = f"{indicator_name} {comparator} {target_value}"
        elif indicator_type == 'indicator_crossover':
            condition_met = check_indicator_crossover(df, i)
            if indicator_name == 'MACD12_26_9' and 'MACD_Line' in df.columns and 'MACD_Signal' in df.columns:
                current_value = f"Line: {df['MACD_Line'].iloc[-1]:.4f}, Signal: {df['MACD_Signal'].iloc[-1]:.4f}"
            elif indicator_name == 'STOCH14_3_3' and 'STOCH_K' in df.columns and 'STOCH_D' in df.columns:
                current_value = f"K: {df['STOCH_K'].iloc[-1]:.4f}, D: {df['STOCH_D'].iloc[-1]:.4f}"
            condition_description = i.get('condition', f"{indicator_name} crossover")
        else:
            condition_met = False
            condition_description = f"{indicator_name} (unsupported type: {indicator_type})"
        
        # Store indicator details
        # Convert values to JSON-serializable types
        serializable_current_value = None
        if current_value is not None:
            if isinstance(current_value, str):
                serializable_current_value = current_value
            else:
                try:
                    # Handle pandas Timestamp and other non-serializable types
                    if hasattr(current_value, 'isoformat'):  # pandas Timestamp
                        serializable_current_value = current_value.isoformat()
                    elif hasattr(current_value, 'item'):  # numpy scalar
                        serializable_current_value = current_value.item()
                    else:
                        serializable_current_value = float(current_value)
                except (ValueError, TypeError):
                    serializable_current_value = str(current_value)
        
        serializable_target_value = None
        if target_value is not None:
            if isinstance(target_value, str):
                serializable_target_value = target_value
            else:
                try:
                    # Handle pandas Timestamp and other non-serializable types
                    if hasattr(target_value, 'isoformat'):  # pandas Timestamp
                        serializable_target_value = target_value.isoformat()
                    elif hasattr(target_value, 'item'):  # numpy scalar
                        serializable_target_value = target_value.item()
                    else:
                        serializable_target_value = float(target_value)
                except (ValueError, TypeError):
                    serializable_target_value = str(target_value)
        
        indicator_details.append({
            'name': indicator_name,
            'type': indicator_type,
            'condition': condition_description,
            'met': bool(condition_met),  # Convert numpy bool_ to Python bool
            'current_value': serializable_current_value,
            'target_value': serializable_target_value
        })
        
        # Count conditions that are met
        if condition_met:
            conditions_met_count += 1
        else:
            all_conditions_met = False
        
        # Emit progress if function provided
        if emit_progress_fn:
            emit_progress_fn(f"Indicator checker progress: {conditions_met_count}/{total_conditions}")
    
    # Emit detailed indicator status if function provided
    if emit_progress_fn and indicator_details:
        emit_progress_fn(json.dumps({
            "type": "indicator_details",
            "payload": {
                "total_indicators": total_conditions,
                "met_indicators": conditions_met_count,
                "indicators": make_json_serializable(indicator_details)
            }
        }))
    
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
        
        elif condition_type == 'pattern_breach':
            # Use shared pattern breach function
            condition_met = check_pattern_breach(df, condition, llm_output)
        
        else:
            print(f"Unknown invalidation condition type: {condition_type}")
            condition_met = False
        
        # Check if this invalidation condition is triggered
        if condition_met:
            invalidation_triggered = True
            triggered_conditions.append(condition_id)
    
    return invalidation_triggered, triggered_conditions

def validate_trading_signal(df, llm_output, emit_progress_fn=None):
    """Comprehensive trading signal validation combining checklist and invalidation checks"""
    # Check if all checklist conditions are met
    checklist_passed = indicator_checker(df, llm_output, emit_progress_fn)
    
    # Check if any invalidation conditions are triggered
    invalidation_triggered, triggered_conditions = invalidation_checker(df, llm_output)
    
    # Get current market values for the return (safely)
    if df is None or df.empty:
        current_price = None
        current_time = None
        current_rsi = None
        current_macd_line = None
        current_macd_signal = None
        current_macd_histogram = None
    else:
        current_price = df['Close'].iloc[-1] if 'Close' in df.columns else None
        current_time = df['Open_time'].iloc[-1] if 'Open_time' in df.columns else None
        current_rsi = df['RSI14'].iloc[-1] if 'RSI14' in df.columns else None
        current_macd_line = df['MACD_Line'].iloc[-1] if 'MACD_Line' in df.columns else None
        current_macd_signal = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns else None
        current_macd_histogram = df['MACD_Histogram'].iloc[-1] if 'MACD_Histogram' in df.columns else None
    
    market_values = {
        'current_price': current_price,
        'current_time': current_time,
        'current_rsi': current_rsi,
        'current_macd_line': current_macd_line,
        'current_macd_signal': current_macd_signal,
        'current_macd_histogram': current_macd_histogram
    }
    
    if invalidation_triggered:
        # Emit structured invalidation data to UI
        if emit_progress_fn:
            emit_progress_fn(json.dumps({
                "type": "invalidation_status",
                "payload": {
                    "status": "invalidated",
                    "triggered_conditions": triggered_conditions,
                    "market_values": make_json_serializable(market_values),
                    "message": f"Signal invalidated due to {len(triggered_conditions)} condition(s): {', '.join(triggered_conditions)}"
                }
            }))
        
        # Send iPhone notification for invalidated trade
        try:
            trade_data = {
                "symbol": llm_output.get("symbol", "Unknown"),
                "current_price": market_values.get("current_price", 0),
                "triggered_conditions": triggered_conditions,
                "current_rsi": market_values.get("current_rsi", 0)
            }
            notification_results = notify_invalidated_trade(trade_data)
            if emit_progress_fn:
                emit_progress_fn(f"📱 Invalidation notifications sent - Pushover: {'✅' if notification_results.get('pushover') else '❌'}, Email: {'✅' if notification_results.get('email') else '❌'}")
        except Exception as e:
            if emit_progress_fn:
                emit_progress_fn(f"⚠️ Invalidation notification failed: {str(e)}")
        
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

def poll_until_decision(symbol, timeframe, llm_output, max_cycles=None, emit_progress_fn=None):
    """Poll market data until trading decision is made"""
    cycles = 0
    wait_seconds = _timeframe_seconds(timeframe)
    
    while True:
        if emit_progress_fn:
            emit_progress_fn(f"Polling cycle {cycles + 1}: fetching fresh market data...")
        
        current_df = fetch_market_dataframe(symbol, timeframe)
        current_df = calculate_rsi14(current_df)
        current_df = calculate_macd12_26_9(current_df)
        current_df = calculate_stoch14_3_3(current_df)
        current_df = calculate_bb20_2(current_df)
        
        if emit_progress_fn:
            emit_progress_fn(f"Polling cycle {cycles + 1}: validating trading signal...")
        
        signal_valid, signal_status, triggered_conditions, market_values = validate_trading_signal(current_df, llm_output, emit_progress_fn)

        if emit_progress_fn:
            emit_progress_fn(f"Polling cycle {cycles + 1}: signal status = {signal_status}")

        if signal_status != "pending":
            if emit_progress_fn:
                emit_progress_fn(f"Polling complete: final status = {signal_status}")
            return signal_valid, signal_status, triggered_conditions, market_values
        
        cycles += 1

        if max_cycles is not None and cycles >= max_cycles:
            if emit_progress_fn:
                emit_progress_fn(f"Polling complete: max cycles ({max_cycles}) reached")
            return signal_valid, signal_status, triggered_conditions, market_values

        if emit_progress_fn:
            emit_progress_fn(f"Polling cycle {cycles + 1}: waiting {wait_seconds} seconds...")
        time.sleep(wait_seconds)

def run_trading_analysis(image_path: str, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> dict:
    """Run the complete trading analysis pipeline.

    If symbol/timeframe are provided, fetch klines and compute indicators FIRST,
    then pass both screenshot and DataFrame to the LLM for enriched analysis.
    Falls back to image-only analysis if data fetching fails.
    """
    try:
        # Step 1: Prepare market data BEFORE calling the LLM
        emit_progress("Step 1: Preparing market data (if provided)...", 1, 14)
        df = pd.DataFrame()
        effective_symbol = symbol or "BTCUSDT"
        effective_timeframe = timeframe or "1m"

        try:
            # Validate timeframe
            valid_intervals = ['1m', '5m', '15m', '30m', '60m', '4h', '1d', '1W', '1M']
            if effective_timeframe not in valid_intervals:
                emit_progress(f"Warning: {effective_timeframe} not valid. Using 1m.", 1, 14)
                effective_timeframe = '1m'

            # Initialize MEXC clients lazily here to fetch data
            from pymexc import spot
            public_spot_client_local = spot.HTTP()

            # Local light wrapper mirroring existing function signature
            def _fetch_market_dataframe(sym: str, tf: str) -> pd.DataFrame:
                kl = public_spot_client_local.klines(symbol=sym, interval=tf, limit=100)
                if not kl:
                    return pd.DataFrame()
                frame = pd.DataFrame(kl)
                frame.columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume']
                frame['Open_time'] = pd.to_datetime(frame['Open_time'], unit='ms')
                frame['Close_time'] = pd.to_datetime(frame['Close_time'], unit='ms')
                frame['Open'] = pd.to_numeric(frame['Open'])
                frame['High'] = pd.to_numeric(frame['High'])
                frame['Low'] = pd.to_numeric(frame['Low'])
                frame['Close'] = pd.to_numeric(frame['Close'])
                frame['Volume'] = pd.to_numeric(frame['Volume'])
                return frame

            df = _fetch_market_dataframe(effective_symbol, effective_timeframe)
            if df.empty:
                emit_progress("Step 1: No klines returned; proceeding without market data", 1, 14)
            else:
                # Calculate indicators we already support later in the pipeline
                df = calculate_rsi14(df)
                try:
                    df = calculate_macd12_26_9(df)
                except Exception:
                    # MACD function may not always be available; continue with RSI only
                    pass
                try:
                    df = calculate_stoch14_3_3(df)
                except Exception:
                    # STOCH function may not always be available; continue with other indicators
                    pass
                try:
                    df = calculate_bb20_2(df)
                except Exception:
                    # BB function may not always be available; continue with other indicators
                    pass
                emit_progress(f"Step 1 Complete: Market data prepared ({len(df)} candles)", 1, 14)
            
        except Exception as e:
            emit_progress(f"Step 1 Warning: Market data prep failed: {e}. Proceeding image-only.", 1, 14)

        # Step 2: Analyze trading chart with LLM (now with df if available)
        emit_progress("Step 2: Analyzing trading chart with LLM...", 2, 14)
        llm_output = analyze_trading_chart(
            image_path,
            symbol=effective_symbol,
            timeframe=effective_timeframe,
            df=df if not df.empty else None,
        )
        emit_progress("Step 2 Complete: Chart analysis finished", 2, 14)

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
        
        # Check if this is a fallback response due to LLM refusal
        is_fallback = llm_output.get('requires_manual_review', False)
        if is_fallback:
            emit_progress(f"Step 4 Warning: Using fallback response due to LLM refusal: {llm_output.get('fallback_reason', 'Unknown reason')}", 4, 14)
            emit_progress("Step 4 Warning: Analysis will continue but with limited functionality", 4, 14)
        
        emit_progress(f"Step 4 Complete: Symbol={symbol}, Timeframe={timeframe}, Screenshot time={time_of_screenshot}", 4, 14)

        # Emit proposed trade direction (from LLM opening signal) for UI
        try:
            proposed_direction = llm_output.get('opening_signal', {}).get('direction', 'unknown')
            emit_progress(
                json.dumps({
                    "type": "proposed_signal",
                    "payload": make_json_serializable({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "direction": proposed_direction
                    })
                })
            )
        except Exception:
            pass

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

        # Step 8: Calculate RSI14, MACD12_26_9, STOCH14_3_3, BB20_2, and ATR14
        emit_progress("Step 8: Calculating technical indicators...", 8, 14)
        df = calculate_rsi14(df)
        df = calculate_macd12_26_9(df)
        df = calculate_stoch14_3_3(df)
        df = calculate_bb20_2(df)
        df = calculate_atr14(df)
        emit_progress("Step 8 Complete: Technical indicators calculation finished", 8, 14)
        
        # Step 8.5: Initial indicator check to show progress
        emit_progress("Step 8.5: Running initial indicator check...", 8, 14)
        indicator_checker(df, llm_output, emit_progress)
        emit_progress("Step 8.5 Complete: Initial indicator check finished", 8, 14)

        # Step 9: Extract latest market indicators
        emit_progress("Step 9: Extracting latest market indicators...", 9, 14)
        if not df.empty and 'RSI14' in df.columns:
            latest_rsi = df['RSI14'].iloc[-1]
            emit_progress(f"Step 9 Complete: Latest RSI14 = {latest_rsi}", 9, 14)
        else:
            emit_progress("Step 9 Complete: No RSI14 data available", 9, 14)
        
        if not df.empty and 'MACD_Line' in df.columns:
            latest_macd_line = df['MACD_Line'].iloc[-1]
            latest_macd_signal = df['MACD_Signal'].iloc[-1]
            latest_macd_histogram = df['MACD_Histogram'].iloc[-1]
            emit_progress(f"Step 9 Complete: Latest MACD Line = {latest_macd_line:.4f}, Signal = {latest_macd_signal:.4f}, Histogram = {latest_macd_histogram:.4f}", 9, 14)
        else:
            emit_progress("Step 9 Complete: No MACD data available", 9, 14)

        # Step 10: Start signal validation polling
        emit_progress("Step 10: Starting signal validation polling...", 10, 14)
        
        # Check if this is a fallback response - skip signal validation if so
        if is_fallback:
            emit_progress("Step 10 Warning: Skipping signal validation due to LLM refusal fallback", 10, 14)
            signal_valid = False
            signal_status = "fallback_no_analysis"
            triggered_conditions = ["llm_refusal"]
            market_values = {
                'current_price': 0,
                'current_time': None,
                'current_rsi': None
            }
            emit_progress("Step 10 Complete: Signal validation skipped (fallback mode)", 10, 14)
        else:
            signal_valid, signal_status, triggered_conditions, market_values = poll_until_decision(symbol, timeframe, llm_output, emit_progress_fn=emit_progress)
            emit_progress(f"Step 10 Complete: Final Signal Status: {signal_status}", 10, 14)

        # Step 11: Check if signal is valid for trade gate
        emit_progress("Step 11: Checking if signal is valid for trade gate...", 11, 14)
        gate_result = None
        
        if is_fallback:
            emit_progress("Step 11 Warning: Skipping trade gate due to LLM refusal fallback", 11, 14)
            gate_result = {
                "should_open": False,
                "direction": "none",
                "confidence": 0.0,
                "reasons": ["LLM refused to analyze chart - manual review required"],
                "warnings": ["No AI analysis available", "Trading decision requires manual review"],
                "execution": {"status": "blocked", "reason": "llm_refusal"},
                "checks": {"llm_analysis": False, "manual_review_required": True}
            }
            emit_progress("Step 11 Complete: Trade gate skipped (fallback mode)", 11, 14)
        elif signal_valid and signal_status == "valid":
            emit_progress("Step 11 Complete: Signal is valid, proceeding to trade gate", 11, 14)
            emit_progress("Step 12: Running LLM trade gate decision...", 12, 14)
            
            checklist_passed = True
            invalidation_triggered_recent = len(triggered_conditions) > 0
            emit_progress(f"Step 12: Gate input - Checklist passed: {checklist_passed}, Invalidation triggered: {invalidation_triggered_recent}")
            emit_progress(f"Step 12: Market values - Price: ${market_values.get('current_price', 0):.2f}, RSI: {market_values.get('current_rsi', 0):.2f}")
            
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

            # Emit structured gate outcome to UI
            try:
                emit_progress(
                    json.dumps({
                        "type": "gate_outcome",
                        "payload": make_json_serializable({
                            "should_open": bool(gate_result.get("should_open", False)),
                            "direction": gate_result.get("direction", "unknown"),
                            "confidence": float(gate_result.get("confidence", 0.0) or 0.0),
                            "reasons": gate_result.get("reasons", [])[:4],
                            "warnings": gate_result.get("warnings", [])[:4],
                            "execution": gate_result.get("execution", {}),
                            "checks": gate_result.get("checks", {})
                        })
                    })
                )
            except Exception:
                # Fallback to a readable line if serialization fails
                emit_progress(f"Gate: Outcome -> should_open={gate_result.get('should_open')}, direction={gate_result.get('direction')}, confidence={gate_result.get('confidence')}")

            emit_progress("Step 14: Checking if trade should be opened...", 14, 14)
            if gate_result.get("should_open") is True:
                confidence = gate_result.get("confidence", 0.0)
                direction = gate_result.get("direction", "unknown")
                entry_price = gate_result.get("execution", {}).get("entry_price", 0)
                stop_loss = gate_result.get("execution", {}).get("stop_loss", 0)
                risk_reward = gate_result.get("execution", {}).get("risk_reward", 0)
                take_profits = gate_result.get("execution", {}).get("take_profits", [])
                
                emit_progress(f"Step 14 Complete: Trade approved for opening - Direction: {direction}, Confidence: {confidence:.2f}", 14, 14)
                emit_progress(f"🚀 TRADE APPROVED: {direction.upper()} at ${entry_price:.2f}, SL: ${stop_loss:.2f}, R/R: {risk_reward:.2f}", 14, 14)
                
                # Place order on MEXC Futures (guarded by env flag MEXC_ENABLE_ORDERS)
                try:
                    if os.getenv("MEXC_ENABLE_ORDERS", "false").lower() in ("1", "true", "yes"): 
                        emit_progress("Step 15: Placing MEXC futures order...", 15, 15)
                        order_outcome = open_trade_with_mexc(llm_output.get("symbol", ""), gate_result)
                        if order_outcome.get("ok"):
                            # Save order response
                            try:
                                order_filename = f"mexc_order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                order_path = os.path.join(output_dir, order_filename)
                                with open(order_path, 'w') as f:
                                    json.dump(order_outcome.get("response"), f, indent=2, default=str)
                                emit_progress(f"Saved order response to {order_path}")
                            except Exception as e:
                                emit_progress(f"⚠️ Failed to save order response: {str(e)}")
                        else:
                            emit_progress(f"Step 15 Failed: Order placement error - {order_outcome.get('error')}")
                    else:
                        emit_progress("Step 15: Live order placement disabled (MEXC_ENABLE_ORDERS is false)")
                except Exception as e:
                    emit_progress(f"⚠️ Order placement exception: {str(e)}")

                # Send iPhone notification for valid trade
                try:
                    trade_data = {
                        "symbol": llm_output.get("symbol", "Unknown"),
                        "direction": direction,
                        "current_price": entry_price,
                        "confidence": confidence,
                        "current_rsi": market_values.get("current_rsi", 0),
                        "stop_loss": stop_loss,
                        "risk_reward": risk_reward,
                        "take_profits": take_profits
                    }
                    notification_results = notify_valid_trade(trade_data)
                    emit_progress(f"📱 Notifications sent - Pushover: {'✅' if notification_results.get('pushover') else '❌'}, Email: {'✅' if notification_results.get('email') else '❌'}", 14, 14)
                except Exception as e:
                    emit_progress(f"⚠️ Notification failed: {str(e)}", 14, 14)
                
                if take_profits:
                    tp_info = ", ".join([f"TP{i+1}: ${tp.get('price', 0):.2f}" for i, tp in enumerate(take_profits)])
                    emit_progress(f"Take Profits: {tp_info}", 14, 14)
            else:
                reasons = gate_result.get("reasons", ["No specific reason provided"])
                direction = gate_result.get("direction", "unknown")
                confidence = gate_result.get("confidence", 0.0)
                emit_progress(f"Step 14 Complete: Trade not approved by gate - Direction: {direction}, Confidence: {confidence:.2f}", 14, 14)
                emit_progress(f"❌ TRADE REJECTED: {direction.upper()} - Reasons: {'; '.join(reasons[:2])}", 14, 14)
        else:
            emit_progress(f"Step 11 Complete: Signal not valid (status: {signal_status}), skipping trade gate", 11, 14)

        return {
            "success": True,
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
        emit_progress(f"ERROR: Analysis failed with {type(e).__name__}: {str(e)}")
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
        <title>Trrrrr</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .paste-section { margin: 15px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .loading { background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            button:disabled { background-color: #6c757d; cursor: not-allowed; }
            #clearBtn:hover { background-color: #c82333; }
        </style>
    </head>
    <body>
        <h1>Trading Chart Analysis</h1>
        <p>Upload a trading chart image to analyze it using AI and get trading signals, or resume analysis from a previous session.</p>
        
        <!-- New Analysis Section -->
        <div style="margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background-color: #f8f9fa;">
            <h3 style="margin-top: 0; color: #495057;">🆕 New Analysis</h3>
            <form id="uploadForm" class="upload-form" enctype="multipart/form-data">
                <input type="file" id="imageFile" name="image" accept="image/*" required>
                <br><br>
                <div class="paste-section">
                    <button type="button" id="pasteBtn" style="background-color: #28a745; margin-right: 10px;">📋 Paste from Clipboard</button>
                    <span id="pasteStatus" style="color: #666; font-size: 14px;"></span>
                    <div style="font-size: 12px; color: #666; margin-top: 5px;">
                        💡 Tip: You can also use Ctrl+V (or Cmd+V on Mac) to paste images
                    </div>
                </div>
                <br>
                <button type="submit" id="submitBtn">Analyze Chart</button>
            </form>
        </div>
        
        <!-- Resume Analysis Section -->
        <div style="margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background-color: #e8f4fd;">
            <h3 style="margin-top: 0; color: #004085;">🔄 Resume Analysis</h3>
            <p style="color: #495057; margin-bottom: 15px;">Select a previous analysis to resume monitoring and validation:</p>
            <button type="button" id="loadFilesBtn" style="background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 15px;">📁 Load Previous Analyses</button>
            <div id="jsonFilesList" style="display: none;">
                <div id="filesContainer" style="max-height: 300px; overflow-y: auto; border: 1px solid #ccc; border-radius: 5px; padding: 10px; background-color: white;">
                    <!-- Files will be loaded here -->
                </div>
            </div>
        </div>
        
        <!-- Control Buttons -->
        <div style="text-align: center; margin: 20px 0;">
            <button type="button" id="clearBtn" style="background-color: #dc3545; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">Clear All</button>
        </div>
        
        <div id="result"></div>
        <div id="progress" style="display: none;">
            <h3>Analysis Progress</h3>
            <div id="progressBar" style="width: 100%; background-color: #f0f0f0; border-radius: 5px; margin: 10px 0;">
                <div id="progressFill" style="width: 0%; height: 20px; background-color: #007bff; border-radius: 5px; transition: width 0.3s;"></div>
            </div>

            <!-- Proposed Trade Signal Summary -->
            <div id="proposedBox" style="display:none; margin: 12px 0; padding: 12px; background-color: #fff; border: 1px solid #ddd; border-radius: 5px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; color:#343a40;">Proposed Trade</h4>
                    <span id="proposedBadge" style="font-weight:bold; padding:2px 8px; border-radius:12px; border:1px solid #ccc; color:#6c757d; background:#f1f3f5; text-transform:uppercase;">unknown</span>
                </div>
                <div style="margin-top:8px; color:#495057;">
                    <span><strong>Symbol:</strong> <span id="proposedSymbol">-</span></span>
                    <span style="margin-left:12px;"><strong>Timeframe:</strong> <span id="proposedTimeframe">-</span></span>
                </div>
            </div>
            
            <!-- Invalidation Status Alert -->
            <div id="invalidationAlert" style="display: none; margin: 15px 0; padding: 15px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">
                <h4 style="margin: 0 0 10px 0; color: #721c24;">⚠️ Signal Invalidated</h4>
                <div id="invalidationMessage" style="color: #721c24; font-weight: bold; margin-bottom: 10px;">
                    <!-- Invalidation message will appear here -->
                </div>
                <div id="invalidationDetails" style="background-color: #fff; padding: 10px; border-radius: 3px; border: 1px solid #f5c6cb;">
                    <h5 style="margin: 0 0 8px 0; color: #721c24;">Triggered Conditions:</h5>
                    <div id="triggeredConditionsList" style="color: #721c24;">
                        <!-- Triggered conditions will appear here -->
                    </div>
                </div>
            </div>
            
            <!-- Indicator Checker Progress -->
            <div id="indicatorProgress" style="display: none; margin: 15px 0; padding: 10px; background-color: #e8f4fd; border: 1px solid #b8daff; border-radius: 5px;">
                <h4 style="margin: 0 0 10px 0; color: #004085;">📊 Indicator Checker Progress</h4>
                <div id="indicatorProgressBar" style="width: 100%; background-color: #f0f0f0; border-radius: 5px; margin: 5px 0;">
                    <div id="indicatorProgressFill" style="width: 0%; height: 15px; background-color: #28a745; border-radius: 5px; transition: width 0.3s;"></div>
                </div>
                <div id="indicatorProgressText" style="text-align: center; font-weight: bold; color: #155724; margin-top: 5px;">
                    Ready to check indicators...
                </div>
                
                <!-- Individual Indicator Status -->
                <div id="indicatorList" style="margin-top: 15px; display: none;">
                    <h5 style="margin: 0 0 10px 0; color: #004085;">Individual Indicators:</h5>
                    <div id="indicatorItems" style="max-height: 200px; overflow-y: auto;">
                        <!-- Indicator items will be populated here -->
                    </div>
                </div>
            </div>
            
            <div id="progressMessages" style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
                <!-- Progress messages will appear here -->
            </div>
        </div>
        
        <script>
            let eventSource = null;
            let analysisComplete = false;
            
            // Check if Clipboard API is supported
            if (!navigator.clipboard || !navigator.clipboard.read) {
                document.getElementById('pasteBtn').disabled = true;
                document.getElementById('pasteBtn').textContent = '📋 Paste (Not Supported)';
                document.getElementById('pasteBtn').style.backgroundColor = '#6c757d';
                document.querySelector('.paste-section div').textContent = '❌ Clipboard API not supported in this browser. Please use the file upload instead.';
                document.querySelector('.paste-section div').style.color = '#dc3545';
            }
            
            // Clipboard paste functionality
            document.getElementById('pasteBtn').addEventListener('click', async function() {
                const pasteStatus = document.getElementById('pasteStatus');
                const imageFile = document.getElementById('imageFile');
                
                try {
                    pasteStatus.textContent = 'Reading clipboard...';
                    
                    // Request clipboard access
                    const clipboardItems = await navigator.clipboard.read();
                    
                    for (const clipboardItem of clipboardItems) {
                        // Check if clipboard contains image data
                        for (const type of clipboardItem.types) {
                            if (type.startsWith('image/')) {
                                const blob = await clipboardItem.getType(type);
                                
                                // Create a File object from the blob
                                const file = new File([blob], 'clipboard-image.png', { type: type });
                                
                                // Create a new FileList-like object
                                const dataTransfer = new DataTransfer();
                                dataTransfer.items.add(file);
                                
                                // Set the file input value
                                imageFile.files = dataTransfer.files;
                                
                                pasteStatus.textContent = '✅ Image pasted successfully!';
                                pasteStatus.style.color = '#28a745';
                                
                                // Clear status after 3 seconds
                                setTimeout(() => {
                                    pasteStatus.textContent = '';
                                }, 3000);
                                
                                return;
                            }
                        }
                    }
                    
                    pasteStatus.textContent = '❌ No image found in clipboard';
                    pasteStatus.style.color = '#dc3545';
                    
                    // Clear status after 3 seconds
                    setTimeout(() => {
                        pasteStatus.textContent = '';
                    }, 3000);
                    
                } catch (error) {
                    console.error('Clipboard access failed:', error);
                    
                    if (error.name === 'NotAllowedError') {
                        pasteStatus.textContent = '❌ Clipboard access denied. Please allow clipboard access and try again.';
                    } else if (error.name === 'NotSupportedError') {
                        pasteStatus.textContent = '❌ Clipboard API not supported in this browser.';
                    } else {
                        pasteStatus.textContent = '❌ Clipboard access failed. Please try uploading a file instead.';
                    }
                    pasteStatus.style.color = '#dc3545';
                    
                    // Clear status after 5 seconds
                    setTimeout(() => {
                        pasteStatus.textContent = '';
                    }, 5000);
                }
            });
            
            // Also handle Ctrl+V keyboard shortcut
            document.addEventListener('keydown', async function(e) {
                if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
                    // Only handle paste if the file input is focused or no input is focused
                    const activeElement = document.activeElement;
                    if (!activeElement || activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA') {
                        return; // Let normal paste behavior work for text inputs
                    }
                    
                    e.preventDefault();
                    document.getElementById('pasteBtn').click();
                }
            });
            
            // Load previous analyses functionality
            document.getElementById('loadFilesBtn').addEventListener('click', async function() {
                const loadBtn = document.getElementById('loadFilesBtn');
                const filesList = document.getElementById('jsonFilesList');
                const filesContainer = document.getElementById('filesContainer');
                
                loadBtn.disabled = true;
                loadBtn.textContent = 'Loading...';
                
                try {
                    const response = await fetch('/list-json-files');
                    const data = await response.json();
                    
                    if (data.success) {
                        filesContainer.innerHTML = '';
                        
                        if (data.files.length === 0) {
                            filesContainer.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">No previous analyses found</div>';
                        } else {
                            data.files.forEach(file => {
                                const fileDiv = document.createElement('div');
                                fileDiv.style.cssText = 'margin-bottom: 10px; padding: 12px; border: 1px solid #ddd; border-radius: 5px; background-color: #fff; cursor: pointer; transition: background-color 0.2s;';
                                
                                const directionColor = file.direction === 'long' ? '#28a745' : file.direction === 'short' ? '#dc3545' : '#6c757d';
                                const isMetColor = file.is_met ? '#28a745' : '#dc3545';
                                const isMetText = file.is_met ? 'MET' : 'NOT MET';
                                
                                fileDiv.innerHTML = `
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <div style="font-weight: bold; color: #495057; margin-bottom: 4px;">${file.symbol} (${file.timeframe})</div>
                                            <div style="font-size: 12px; color: #666; margin-bottom: 4px;">
                                                <span style="color: ${directionColor}; font-weight: bold;">${file.direction.toUpperCase()}</span> • 
                                                <span style="color: ${isMetColor}; font-weight: bold;">${isMetText}</span> • 
                                                ${file.time_of_screenshot}
                                            </div>
                                            <div style="font-size: 11px; color: #999;">${file.created}</div>
                                        </div>
                                        <button class="resumeBtn" style="background-color: #007bff; color: white; border: none; padding: 6px 12px; border-radius: 3px; cursor: pointer; font-size: 12px;" data-filepath="${file.filepath}">
                                            Resume
                                        </button>
                                    </div>
                                `;
                                
                                // Add hover effect
                                fileDiv.addEventListener('mouseenter', function() {
                                    this.style.backgroundColor = '#f8f9fa';
                                });
                                fileDiv.addEventListener('mouseleave', function() {
                                    this.style.backgroundColor = '#fff';
                                });
                                
                                filesContainer.appendChild(fileDiv);
                            });
                            
                            // Add event listeners to resume buttons
                            document.querySelectorAll('.resumeBtn').forEach(btn => {
                                btn.addEventListener('click', function(e) {
                                    e.stopPropagation();
                                    resumeAnalysis(this.dataset.filepath);
                                });
                            });
                        }
                        
                        filesList.style.display = 'block';
                        loadBtn.textContent = 'Refresh List';
                    } else {
                        alert('Error loading files: ' + data.error);
                        loadBtn.textContent = 'Load Previous Analyses';
                    }
                } catch (error) {
                    alert('Error loading files: ' + error.message);
                    loadBtn.textContent = 'Load Previous Analyses';
                } finally {
                    loadBtn.disabled = false;
                }
            });
            
            // Resume analysis function
            async function resumeAnalysis(filepath) {
                const resultDiv = document.getElementById('result');
                const progressDiv = document.getElementById('progress');
                const progressMessages = document.getElementById('progressMessages');
                const progressFill = document.getElementById('progressFill');
                
                try {
                    const response = await fetch('/resume-analysis', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ filepath: filepath })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        // Clear previous results
                        resultDiv.innerHTML = '';
                        progressDiv.style.display = 'block';
                        progressMessages.innerHTML = '';
                        progressFill.style.width = '0%';
                        
                        // Hide other sections
                        document.getElementById('invalidationAlert').style.display = 'none';
                        document.getElementById('indicatorProgress').style.display = 'none';
                        document.getElementById('proposedBox').style.display = 'none';
                        
                        // Start listening for progress updates
                        if (eventSource) {
                            eventSource.close();
                        }
                        
                        eventSource = new EventSource('/progress');
                        eventSource.onmessage = function(event) {
                            const data = JSON.parse(event.data);
                            
                            if (data.message === 'keep-alive') {
                                return;
                            }
                            
                            // Handle all the same progress updates as new analysis
                            try {
                                const maybe = JSON.parse(data.message);
                                if (maybe && maybe.type === 'gate_outcome' && maybe.payload) {
                                    const p = maybe.payload;
                                    const approved = !!p.should_open;
                                    const badge = approved ? '<span style="color:#155724;background:#d4edda;border:1px solid #c3e6cb;padding:2px 6px;border-radius:3px;">APPROVED</span>' : '<span style="color:#721c24;background:#f8d7da;border:1px solid #f5c6cb;padding:2px 6px;border-radius:3px;">REJECTED</span>';
                                    
                                    let gateDisplay = `
                                        <div class="result ${approved ? 'success' : 'error'}">
                                            <h3>Resumed Analysis - Gate Decision ${badge}</h3>
                                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0;">
                                                <div>
                                                    <h4 style="margin: 0 0 10px 0; color: ${approved ? '#155724' : '#721c24'};">Trade Parameters</h4>
                                                    <p><strong>Direction:</strong> <span style="color: ${p.direction === 'long' ? '#28a745' : '#dc3545'}; font-weight: bold; text-transform: uppercase;">${p.direction || 'unknown'}</span></p>
                                                    <p><strong>Confidence:</strong> ${(p.confidence ?? 0).toFixed(2)}/1.0</p>
                                                    <p><strong>Entry Type:</strong> ${p.execution?.entry_type || 'market'}</p>
                                                    <p><strong>Entry Price:</strong> $${(p.execution?.entry_price || 0).toFixed(2)}</p>
                                                </div>
                                                <div>
                                                    <h4 style="margin: 0 0 10px 0; color: ${approved ? '#155724' : '#721c24'};">Risk Management</h4>
                                                    <p><strong>Stop Loss:</strong> $${(p.execution?.stop_loss || 0).toFixed(2)}</p>
                                                    <p><strong>Risk/Reward:</strong> ${(p.execution?.risk_reward || 0).toFixed(2)}</p>
                                                    <p><strong>Position Size:</strong> ${p.execution?.position_size_note || 'N/A'}</p>
                                                </div>
                                            </div>
                                    `;
                                    
                                    // Add all the same gate display logic as in new analysis
                                    if (p.execution?.take_profits && Array.isArray(p.execution.take_profits) && p.execution.take_profits.length > 0) {
                                        gateDisplay += `
                                            <div style="margin: 15px 0;">
                                                <h4 style="margin: 0 0 10px 0; color: ${approved ? '#155724' : '#721c24'};">Take Profit Levels</h4>
                                                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                                        `;
                                        p.execution.take_profits.forEach((tp, index) => {
                                            gateDisplay += `
                                                <div style="background: #e8f5e8; border: 1px solid #c3e6cb; padding: 8px 12px; border-radius: 4px; text-align: center;">
                                                    <div style="font-weight: bold; color: #155724;">TP${index + 1}</div>
                                                    <div style="color: #155724;">$${(tp.price || 0).toFixed(2)}</div>
                                                    <div style="font-size: 12px; color: #666;">${((tp.portion || 0) * 100).toFixed(0)}%</div>
                                                </div>
                                            `;
                                        });
                                        gateDisplay += `</div></div>`;
                                    }
                                    
                                    if (Array.isArray(p.reasons) && p.reasons.length) {
                                        gateDisplay += `
                                            <div style="margin: 15px 0;">
                                                <h4 style="margin: 0 0 10px 0; color: ${approved ? '#155724' : '#721c24'};">Reasons for Decision</h4>
                                                <ul style="margin: 0; padding-left: 20px;">
                                        `;
                                        p.reasons.slice(0, 5).forEach(reason => {
                                            gateDisplay += `<li style="margin: 5px 0; color: ${approved ? '#155724' : '#721c24'};">${reason}</li>`;
                                        });
                                        gateDisplay += `</ul></div>`;
                                    }
                                    
                                    if (Array.isArray(p.warnings) && p.warnings.length) {
                                        gateDisplay += `
                                            <div style="margin: 15px 0; padding: 10px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px;">
                                                <h4 style="margin: 0 0 10px 0; color: #856404;">⚠️ Warnings</h4>
                                                <ul style="margin: 0; padding-left: 20px;">
                                        `;
                                        p.warnings.slice(0, 3).forEach(warning => {
                                            gateDisplay += `<li style="margin: 5px 0; color: #856404;">${warning}</li>`;
                                        });
                                        gateDisplay += `</ul></div>`;
                                    }
                                    
                                    gateDisplay += `</div>`;
                                    resultDiv.innerHTML = gateDisplay;
                                } else if (maybe && maybe.type === 'proposed_signal' && maybe.payload) {
                                    // Handle proposed signal updates
                                    const p = maybe.payload;
                                    const proposedBox = document.getElementById('proposedBox');
                                    const proposedBadge = document.getElementById('proposedBadge');
                                    const proposedSymbol = document.getElementById('proposedSymbol');
                                    const proposedTimeframe = document.getElementById('proposedTimeframe');
                                    
                                    proposedSymbol.textContent = p.symbol || '-';
                                    proposedTimeframe.textContent = p.timeframe || '-';
                                    const dir = (p.direction || 'unknown').toLowerCase();
                                    proposedBadge.textContent = dir;
                                    if (dir === 'long') {
                                        proposedBadge.style.color = '#155724';
                                        proposedBadge.style.backgroundColor = '#d4edda';
                                        proposedBadge.style.borderColor = '#c3e6cb';
                                    } else if (dir === 'short') {
                                        proposedBadge.style.color = '#721c24';
                                        proposedBadge.style.backgroundColor = '#f8d7da';
                                        proposedBadge.style.borderColor = '#f5c6cb';
                                    } else {
                                        proposedBadge.style.color = '#6c757d';
                                        proposedBadge.style.backgroundColor = '#f1f3f5';
                                        proposedBadge.style.borderColor = '#ccc';
                                    }
                                    proposedBox.style.display = 'block';
                                } else if (maybe && maybe.type === 'invalidation_status' && maybe.payload) {
                                    // Handle invalidation status updates
                                    const payload = maybe.payload;
                                    const invalidationAlert = document.getElementById('invalidationAlert');
                                    const invalidationMessage = document.getElementById('invalidationMessage');
                                    const triggeredConditionsList = document.getElementById('triggeredConditionsList');
                                    
                                    invalidationAlert.style.display = 'block';
                                    invalidationMessage.textContent = payload.message || 'Signal has been invalidated';
                                    
                                    if (payload.triggered_conditions && payload.triggered_conditions.length > 0) {
                                        triggeredConditionsList.innerHTML = payload.triggered_conditions.map(condition => 
                                            `<div style="margin: 4px 0; padding: 4px; background-color: #f8d7da; border-radius: 3px; border-left: 3px solid #dc3545;">
                                                <strong>${condition}</strong>
                                            </div>`
                                        ).join('');
                                    } else {
                                        triggeredConditionsList.innerHTML = '<div style="color: #666;">No specific conditions identified</div>';
                                    }
                                    
                                    resultDiv.innerHTML = `
                                        <div class="result error">
                                            <h3>⚠️ Signal Invalidated (Resumed Analysis)</h3>
                                            <p><strong>Status:</strong> ${payload.status}</p>
                                            <p><strong>Message:</strong> ${payload.message}</p>
                                            ${payload.triggered_conditions && payload.triggered_conditions.length ? 
                                                `<p><strong>Triggered Conditions:</strong> ${payload.triggered_conditions.join(', ')}</p>` : ''}
                                            <p style="color: #666; font-size: 14px; margin-top: 10px;">
                                                The trading signal has been invalidated and will not proceed to the trade gate.
                                            </p>
                                        </div>
                                    `;
                                } else if (maybe && maybe.type === 'indicator_details' && maybe.payload) {
                                    // Handle detailed indicator information
                                    const payload = maybe.payload;
                                    const indicators = payload.indicators || [];
                                    const indicatorProgress = document.getElementById('indicatorProgress');
                                    const indicatorList = document.getElementById('indicatorList');
                                    const indicatorItems = document.getElementById('indicatorItems');
                                    
                                    indicatorList.style.display = 'block';
                                    indicatorItems.innerHTML = '';
                                    
                                    indicators.forEach((indicator, index) => {
                                        const itemDiv = document.createElement('div');
                                        itemDiv.style.marginBottom = '8px';
                                        itemDiv.style.padding = '8px';
                                        itemDiv.style.borderRadius = '4px';
                                        itemDiv.style.border = '1px solid #ddd';
                                        itemDiv.style.backgroundColor = indicator.met ? '#d4edda' : '#f8d7da';
                                        
                                        const statusIcon = indicator.met ? '✅' : '❌';
                                        const statusText = indicator.met ? 'MET' : 'NOT MET';
                                        const statusColor = indicator.met ? '#155724' : '#721c24';
                                        
                                        let valueText = '';
                                        if (indicator.current_value !== null && indicator.current_value !== undefined) {
                                            valueText = `<br><small style="color: #666;">Current: ${indicator.current_value}</small>`;
                                        }
                                        if (indicator.target_value !== null && indicator.target_value !== undefined) {
                                            valueText += `<br><small style="color: #666;">Target: ${indicator.target_value}</small>`;
                                        }
                                        
                                        itemDiv.innerHTML = `
                                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                                <div>
                                                    <strong style="color: ${statusColor};">${statusIcon} ${indicator.name}</strong>
                                                    <br><small style="color: #666;">${indicator.condition}</small>
                                                    ${valueText}
                                                </div>
                                                <span style="color: ${statusColor}; font-weight: bold; font-size: 12px;">${statusText}</span>
                                            </div>
                                        `;
                                        
                                        indicatorItems.appendChild(itemDiv);
                                    });
                                }
                            } catch (_) {
                                // not a structured payload; continue normal flow
                            }

                            // Handle indicator checker progress updates
                            if (data.message.includes('Indicator checker progress:')) {
                                const match = data.message.match(/(\\d+)\\/(\\d+)/);
                                if (match) {
                                    const met = parseInt(match[1]);
                                    const total = parseInt(match[2]);
                                    const percentage = (met / total) * 100;
                                    
                                    const indicatorProgress = document.getElementById('indicatorProgress');
                                    const indicatorProgressFill = document.getElementById('indicatorProgressFill');
                                    const indicatorProgressText = document.getElementById('indicatorProgressText');
                                    
                                    indicatorProgress.style.display = 'block';
                                    indicatorProgressFill.style.width = percentage + '%';
                                    indicatorProgressText.textContent = `${met}/${total} indicators met (${percentage.toFixed(1)}%)`;
                                    
                                    if (percentage === 100) {
                                        indicatorProgressFill.style.backgroundColor = '#28a745';
                                        indicatorProgressText.style.color = '#155724';
                                    } else if (percentage >= 50) {
                                        indicatorProgressFill.style.backgroundColor = '#ffc107';
                                        indicatorProgressText.style.color = '#856404';
                                    } else {
                                        indicatorProgressFill.style.backgroundColor = '#dc3545';
                                        indicatorProgressText.style.color = '#721c24';
                                    }
                                }
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
                                    document.getElementById('invalidationAlert').style.display = 'none';
                                    document.getElementById('indicatorProgress').style.display = 'none';
                                    document.getElementById('indicatorList').style.display = 'none';
                                }, 2000);
                            }
                        };
                        
                        eventSource.onerror = function(event) {
                            console.error('EventSource failed:', event);
                            if (eventSource) {
                                eventSource.close();
                            }
                        };
                        
                        analysisComplete = false;
                    } else {
                        alert('Error resuming analysis: ' + data.error);
                    }
                } catch (error) {
                    alert('Error resuming analysis: ' + error.message);
                }
            }

            // Clear button functionality
            document.getElementById('clearBtn').addEventListener('click', function() {
                // Clear file input
                document.getElementById('imageFile').value = '';
                
                // Clear paste status
                document.getElementById('pasteStatus').textContent = '';
                
                // Clear result div
                document.getElementById('result').innerHTML = '';
                
                // Hide and reset progress section
                const progressDiv = document.getElementById('progress');
                progressDiv.style.display = 'none';
                
                // Reset progress bar
                document.getElementById('progressFill').style.width = '0%';
                
                // Clear progress messages
                document.getElementById('progressMessages').innerHTML = '';
                
                // Hide and reset invalidation alert
                const invalidationAlert = document.getElementById('invalidationAlert');
                invalidationAlert.style.display = 'none';
                document.getElementById('invalidationMessage').textContent = '';
                document.getElementById('triggeredConditionsList').innerHTML = '';
                
                // Hide and reset indicator progress
                const indicatorProgress = document.getElementById('indicatorProgress');
                indicatorProgress.style.display = 'none';
                document.getElementById('indicatorProgressFill').style.width = '0%';
                document.getElementById('indicatorProgressText').textContent = 'Ready to check indicators...';
                document.getElementById('indicatorList').style.display = 'none';
                document.getElementById('indicatorItems').innerHTML = '';
                
                // Hide JSON files list
                document.getElementById('jsonFilesList').style.display = 'none';
                document.getElementById('loadFilesBtn').textContent = '📁 Load Previous Analyses';
                
                // Reset submit button
                const submitBtn = document.getElementById('submitBtn');
                submitBtn.disabled = false;
                submitBtn.textContent = 'Analyze Chart';
                
                // Close any existing event source
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                }
                
                // Reset analysis complete flag
                analysisComplete = false;
            });
            
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('imageFile');
                const submitBtn = document.getElementById('submitBtn');
                const resultDiv = document.getElementById('result');
                const progressDiv = document.getElementById('progress');
                const progressMessages = document.getElementById('progressMessages');
                const progressFill = document.getElementById('progressFill');
                const indicatorProgress = document.getElementById('indicatorProgress');
                const indicatorProgressFill = document.getElementById('indicatorProgressFill');
                const indicatorProgressText = document.getElementById('indicatorProgressText');
                const indicatorList = document.getElementById('indicatorList');
                const indicatorItems = document.getElementById('indicatorItems');
                const proposedBox = document.getElementById('proposedBox');
                const proposedBadge = document.getElementById('proposedBadge');
                const proposedSymbol = document.getElementById('proposedSymbol');
                const proposedTimeframe = document.getElementById('proposedTimeframe');
                
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
                
                // Hide invalidation alert
                const invalidationAlert = document.getElementById('invalidationAlert');
                invalidationAlert.style.display = 'none';
                document.getElementById('invalidationMessage').textContent = '';
                document.getElementById('triggeredConditionsList').innerHTML = '';
                
                indicatorProgress.style.display = 'none';
                indicatorProgressFill.style.width = '0%';
                indicatorProgressText.textContent = 'Ready to check indicators...';
                indicatorList.style.display = 'none';
                indicatorItems.innerHTML = '';
                // Reset proposed box
                proposedBox.style.display = 'none';
                proposedBadge.textContent = 'unknown';
                proposedBadge.style.color = '#6c757d';
                proposedBadge.style.backgroundColor = '#f1f3f5';
                proposedBadge.style.borderColor = '#ccc';
                proposedSymbol.textContent = '-';
                proposedTimeframe.textContent = '-';
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
                    
                    // Intercept structured gate outcome payloads
                    try {
                        const maybe = JSON.parse(data.message);
                        if (maybe && maybe.type === 'gate_outcome' && maybe.payload) {
                            const resultDiv = document.getElementById('result');
                            const p = maybe.payload;
                            const approved = !!p.should_open;
                            const badge = approved ? '<span style="color:#155724;background:#d4edda;border:1px solid #c3e6cb;padding:2px 6px;border-radius:3px;">APPROVED</span>' : '<span style="color:#721c24;background:#f8d7da;border:1px solid #f5c6cb;padding:2px 6px;border-radius:3px;">REJECTED</span>';
                            // Create comprehensive gate display
                            let gateDisplay = `
                                <div class="result ${approved ? 'success' : 'error'}" style="border: 2px solid ${approved ? '#28a745' : '#dc3545'}; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid ${approved ? '#c3e6cb' : '#f5c6cb'};">
                                        <h3 style="margin: 0; font-size: 24px;">${approved ? '🚀 TRADE APPROVED' : '❌ TRADE REJECTED'}</h3>
                                        <div style="text-align: right;">
                                            ${badge}
                                            <div style="font-size: 14px; color: #666; margin-top: 5px;">Confidence: ${(p.confidence ?? 0).toFixed(2)}/1.0</div>
                                        </div>
                                    </div>
                                    
                                    <!-- Quick Summary -->
                                    <div style="background: ${approved ? 'linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)' : 'linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)'}; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid ${approved ? '#c3e6cb' : '#f5c6cb'};">
                                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; text-align: center;">
                                            <div>
                                                <div style="font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">Direction</div>
                                                <div style="font-size: 18px; font-weight: bold; color: ${p.direction === 'long' ? '#28a745' : '#dc3545'};">${(p.direction || 'unknown').toUpperCase()}</div>
                                            </div>
                                            <div>
                                                <div style="font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">Entry Price</div>
                                                <div style="font-size: 18px; font-weight: bold; color: ${approved ? '#155724' : '#721c24'};">$${(p.execution?.entry_price || 0).toFixed(2)}</div>
                                            </div>
                                            <div>
                                                <div style="font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">Stop Loss</div>
                                                <div style="font-size: 18px; font-weight: bold; color: ${approved ? '#155724' : '#721c24'};">$${(p.execution?.stop_loss || 0).toFixed(2)}</div>
                                            </div>
                                            <div>
                                                <div style="font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">Risk/Reward</div>
                                                <div style="font-size: 18px; font-weight: bold; color: ${approved ? '#155724' : '#721c24'};">${(p.execution?.risk_reward || 0).toFixed(2)}</div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0;">
                                        <div>
                                            <h4 style="margin: 0 0 10px 0; color: ${approved ? '#155724' : '#721c24'};">Trade Parameters</h4>
                                            <p><strong>Direction:</strong> <span style="color: ${p.direction === 'long' ? '#28a745' : '#dc3545'}; font-weight: bold; text-transform: uppercase;">${p.direction || 'unknown'}</span></p>
                                            <p><strong>Confidence:</strong> ${(p.confidence ?? 0).toFixed(2)}/1.0</p>
                                            <p><strong>Entry Type:</strong> ${p.execution?.entry_type || 'market'}</p>
                                            <p><strong>Entry Price:</strong> $${(p.execution?.entry_price || 0).toFixed(2)}</p>
                                        </div>
                                        <div>
                                            <h4 style="margin: 0 0 10px 0; color: ${approved ? '#155724' : '#721c24'};">Risk Management</h4>
                                            <p><strong>Stop Loss:</strong> $${(p.execution?.stop_loss || 0).toFixed(2)}</p>
                                            <p><strong>Risk/Reward:</strong> ${(p.execution?.risk_reward || 0).toFixed(2)}</p>
                                            <p><strong>Position Size:</strong> ${p.execution?.position_size_note || 'N/A'}</p>
                                        </div>
                                    </div>
                            `;
                            
                            // Add Take Profit levels if available
                            if (p.execution?.take_profits && Array.isArray(p.execution.take_profits) && p.execution.take_profits.length > 0) {
                                gateDisplay += `
                                    <div style="margin: 15px 0;">
                                        <h4 style="margin: 0 0 10px 0; color: ${approved ? '#155724' : '#721c24'};">Take Profit Levels</h4>
                                        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                                `;
                                p.execution.take_profits.forEach((tp, index) => {
                                    gateDisplay += `
                                        <div style="background: #e8f5e8; border: 1px solid #c3e6cb; padding: 8px 12px; border-radius: 4px; text-align: center;">
                                            <div style="font-weight: bold; color: #155724;">TP${index + 1}</div>
                                            <div style="color: #155724;">$${(tp.price || 0).toFixed(2)}</div>
                                            <div style="font-size: 12px; color: #666;">${((tp.portion || 0) * 100).toFixed(0)}%</div>
                                        </div>
                                    `;
                                });
                                gateDisplay += `</div></div>`;
                            }
                            
                            // Add reasons and warnings
                            if (Array.isArray(p.reasons) && p.reasons.length) {
                                gateDisplay += `
                                    <div style="margin: 15px 0;">
                                        <h4 style="margin: 0 0 10px 0; color: ${approved ? '#155724' : '#721c24'};">Reasons for Decision</h4>
                                        <ul style="margin: 0; padding-left: 20px;">
                                `;
                                p.reasons.slice(0, 5).forEach(reason => {
                                    gateDisplay += `<li style="margin: 5px 0; color: ${approved ? '#155724' : '#721c24'};">${reason}</li>`;
                                });
                                gateDisplay += `</ul></div>`;
                            }
                            
                            if (Array.isArray(p.warnings) && p.warnings.length) {
                                gateDisplay += `
                                    <div style="margin: 15px 0; padding: 10px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px;">
                                        <h4 style="margin: 0 0 10px 0; color: #856404;">⚠️ Warnings</h4>
                                        <ul style="margin: 0; padding-left: 20px;">
                                `;
                                p.warnings.slice(0, 3).forEach(warning => {
                                    gateDisplay += `<li style="margin: 5px 0; color: #856404;">${warning}</li>`;
                                });
                                gateDisplay += `</ul></div>`;
                            }
                            
                            // Add detailed execution info in collapsible section
                            if (p.execution) {
                                gateDisplay += `
                                    <details style="margin: 15px 0;">
                                        <summary style="cursor: pointer; font-weight: bold; color: ${approved ? '#155724' : '#721c24'}; padding: 8px; background: #f8f9fa; border-radius: 4px;">View Detailed Execution Info</summary>
                                        <pre style="white-space:pre-wrap;background:#fff;border:1px solid #eee;padding:12px;border-radius:4px;margin-top:8px;font-size:12px;">${JSON.stringify(p.execution, null, 2)}</pre>
                                    </details>
                                `;
                            }
                            
                            gateDisplay += `</div>`;
                            resultDiv.innerHTML = gateDisplay;
                        } else if (maybe && maybe.type === 'proposed_signal' && maybe.payload) {
                            const p = maybe.payload;
                            // Update proposed box
                            proposedSymbol.textContent = p.symbol || '-';
                            proposedTimeframe.textContent = p.timeframe || '-';
                            const dir = (p.direction || 'unknown').toLowerCase();
                            proposedBadge.textContent = dir;
                            if (dir === 'long') {
                                proposedBadge.style.color = '#155724';
                                proposedBadge.style.backgroundColor = '#d4edda';
                                proposedBadge.style.borderColor = '#c3e6cb';
                            } else if (dir === 'short') {
                                proposedBadge.style.color = '#721c24';
                                proposedBadge.style.backgroundColor = '#f8d7da';
                                proposedBadge.style.borderColor = '#f5c6cb';
                            } else {
                                proposedBadge.style.color = '#6c757d';
                                proposedBadge.style.backgroundColor = '#f1f3f5';
                                proposedBadge.style.borderColor = '#ccc';
                            }
                            proposedBox.style.display = 'block';
                        } else if (maybe && maybe.type === 'invalidation_status' && maybe.payload) {
                            // Handle invalidation status updates
                            const payload = maybe.payload;
                            const invalidationAlert = document.getElementById('invalidationAlert');
                            const invalidationMessage = document.getElementById('invalidationMessage');
                            const triggeredConditionsList = document.getElementById('triggeredConditionsList');
                            
                            // Show the invalidation alert
                            invalidationAlert.style.display = 'block';
                            
                            // Update the message
                            invalidationMessage.textContent = payload.message || 'Signal has been invalidated';
                            
                            // Update the triggered conditions list
                            if (payload.triggered_conditions && payload.triggered_conditions.length > 0) {
                                triggeredConditionsList.innerHTML = payload.triggered_conditions.map(condition => 
                                    `<div style="margin: 4px 0; padding: 4px; background-color: #f8d7da; border-radius: 3px; border-left: 3px solid #dc3545;">
                                        <strong>${condition}</strong>
                                    </div>`
                                ).join('');
                            } else {
                                triggeredConditionsList.innerHTML = '<div style="color: #666;">No specific conditions identified</div>';
                            }
                            
                            // Also update the result div to show invalidation status
                            const resultDiv = document.getElementById('result');
                            resultDiv.innerHTML = `
                                <div class="result error">
                                    <h3>⚠️ Signal Invalidated</h3>
                                    <p><strong>Status:</strong> ${payload.status}</p>
                                    <p><strong>Message:</strong> ${payload.message}</p>
                                    ${payload.triggered_conditions && payload.triggered_conditions.length ? 
                                        `<p><strong>Triggered Conditions:</strong> ${payload.triggered_conditions.join(', ')}</p>` : ''}
                                    <p style="color: #666; font-size: 14px; margin-top: 10px;">
                                        The trading signal has been invalidated and will not proceed to the trade gate.
                                    </p>
                                </div>
                            `;
                        } else if (maybe && maybe.type === 'indicator_details' && maybe.payload) {
                            // Handle detailed indicator information
                            const payload = maybe.payload;
                            const indicators = payload.indicators || [];
                            
                            // Show the indicator list
                            indicatorList.style.display = 'block';
                            
                            // Clear and populate indicator items
                            indicatorItems.innerHTML = '';
                            
                            indicators.forEach((indicator, index) => {
                                const itemDiv = document.createElement('div');
                                itemDiv.style.marginBottom = '8px';
                                itemDiv.style.padding = '8px';
                                itemDiv.style.borderRadius = '4px';
                                itemDiv.style.border = '1px solid #ddd';
                                itemDiv.style.backgroundColor = indicator.met ? '#d4edda' : '#f8d7da';
                                
                                const statusIcon = indicator.met ? '✅' : '❌';
                                const statusText = indicator.met ? 'MET' : 'NOT MET';
                                const statusColor = indicator.met ? '#155724' : '#721c24';
                                
                                let valueText = '';
                                if (indicator.current_value !== null && indicator.current_value !== undefined) {
                                    valueText = `<br><small style="color: #666;">Current: ${indicator.current_value}</small>`;
                                }
                                if (indicator.target_value !== null && indicator.target_value !== undefined) {
                                    valueText += `<br><small style="color: #666;">Target: ${indicator.target_value}</small>`;
                                }
                                
                                itemDiv.innerHTML = `
                                    <div style="display: flex; align-items: center; justify-content: space-between;">
                                        <div>
                                            <strong style="color: ${statusColor};">${statusIcon} ${indicator.name}</strong>
                                            <br><small style="color: #666;">${indicator.condition}</small>
                                            ${valueText}
                                        </div>
                                        <span style="color: ${statusColor}; font-weight: bold; font-size: 12px;">${statusText}</span>
                                    </div>
                                `;
                                
                                indicatorItems.appendChild(itemDiv);
                            });
                        }
                    } catch (_) {
                        // not a structured payload; continue normal flow
                    }

                    // Handle indicator checker progress updates
                    if (data.message.includes('Indicator checker progress:')) {
                        const match = data.message.match(/(\\d+)\\/(\\d+)/);
                        if (match) {
                            const met = parseInt(match[1]);
                            const total = parseInt(match[2]);
                            const percentage = (met / total) * 100;
                            
                            indicatorProgress.style.display = 'block';
                            indicatorProgressFill.style.width = percentage + '%';
                            indicatorProgressText.textContent = `${met}/${total} indicators met (${percentage.toFixed(1)}%)`;
                            
                            // Change color based on progress
                            if (percentage === 100) {
                                indicatorProgressFill.style.backgroundColor = '#28a745'; // Green
                                indicatorProgressText.style.color = '#155724';
                            } else if (percentage >= 50) {
                                indicatorProgressFill.style.backgroundColor = '#ffc107'; // Yellow
                                indicatorProgressText.style.color = '#856404';
                            } else {
                                indicatorProgressFill.style.backgroundColor = '#dc3545'; // Red
                                indicatorProgressText.style.color = '#721c24';
                            }
                        }
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
                            invalidationAlert.style.display = 'none';
                            indicatorProgress.style.display = 'none';
                            indicatorList.style.display = 'none';
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
                                // Check if we have meaningful results to preserve
                                const hasGateResults = resultDiv.innerHTML.includes('Gate Decision') || resultDiv.innerHTML.includes('APPROVED') || resultDiv.innerHTML.includes('REJECTED');
                                const hasInvalidationResults = resultDiv.innerHTML.includes('Signal Invalidated');
                                const hasDetailedResults = hasGateResults || hasInvalidationResults;
                                
                                if (!hasDetailedResults) {
                                    // Only show generic completion message if no specific results were displayed
                                    resultDiv.innerHTML = `
                                        <div class="result success">
                                            <h3>Analysis Complete!</h3>
                                            <p><strong>Job ID:</strong> ${data.job_id}</p>
                                            <p>Check the progress messages above for detailed results including gate decision logs.</p>
                                            <p>Results have been saved to the llm_outputs directory.</p>
                                        </div>
                                    `;
                                } else {
                                    // Add completion footer to existing results without overwriting them
                                    const existingContent = resultDiv.innerHTML;
                                    resultDiv.innerHTML = existingContent + `
                                        <div class="result success" style="margin-top: 15px; padding: 15px; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border: 1px solid #c3e6cb; color: #155724; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                                <div>
                                                    <h4 style="margin: 0 0 5px 0; font-size: 18px;">✅ Analysis Complete!</h4>
                                                    <p style="margin: 0; font-size: 14px; opacity: 0.9;"><strong>Job ID:</strong> ${data.job_id}</p>
                                                </div>
                                                <div style="text-align: right; font-size: 12px; opacity: 0.8;">
                                                    <div>Results saved to llm_outputs directory</div>
                                                    <div>Check progress messages for full details</div>
                                                </div>
                                            </div>
                                        </div>
                                    `;
                                }
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
                        invalidationAlert.style.display = 'none';
                        indicatorProgress.style.display = 'none';
                        indicatorList.style.display = 'none';
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
                    invalidationAlert.style.display = 'none';
                    indicatorProgress.style.display = 'none';
                    indicatorList.style.display = 'none';
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
        
        # Clear the progress queue before starting
        while not progress_queue.empty():
            progress_queue.get()

        # Run the trading analysis in a separate thread and clean up the temp file after
        def run_analysis_and_cleanup():
            try:
                run_trading_analysis(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        # Start analysis in background thread
        analysis_thread = threading.Thread(target=run_analysis_and_cleanup)
        analysis_thread.start()

        # Return immediately with a job ID
        return jsonify({
            'success': True,
            'message': 'Analysis started',
            'job_id': 'analysis_' + str(int(time.time()))
        })
    else:
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload an image file.'})

@app.route('/list-json-files')
def list_json_files():
    """List all available JSON files for resuming analysis"""
    try:
        output_dir = "llm_outputs"
        if not os.path.exists(output_dir):
            return jsonify({'success': True, 'files': []})
        
        json_files = []
        for filename in os.listdir(output_dir):
            if filename.endswith('.json') and filename.startswith('llm_output_'):
                file_path = os.path.join(output_dir, filename)
                file_stat = os.stat(file_path)
                
                # Try to read the JSON to extract basic info
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    json_files.append({
                        'filename': filename,
                        'filepath': file_path,
                        'created': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': data.get('symbol', 'Unknown'),
                        'timeframe': data.get('timeframe', 'Unknown'),
                        'direction': data.get('opening_signal', {}).get('direction', 'Unknown'),
                        'is_met': data.get('opening_signal', {}).get('is_met', False),
                        'time_of_screenshot': data.get('time_of_screenshot', 'Unknown')
                    })
                except Exception as e:
                    # If we can't read the JSON, still include the file
                    json_files.append({
                        'filename': filename,
                        'filepath': file_path,
                        'created': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': 'Unknown',
                        'timeframe': 'Unknown',
                        'direction': 'Unknown',
                        'is_met': False,
                        'time_of_screenshot': 'Unknown',
                        'error': str(e)
                    })
        
        # Sort by creation time (newest first)
        json_files.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({'success': True, 'files': json_files})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/resume-analysis', methods=['POST'])
def resume_analysis():
    """Resume analysis from an existing JSON file"""
    try:
        data = request.get_json()
        if not data or 'filepath' not in data:
            return jsonify({'success': False, 'error': 'No filepath provided'})
        
        filepath = data['filepath']
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        # Load the existing LLM output
        with open(filepath, 'r') as f:
            llm_output = json.load(f)
        
        # Clear the progress queue before starting
        while not progress_queue.empty():
            progress_queue.get()
        
        # Run the trading analysis from the loaded data
        def run_resumed_analysis():
            try:
                run_trading_analysis_from_llm_output(llm_output, filepath)
            except Exception as e:
                emit_progress(f"ERROR: Resumed analysis failed: {str(e)}")
        
        # Start analysis in background thread
        analysis_thread = threading.Thread(target=run_resumed_analysis)
        analysis_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Resumed analysis started',
            'job_id': 'resumed_analysis_' + str(int(time.time()))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def run_trading_analysis_from_llm_output(llm_output: dict, original_filepath: str) -> dict:
    """Run trading analysis starting from existing LLM output"""
    try:
        # Step 1: Load existing LLM output
        emit_progress("Step 1: Loading existing analysis...", 1, 12)
        emit_progress(f"Step 1 Complete: Loaded analysis for {llm_output.get('symbol', 'Unknown')} from {os.path.basename(original_filepath)}", 1, 12)

        # Step 2: Extract trading parameters
        emit_progress("Step 2: Extracting trading parameters from loaded analysis...", 2, 12)
        symbol = llm_output['symbol']
        timeframe = llm_output['timeframe']
        time_of_screenshot = llm_output['time_of_screenshot']
        emit_progress(f"Step 2 Complete: Symbol={symbol}, Timeframe={timeframe}, Screenshot time={time_of_screenshot}", 2, 12)

        # Step 3: Validate timeframe
        emit_progress("Step 3: Validating timeframe against MEXC supported intervals...", 3, 12)
        valid_intervals = ['1m', '5m', '15m', '30m', '60m', '4h', '1d', '1W', '1M']
        if timeframe not in valid_intervals:
            emit_progress(f"Warning: {timeframe} is not a valid MEXC interval. Using 1m as default.", 3, 12)
            timeframe = '1m'
        emit_progress(f"Step 3 Complete: Using timeframe {timeframe}", 3, 12)

        # Step 4: Initialize MEXC API clients
        emit_progress("Step 4: Initializing MEXC API clients...", 4, 12)
        try:
            from pymexc import spot
            spot_client = spot.HTTP(api_key=os.getenv("MEXC_API_KEY"), api_secret=os.getenv("MEXC_API_SECRET"))
            public_spot_client = spot.HTTP()
            ws_spot_client = spot.WebSocket(api_key=os.getenv("MEXC_API_KEY"), api_secret=os.getenv("MEXC_API_SECRET"))
            emit_progress("Step 4 Complete: MEXC API clients initialized", 4, 12)
        except Exception as e:
            emit_progress(f"Step 4 Warning: MEXC API initialization failed: {e}", 4, 12)
            emit_progress("Step 4 Complete: Continuing without MEXC API clients", 4, 12)

        # Step 5: Fetch current market data
        emit_progress("Step 5: Fetching current market data...", 5, 12)
        df = fetch_market_dataframe(symbol, timeframe)
        emit_progress(f"Step 5 Complete: Market data fetched, {len(df)} candles retrieved", 5, 12)

        # Step 6: Calculate technical indicators (RSI, MACD, Stoch, BB, ATR)
        emit_progress("Step 6: Calculating technical indicators...", 6, 12)
        df = calculate_rsi14(df)
        df = calculate_macd12_26_9(df)
        df = calculate_stoch14_3_3(df)
        df = calculate_bb20_2(df)
        df = calculate_atr14(df)
        emit_progress("Step 6 Complete: Technical indicators calculation finished", 6, 12)
        
        # Step 7: Initial indicator check
        emit_progress("Step 7: Running initial indicator check...", 7, 12)
        indicator_checker(df, llm_output, emit_progress)
        emit_progress("Step 7 Complete: Initial indicator check finished", 7, 12)

        # Step 8: Extract latest market indicators
        emit_progress("Step 8: Extracting latest market indicators...", 8, 12)
        if not df.empty and 'RSI14' in df.columns:
            latest_rsi = df['RSI14'].iloc[-1]
            emit_progress(f"Step 8 Complete: Latest RSI14 = {latest_rsi}", 8, 12)
        else:
            emit_progress("Step 8 Complete: No RSI14 data available", 8, 12)
        
        if not df.empty and 'MACD_Line' in df.columns:
            latest_macd_line = df['MACD_Line'].iloc[-1]
            latest_macd_signal = df['MACD_Signal'].iloc[-1]
            latest_macd_histogram = df['MACD_Histogram'].iloc[-1]
            emit_progress(f"Step 8 Complete: Latest MACD Line = {latest_macd_line:.4f}, Signal = {latest_macd_signal:.4f}, Histogram = {latest_macd_histogram:.4f}", 8, 12)
        else:
            emit_progress("Step 8 Complete: No MACD data available", 8, 12)

        # Step 9: Start signal validation polling
        emit_progress("Step 9: Starting signal validation polling...", 9, 12)
        signal_valid, signal_status, triggered_conditions, market_values = poll_until_decision(symbol, timeframe, llm_output, emit_progress_fn=emit_progress)
        emit_progress(f"Step 9 Complete: Final Signal Status: {signal_status}", 9, 12)

        # Step 10: Check if signal is valid for trade gate
        emit_progress("Step 10: Checking if signal is valid for trade gate...", 10, 12)
        gate_result = None
        
        if signal_valid and signal_status == "valid":
            emit_progress("Step 10 Complete: Signal is valid, proceeding to trade gate", 10, 12)
            emit_progress("Step 11: Running LLM trade gate decision...", 11, 12)
            
            checklist_passed = True
            invalidation_triggered_recent = len(triggered_conditions) > 0
            emit_progress(f"Step 11: Gate input - Checklist passed: {checklist_passed}, Invalidation triggered: {invalidation_triggered_recent}")
            emit_progress(f"Step 11: Market values - Price: ${market_values.get('current_price', 0):.2f}, RSI: {market_values.get('current_rsi', 0):.2f}")
            
            gate_result = llm_trade_gate_decision(
                base_llm_output=llm_output,
                market_values=market_values,
                checklist_passed=checklist_passed,
                invalidation_triggered=invalidation_triggered_recent,
                triggered_conditions=triggered_conditions,
            )
            emit_progress("Step 11 Complete: LLM trade gate decision finished", 11, 12)
            
            emit_progress("Step 12: Saving gate result...", 12, 12)
            output_dir = "llm_outputs"
            gate_filename = f"llm_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            gate_path = os.path.join(output_dir, gate_filename)
            with open(gate_path, 'w') as f:
                json.dump(gate_result, f, indent=2, default=str)
            emit_progress(f"Step 12 Complete: Gate result saved to {gate_path}", 12, 12)

            # Emit structured gate outcome to UI
            try:
                emit_progress(
                    json.dumps({
                        "type": "gate_outcome",
                        "payload": make_json_serializable({
                            "should_open": bool(gate_result.get("should_open", False)),
                            "direction": gate_result.get("direction", "unknown"),
                            "confidence": float(gate_result.get("confidence", 0.0) or 0.0),
                            "reasons": gate_result.get("reasons", [])[:4],
                            "warnings": gate_result.get("warnings", [])[:4],
                            "execution": gate_result.get("execution", {}),
                            "checks": gate_result.get("checks", {})
                        })
                    })
                )
            except Exception:
                emit_progress(f"Gate: Outcome -> should_open={gate_result.get('should_open')}, direction={gate_result.get('direction')}, confidence={gate_result.get('confidence')}")

            if gate_result.get("should_open") is True:
                confidence = gate_result.get("confidence", 0.0)
                direction = gate_result.get("direction", "unknown")
                entry_price = gate_result.get("execution", {}).get("entry_price", 0)
                stop_loss = gate_result.get("execution", {}).get("stop_loss", 0)
                risk_reward = gate_result.get("execution", {}).get("risk_reward", 0)
                take_profits = gate_result.get("execution", {}).get("take_profits", [])
                
                emit_progress(f"Step 12 Complete: Trade approved for opening - Direction: {direction}, Confidence: {confidence:.2f}", 12, 12)
                emit_progress(f"🚀 TRADE APPROVED: {direction.upper()} at ${entry_price:.2f}, SL: ${stop_loss:.2f}, R/R: {risk_reward:.2f}", 12, 12)
                
                # Place order on MEXC Futures (guarded by env flag MEXC_ENABLE_ORDERS)
                try:
                    if os.getenv("MEXC_ENABLE_ORDERS", "false").lower() in ("1", "true", "yes"): 
                        emit_progress("Step 13: Placing MEXC futures order...", 13, 13)
                        order_outcome = open_trade_with_mexc(llm_output.get("symbol", ""), gate_result)
                        if order_outcome.get("ok"):
                            # Save order response
                            try:
                                output_dir = "llm_outputs"
                                os.makedirs(output_dir, exist_ok=True)
                                order_filename = f"mexc_order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                order_path = os.path.join(output_dir, order_filename)
                                with open(order_path, 'w') as f:
                                    json.dump(order_outcome.get("response"), f, indent=2, default=str)
                                emit_progress(f"Saved order response to {order_path}")
                            except Exception as e:
                                emit_progress(f"⚠️ Failed to save order response: {str(e)}")
                        else:
                            emit_progress(f"Step 13 Failed: Order placement error - {order_outcome.get('error')}")
                    else:
                        emit_progress("Step 13: Live order placement disabled (MEXC_ENABLE_ORDERS is false)")
                except Exception as e:
                    emit_progress(f"⚠️ Order placement exception: {str(e)}")

                # Send iPhone notification for valid trade
                try:
                    trade_data = {
                        "symbol": llm_output.get("symbol", "Unknown"),
                        "direction": direction,
                        "current_price": entry_price,
                        "confidence": confidence,
                        "current_rsi": market_values.get("current_rsi", 0),
                        "stop_loss": stop_loss,
                        "risk_reward": risk_reward,
                        "take_profits": take_profits
                    }
                    notification_results = notify_valid_trade(trade_data)
                    emit_progress(f"📱 Notifications sent - Pushover: {'✅' if notification_results.get('pushover') else '❌'}, Email: {'✅' if notification_results.get('email') else '❌'}", 12, 12)
                except Exception as e:
                    emit_progress(f"⚠️ Notification failed: {str(e)}", 12, 12)
                
                if take_profits:
                    tp_info = ", ".join([f"TP{i+1}: ${tp.get('price', 0):.2f}" for i, tp in enumerate(take_profits)])
                    emit_progress(f"Take Profits: {tp_info}", 12, 12)
            else:
                reasons = gate_result.get("reasons", ["No specific reason provided"])
                direction = gate_result.get("direction", "unknown")
                confidence = gate_result.get("confidence", 0.0)
                emit_progress(f"Step 12 Complete: Trade not approved by gate - Direction: {direction}, Confidence: {confidence:.2f}", 12, 12)
                emit_progress(f"❌ TRADE REJECTED: {direction.upper()} - Reasons: {'; '.join(reasons[:2])}", 12, 12)
        else:
            emit_progress(f"Step 10 Complete: Signal not valid (status: {signal_status}), skipping trade gate", 10, 12)

        return {
            "success": True,
            "llm_output": llm_output,
            "signal_status": signal_status,
            "signal_valid": signal_valid,
            "market_values": market_values,
            "gate_result": gate_result,
            "original_file": original_filepath
        }

    except Exception as e:
        emit_progress(f"ERROR: Resumed analysis failed with {type(e).__name__}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

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
