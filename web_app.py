import os
import json
import base64
import tempfile
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import litellm
from prompt import OPENAI_VISION_PROMPT, TRADE_GATE_PROMPT
import pandas as pd
from datetime import datetime
import time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import subprocess
import sys
import threading
import queue
import time

# Import notification service
from services.notification_service import (
    notify_valid_trade, 
    notify_invalidated_trade,
    notify_rejected_trade,
    send_image_to_telegram,
    send_initial_analysis_to_telegram,
    send_polling_start_to_telegram,
    send_analysis_to_telegram
)

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


# Configure LiteLLM models (can be changed via environment variable)
# Vision model for chart analysis (must support vision)
LITELLM_VISION_MODEL = os.getenv("LITELLM_VISION_MODEL", os.getenv("LITELLM_MODEL", "gpt-4o"))
# Text model for trade gate decisions (can be any model, including non-vision models like DeepSeek)
LITELLM_TEXT_MODEL = os.getenv("LITELLM_TEXT_MODEL", os.getenv("LITELLM_MODEL", "gpt-4o"))

# Multi-model comparison configuration
# Models to test for trading decisions (can be configured via environment variables)
MULTI_MODEL_CHATGPT = os.getenv("MULTI_MODEL_CHATGPT", "gpt-4o")  # ChatGPT5.1 - using gpt-4o as default
MULTI_MODEL_DEEPSEEK = os.getenv("MULTI_MODEL_DEEPSEEK", "deepseek/deepseek-chat")  # deepseek-chat
MULTI_MODEL_GEMINI = os.getenv("MULTI_MODEL_GEMINI", "gemini/gemini-1.5-pro")  # Google Gemini

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Global progress queue for real-time updates
progress_queue = queue.Queue()

# Track Telegram bot initialization to avoid duplicate starts (e.g., Flask reloader)
_telegram_bot_started = False
_telegram_bot_lock = threading.Lock()


def start_telegram_bot_once():
    """Start the Telegram bot only once per process."""
    global _telegram_bot_started

    with _telegram_bot_lock:
        if _telegram_bot_started:
            return

        try:
            start_bot()
            print("✅ Telegram bot started for command control")
        except Exception as e:
            print(f"⚠️  Could not start Telegram bot: {e}")
        finally:
            # Mark as attempted to prevent repeated retries across Flask hooks
            _telegram_bot_started = True


def ensure_telegram_bot_started():
    """Ensure the Telegram bot is running before the app starts serving requests."""
    start_telegram_bot_once()

TRUTHY_VALUES = {"1", "true", "yes", "on"}

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
    # progress_queue.put(progress_data) # Disabled for API-only mode
    print(message)  # Also print to console

def open_trade_with_mexc(symbol: str, gate_result: Dict[str, Any]) -> Dict[str, Any]:
    """Place a futures order on MEXC based on the gate result.
    
    Environment variables:
    - MEXC_API_KEY: Your MEXC API key
    - MEXC_API_SECRET: Your MEXC API secret
    - MEXC_DEFAULT_SIZE: Order size in USDT (default: 10.0)
    - MEXC_LEVERAGE: Leverage multiplier (default: 30, max: 200)
    - MEXC_ENABLE_ORDERS: Set to 'true' to enable live trading
    """
    try:
        from pymexc import futures
        
        # Get credentials and settings from environment
        api_key = os.getenv("MEXC_API_KEY")
        api_secret = os.getenv("MEXC_API_SECRET")
        
        if not api_key or not api_secret:
            return {"ok": False, "error": "MEXC_API_KEY or MEXC_API_SECRET not set in environment"}
        
        # Get trading parameters
        try:
            default_size = float(os.getenv("MEXC_DEFAULT_SIZE", "10.0"))
        except Exception:
            default_size = 10.0
        
        try:
            leverage = int(os.getenv("MEXC_LEVERAGE", "30"))
        except Exception:
            leverage = 30
        
        # Initialize MEXC Futures client
        futures_client = futures.HTTP(api_key=api_key, api_secret=api_secret)
        
        # Extract trading parameters from gate result
        direction = (gate_result.get("direction") or "").lower()
        execution = gate_result.get("execution", {})
        entry_price = execution.get("entry_price")
        stop_loss = execution.get("stop_loss")
        take_profits = execution.get("take_profits") or []
        
        # Convert symbol to MEXC format (e.g., BTCUSDT, ETHUSDT)
        mexc_symbol = symbol.replace(".P", "").replace("USDT", "").upper() + "_USDT"
        
        # Determine order side (1 = buy/long, 2 = sell/short, 3 = close long, 4 = close short)
        side = 1 if direction == "long" else 2
        
        print(f"📊 MEXC Futures Order Parameters:")
        print(f"   Symbol: {mexc_symbol}")
        print(f"   Side: {'LONG' if side == 1 else 'SHORT'}")
        print(f"   Leverage: {leverage}x")
        print(f"   Size: ${default_size} USDT")
        if entry_price:
            print(f"   Entry Price: ${entry_price:.2f}")
        if stop_loss:
            print(f"   Stop Loss: ${stop_loss:.2f}")
        
        # Set leverage for the symbol
        try:
            leverage_result = futures_client.change_leverage(
                symbol=mexc_symbol,
                leverage=leverage,
                openType=1  # 1 = isolated, 2 = cross
            )
            print(f"   ✅ Leverage set to {leverage}x: {leverage_result}")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not set leverage: {e}")
            # Continue anyway - leverage might already be set
        
        # Place the main order
        # Order type: 1 = limit, 2 = market, 3 = post-only, 4 = IOC, 5 = FOK
        order_params = {
            "symbol": mexc_symbol,
            "side": side,
            "type": 2,  # Market order
            "vol": default_size / (entry_price if entry_price else 1),  # Calculate volume from USDT amount
        }
        
        print(f"\n🚀 Placing MEXC Futures Order...")
        order_result = futures_client.create_order(**order_params)
        
        print(f"✅ Order placed successfully!")
        print(f"   Order ID: {order_result.get('orderId', 'N/A')}")
        print(f"   Status: {order_result.get('status', 'N/A')}")
        
        # Set stop loss if provided
        if stop_loss and order_result.get('orderId'):
            try:
                # Stop loss order (trigger order)
                sl_side = 4 if direction == "long" else 3  # 3 = close long, 4 = close short
                sl_params = {
                    "symbol": mexc_symbol,
                    "side": sl_side,
                    "type": 3,  # Trigger market order
                    "vol": order_params["vol"],
                    "triggerPrice": stop_loss,
                    "triggerType": 1,  # 1 = mark price, 2 = last price
                    "executeCycle": 1  # 1 = always valid
                }
                sl_result = futures_client.create_order(**sl_params)
                print(f"✅ Stop Loss set at ${stop_loss:.2f}")
                print(f"   SL Order ID: {sl_result.get('orderId', 'N/A')}")
            except Exception as e:
                print(f"⚠️  Warning: Could not set stop loss: {e}")
        
        # Set take profit orders if provided
        if take_profits and order_result.get('orderId'):
            for i, tp in enumerate(take_profits, 1):
                try:
                    tp_price = tp.get('price')
                    tp_portion = tp.get('portion', 1.0)
                    if tp_price:
                        tp_side = 4 if direction == "long" else 3
                        tp_vol = order_params["vol"] * tp_portion
                        tp_params = {
                            "symbol": mexc_symbol,
                            "side": tp_side,
                            "type": 3,  # Trigger market order
                            "vol": tp_vol,
                            "triggerPrice": tp_price,
                            "triggerType": 1,
                            "executeCycle": 1
                        }
                        tp_result = futures_client.create_order(**tp_params)
                        print(f"✅ Take Profit {i} set at ${tp_price:.2f} ({tp_portion*100:.0f}% of position)")
                except Exception as e:
                    print(f"⚠️  Warning: Could not set TP{i}: {e}")
        
        return {
            "ok": True,
            "response": order_result,
            "symbol": mexc_symbol,
            "leverage": leverage,
            "size": default_size
        }
        
    except Exception as e:
        error_msg = f"MEXC futures order failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {"ok": False, "error": error_msg}

def _convert_symbol_to_hyperliquid_coin(symbol: str) -> str:
    """Convert symbol notation to Hyperliquid coin name (e.g., BTCUSDT.P -> BTC, BTCUSDC -> BTC)."""
    if not symbol:
        return ""

    cleaned = symbol.upper().strip()
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace(".P", "")
    cleaned = cleaned.replace("-", "")

    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0]

    # Remove quote currency suffixes (order matters - check longer suffixes first)
    if cleaned.endswith("_USDC"):
        cleaned = cleaned[:-5]
    elif cleaned.endswith("USDC"):
        cleaned = cleaned[:-4]
    elif cleaned.endswith("_USDT"):
        cleaned = cleaned[:-5]
    elif cleaned.endswith("USDT"):
        cleaned = cleaned[:-4]
    elif cleaned.endswith("_USD"):
        cleaned = cleaned[:-4]
    elif cleaned.endswith("USD"):
        cleaned = cleaned[:-3]

    # Remove perpetual suffixes
    if cleaned.endswith("_PERP"):
        cleaned = cleaned[:-5]
    elif cleaned.endswith("PERP"):
        cleaned = cleaned[:-4]

    cleaned = cleaned.replace("_", "")
    cleaned = cleaned.strip()

    return cleaned or symbol.upper().strip()

def _parse_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _get_hyperliquid_base_url() -> Optional[str]:
    """Get Hyperliquid base URL based on testnet setting or explicit base URL."""
    # Check if explicit base URL is set (takes precedence)
    explicit_base_url = os.getenv("HYPERLIQUID_BASE_URL")
    if explicit_base_url:
        return explicit_base_url
    
    # Check if testnet is enabled
    testnet_enabled = os.getenv("HYPERLIQUID_TESTNET", "false").lower() in ("true", "1", "yes")
    if testnet_enabled:
        return "https://api.hyperliquid-testnet.xyz"
    
    # Default to mainnet (None means mainnet for Hyperliquid SDK)
    return None

def _init_hyperliquid_info():
    """Initialize Hyperliquid Info client for public market data."""
    try:
        from hyperliquid.info import Info
    except ImportError as exc:
        return None, f"Hyperliquid SDK not installed: {exc}"
    
    base_url = _get_hyperliquid_base_url()
    skip_ws = os.getenv("HYPERLIQUID_SKIP_WS", "true").lower() in ("true", "1", "yes")
    
    try:
        info = Info(base_url=base_url, skip_ws=skip_ws)
        return info, None
    except Exception as exc:
        return None, f"Failed to initialize Hyperliquid Info client: {exc}"

def _init_hyperliquid_exchange():
    try:
        from eth_account import Account
        from hyperliquid.exchange import Exchange
    except ImportError as exc:
        return None, f"Hyperliquid SDK not installed: {exc}"

    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    if not private_key:
        return None, "HYPERLIQUID_PRIVATE_KEY not set in environment"

    try:
        wallet = Account.from_key(private_key)
    except Exception as exc:
        return None, f"Invalid Hyperliquid private key: {exc}"

    base_url = _get_hyperliquid_base_url()
    account_address = os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS") or None
    vault_address = os.getenv("HYPERLIQUID_VAULT_ADDRESS") or None

    timeout = None
    timeout_env = os.getenv("HYPERLIQUID_TIMEOUT")
    if timeout_env:
        try:
            timeout = float(timeout_env)
        except Exception:
            timeout = None

    try:
        exchange = Exchange(
            wallet,
            base_url=base_url,
            account_address=account_address,
            vault_address=vault_address,
            timeout=timeout,
        )
        return exchange, None
    except Exception as exc:
        return None, f"Failed to initialize Hyperliquid exchange: {exc}"

def open_trade_with_hyperliquid(symbol: str, gate_result: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an opening order on Hyperliquid based on gate result."""
    exchange, error = _init_hyperliquid_exchange()
    if error:
        return {"ok": False, "error": error}

    coin = _convert_symbol_to_hyperliquid_coin(symbol)
    if not coin:
        return {"ok": False, "error": f"Unable to convert symbol '{symbol}' to Hyperliquid coin"}

    direction = (gate_result.get("direction") or "").lower()
    if direction not in ("long", "short"):
        return {"ok": False, "error": f"Unknown trade direction: {direction}"}

    execution = gate_result.get("execution", {}) or {}
    entry_type = (execution.get("entry_type") or "market").lower()
    entry_price = execution.get("entry_price")

    size = _parse_float_env("HYPERLIQUID_DEFAULT_SIZE", 0.001)
    if size <= 0:
        size = 0.001

    slippage = _parse_float_env("HYPERLIQUID_SLIPPAGE", 0.01)
    tif = os.getenv("HYPERLIQUID_LIMIT_TIF", "Gtc")

    is_buy = direction == "long"

    try:
        if entry_type == "limit" and entry_price:
            limit_px = float(entry_price)
            order_type = {"limit": {"tif": tif}}
            response = exchange.order(
                name=coin,
                is_buy=is_buy,
                sz=size,
                limit_px=limit_px,
                order_type=order_type,
                reduce_only=False,
            )
        else:
            response = exchange.market_open(
                name=coin,
                is_buy=is_buy,
                sz=size,
                px=float(entry_price) if entry_price else None,
                slippage=slippage,
            )
        return {"ok": True, "response": response, "coin": coin, "size": size}
    except Exception as exc:
        return {"ok": False, "error": f"Hyperliquid order failed: {exc}"}

def close_trade_with_hyperliquid(symbol: str, current_price: Optional[float] = None) -> Dict[str, Any]:
    """Close an existing position on Hyperliquid for the given symbol using a limit order."""
    exchange, error = _init_hyperliquid_exchange()
    if error:
        return {"ok": False, "error": error}

    coin = _convert_symbol_to_hyperliquid_coin(symbol)
    if not coin:
        return {"ok": False, "error": f"Unable to convert symbol '{symbol}' to Hyperliquid coin"}

    slippage = max(_parse_float_env("HYPERLIQUID_CLOSE_SLIPPAGE", 0.002), 0.0)
    tif = os.getenv("HYPERLIQUID_CLOSE_TIF", "Gtc")

    try:
        address = exchange.wallet.address
        if exchange.account_address:
            address = exchange.account_address
        if exchange.vault_address:
            address = exchange.vault_address

        user_state = exchange.info.user_state(address)
        positions = user_state.get("assetPositions", []) if isinstance(user_state, dict) else []

        for position_data in positions:
            position = position_data.get("position", {}) if isinstance(position_data, dict) else {}
            if position.get("coin") != coin:
                continue

            try:
                szi = float(position.get("szi", 0))
            except (TypeError, ValueError):
                szi = 0.0

            if abs(szi) < 1e-12:
                continue

            size = abs(szi)
            is_buy = szi < 0  # short -> buy to close, long -> sell to close

            reference_price = None
            if current_price is not None:
                try:
                    reference_price = float(current_price)
                except (TypeError, ValueError):
                    reference_price = None

            if reference_price is None or reference_price <= 0:
                try:
                    mids = exchange.info.all_mids()
                    if isinstance(mids, dict) and mids.get(coin) is not None:
                        reference_price = float(mids[coin])
                except Exception:
                    reference_price = None

            if reference_price is None or reference_price <= 0:
                return {"ok": False, "error": "Unable to determine reference price for Hyperliquid limit close"}

            if is_buy:
                limit_px = reference_price * (1 + slippage)
            else:
                limit_px = reference_price * (1 - slippage)

            if limit_px <= 0:
                limit_px = reference_price

            try:
                response = exchange.order(
                    name=coin,
                    is_buy=is_buy,
                    sz=size,
                    limit_px=limit_px,
                    order_type={"limit": {"tif": tif}},
                    reduce_only=True,
                )
            except Exception as exc:
                return {"ok": False, "error": f"Hyperliquid close failed: {exc}"}

            return {"ok": True, "response": response, "coin": coin, "size": size, "limit_px": limit_px}

        return {"ok": True, "response": None, "message": "No active Hyperliquid position found"}
    except Exception as exc:
        return {"ok": False, "error": f"Hyperliquid close failed: {exc}"}

def analyze_trading_chart(image_path: str, symbol: str = None, timeframe: str = None, df: pd.DataFrame = None) -> dict:
    # Note: df can be a DataFrame or a dict with 'minute', 'hourly', 'daily' keys
    """Analyze trading chart using LLM Vision API with optional market data"""
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
    
    # Prepare the prompt based on whether we have market data
    # df can be a dict with 'minute', 'hourly', 'daily' keys or a single DataFrame
    if df is not None:
        if isinstance(df, dict):
            df_minute = df.get('minute')
            df_hourly = df.get('hourly')
            df_daily = df.get('daily')
        else:
            df_minute = df
            df_hourly = None
            df_daily = None
        
        if df_minute is not None and not df_minute.empty:
            emit_progress("Chart: Preparing enhanced prompt with market data...")
            # Create a summary of the market data for the LLM
            market_data_summary = create_market_data_summary(
                df=df_minute, 
                df_hourly=df_hourly, 
                df_daily=df_daily, 
                symbol=symbol, 
                timeframe=timeframe
            )
            enhanced_prompt = OPENAI_VISION_PROMPT + f"\n\n## MARKET DATA CONTEXT\n{symbol} {timeframe} - Latest market data:\n{market_data_summary}\n\nUse this market data to enhance your analysis and provide more accurate technical indicator values. Pay special attention to the hourly and daily timeframes for trend context."
        else:
            emit_progress("Chart: Using standard prompt without market data...")
            enhanced_prompt = OPENAI_VISION_PROMPT
    else:
        emit_progress("Chart: Using standard prompt without market data...")
        enhanced_prompt = OPENAI_VISION_PROMPT
    
    emit_progress(f"Chart: Sending request to LLM Vision API (model: {LITELLM_VISION_MODEL})...")
    try:
        response = litellm.completion(
            model=LITELLM_VISION_MODEL,
            messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": [{"type": "text", "text": "Analyze this trading chart."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
            ],
            response_format={"type": "json_object"},
            timeout=60  # Add 60 second timeout
        )
        emit_progress("Chart: Received response from LLM Vision API")
        emit_progress(f"Chart: Response has {len(response.choices)} choices")
        if response.choices:
            emit_progress(f"Chart: First choice finish_reason: {response.choices[0].finish_reason}")
    except Exception as e:
        emit_progress(f"Chart: Error calling LLM API: {str(e)}")
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
            retry_response = litellm.completion(
                model=LITELLM_VISION_MODEL,
                messages=[
                    {"role": "system", "content": enhanced_prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON. No additional text or formatting."},
                    {"role": "user", "content": [{"type": "text", "text": "Analyze this trading chart."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
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

def analyze_trading_chart_with_model(image_path: str, model_name: str, symbol: str = None, timeframe: str = None, df: pd.DataFrame = None) -> dict:
    # Note: df can be a DataFrame or a dict with 'minute', 'hourly', 'daily' keys
    """Analyze trading chart using a specific LLM Vision API model"""
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_data = base64.b64encode(image_bytes).decode("utf-8")
        
        if not image_data:
            raise ValueError("Image file appears to be empty")
        
        # Prepare the prompt based on whether we have market data
        # df can be a dict with 'minute', 'hourly', 'daily' keys or a single DataFrame
        if df is not None:
            if isinstance(df, dict):
                df_minute = df.get('minute')
                df_hourly = df.get('hourly')
                df_daily = df.get('daily')
            else:
                df_minute = df
                df_hourly = None
                df_daily = None
            
            if df_minute is not None and not df_minute.empty:
                market_data_summary = create_market_data_summary(
                    df=df_minute, 
                    df_hourly=df_hourly, 
                    df_daily=df_daily, 
                    symbol=symbol, 
                    timeframe=timeframe
                )
                enhanced_prompt = OPENAI_VISION_PROMPT + f"\n\n## MARKET DATA CONTEXT\n{symbol} {timeframe} - Latest market data:\n{market_data_summary}\n\nUse this market data to enhance your analysis and provide more accurate technical indicator values. Pay special attention to the hourly and daily timeframes for trend context."
            else:
                enhanced_prompt = OPENAI_VISION_PROMPT
        else:
            enhanced_prompt = OPENAI_VISION_PROMPT
        
        response = litellm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": [{"type": "text", "text": "Analyze this trading chart."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
            ],
            response_format={"type": "json_object"},
            timeout=120  # Longer timeout for multi-model analysis
        )
        
        if not response.choices or not response.choices[0].message.content:
            # Try retry without response_format
            response = litellm.completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": enhanced_prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON. No additional text or formatting."},
                    {"role": "user", "content": [{"type": "text", "text": "Analyze this trading chart."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
                ],
                timeout=120
            )
        
        content = response.choices[0].message.content if response.choices and response.choices[0].message.content else None
        if not content:
            raise ValueError(f"{model_name} returned empty content")
        
        # Clean markdown formatting if present
        if content.strip().startswith('```json'):
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
        
        result = json.loads(content)
        return result
    except Exception as e:
        # Return error information instead of raising
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "model": model_name,
            "symbol": symbol or "UNKNOWN",
            "timeframe": timeframe or "1m"
        }

def analyze_trading_chart_multi_model(image_path: str, symbol: str = None, timeframe: str = None, df: pd.DataFrame = None) -> dict:
    # Note: df can be a DataFrame or a dict with 'minute', 'hourly', 'daily' keys
    """Analyze trading chart using multiple models in parallel and compare results"""
    emit_progress("🔬 Starting multi-model analysis...")
    emit_progress(f"   Testing models: {MULTI_MODEL_CHATGPT}, {MULTI_MODEL_DEEPSEEK}, {MULTI_MODEL_GEMINI}")
    
    # Define models to test
    models_to_test = {
        "ChatGPT5.1": MULTI_MODEL_CHATGPT,
        "DeepSeek": MULTI_MODEL_DEEPSEEK,
        "Gemini": MULTI_MODEL_GEMINI
    }
    
    # Store results from each model
    results = {}
    errors = {}
    
    # Thread-safe storage
    results_lock = threading.Lock()
    
    def analyze_with_model(model_display_name: str, model_name: str):
        """Thread function to analyze with a specific model"""
        try:
            # Check for required API keys before attempting analysis
            api_key_hint = None
            if "deepseek" in model_name.lower():
                if not os.getenv("DEEPSEEK_API_KEY"):
                    api_key_hint = "DEEPSEEK_API_KEY not found in environment"
                else:
                    # DeepSeek doesn't support vision - warn but still try
                    emit_progress(f"   ⚠️  Warning: DeepSeek does not support vision/image analysis. This may fail.")
            elif "gpt" in model_name.lower() or "openai" in model_name.lower():
                if not os.getenv("OPENAI_API_KEY"):
                    api_key_hint = "OPENAI_API_KEY not found in environment"
            elif "gemini" in model_name.lower():
                if not os.getenv("GEMINI_API_KEY"):
                    api_key_hint = "GEMINI_API_KEY not found in environment"
            
            if api_key_hint:
                with results_lock:
                    error_msg = f"API key missing: {api_key_hint}. Add the required API key to your .env file."
                    errors[model_display_name] = {
                        "error": error_msg,
                        "error_type": "MissingAPIKey",
                        "model": model_name,
                        "hint": api_key_hint
                    }
                    emit_progress(f"   ❌ {model_display_name} skipped: {error_msg}")
                return
            
            emit_progress(f"   🚀 Starting analysis with {model_display_name} ({model_name})...")
            start_time = time.time()
            result = analyze_trading_chart_with_model(image_path, model_name, symbol, timeframe, df)
            elapsed_time = time.time() - start_time
            
            with results_lock:
                if "error" in result:
                    error_msg = result.get('error', 'Unknown error')
                    # Provide more helpful error messages
                    if "provider not provided" in error_msg.lower() or "not provi" in error_msg.lower():
                        error_msg = f"API key missing. Check your .env file for the required API key for {model_name}."
                    elif "deepseek" in error_msg.lower() or "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                        error_msg = f"Authentication failed. Check your DEEPSEEK_API_KEY in .env file."
                    
                    errors[model_display_name] = {
                        "error": error_msg,
                        "error_type": "AnalysisError",
                        "model": model_name
                    }
                    emit_progress(f"   ❌ {model_display_name} failed: {error_msg}")
                else:
                    results[model_display_name] = {
                        "model_name": model_name,
                        "result": result,
                        "elapsed_time": elapsed_time,
                        "timestamp": datetime.now().isoformat()
                    }
                    direction = result.get('opening_signal', {}).get('direction', 'unknown')
                    confidence = result.get('validity_assessment', {}).get('core_alignment_score', 0)
                    emit_progress(f"   ✅ {model_display_name} completed in {elapsed_time:.2f}s - Direction: {direction}, Confidence: {confidence}")
        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__
            
            # Provide more helpful error messages based on exception type
            if "BadRequestError" in error_type or "provider not provided" in error_str.lower():
                if "deepseek" in model_name.lower():
                    error_msg = f"DeepSeek API key missing. Add DEEPSEEK_API_KEY to your .env file."
                elif "gemini" in model_name.lower():
                    error_msg = f"Gemini API key missing. Add GEMINI_API_KEY to your .env file."
                else:
                    error_msg = f"API key missing for {model_name}. Check your .env file."
            elif "DeepseekException" in error_type or "authentication" in error_str.lower():
                if "gemini" in model_name.lower():
                    error_msg = f"Gemini authentication failed. Check your GEMINI_API_KEY in .env file."
                else:
                    error_msg = f"DeepSeek authentication failed. Check your DEEPSEEK_API_KEY in .env file."
            elif "vision" in error_str.lower() or "image" in error_str.lower():
                error_msg = f"{model_display_name} does not support vision/image analysis. This model cannot analyze charts."
            else:
                error_msg = error_str
                # Truncate very long error messages
                if len(error_msg) > 150:
                    error_msg = error_msg[:147] + "..."
            
            with results_lock:
                errors[model_display_name] = {
                    "error": error_msg,
                    "error_type": error_type,
                    "model": model_name
                }
                emit_progress(f"   ❌ {model_display_name} exception: {error_msg}")
    
    # Start all analyses in parallel
    threads = []
    for model_display_name, model_name in models_to_test.items():
        thread = threading.Thread(target=analyze_with_model, args=(model_display_name, model_name))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    emit_progress(f"   📊 Multi-model analysis complete: {len(results)} successful, {len(errors)} failed")
    
    # Create comparison summary
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "image_path": image_path,
        "symbol": symbol,
        "timeframe": timeframe,
        "models_tested": list(models_to_test.keys()),
        "results": results,
        "errors": errors,
        "summary": {}
    }
    
    # Generate summary comparison
    if results:
        directions = {}
        confidences = {}
        times = {}
        
        for model_name, data in results.items():
            result = data["result"]
            direction = result.get('opening_signal', {}).get('direction', 'unknown')
            confidence = result.get('validity_assessment', {}).get('core_alignment_score', 0)
            
            directions[model_name] = direction
            confidences[model_name] = confidence
            times[model_name] = data["elapsed_time"]
        
        # Calculate consensus direction (most common direction)
        consensus_direction = None
        if directions:
            direction_counts = {}
            for direction in directions.values():
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
            consensus_direction = max(direction_counts.items(), key=lambda x: x[1])[0] if direction_counts else None
        
        comparison["summary"] = {
            "directions": directions,
            "confidences": confidences,
            "response_times": times,
            "agreement": len(set(directions.values())) == 1 if directions else False,
            "consensus_direction": consensus_direction,
            "fastest_model": min(times.items(), key=lambda x: x[1])[0] if times else None,
            "note": "Confidence scores are shown for reference only. No automatic selection based on confidence."
        }
        
        emit_progress(f"   📈 Summary: Consensus direction = {comparison['summary'].get('consensus_direction', 'N/A')}")
        emit_progress(f"   📊 All models' confidences: {confidences}")
        emit_progress(f"   ⚡ Fastest model: {comparison['summary'].get('fastest_model', 'N/A')}")
        emit_progress(f"   ℹ️  Note: Review all results manually to determine best model")
    
    # Save comparison results
    output_dir = "llm_outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_filename = f"multi_model_comparison_{timestamp}.json"
    comparison_path = os.path.join(output_dir, comparison_filename)
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    emit_progress(f"   💾 Comparison saved to {comparison_path}")
    
    # Store the path in the comparison dict for easy access
    comparison["comparison_file_path"] = comparison_path
    
    return comparison

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

def create_market_data_summary(df: pd.DataFrame = None, df_hourly: pd.DataFrame = None, df_daily: pd.DataFrame = None, symbol: str = None, timeframe: str = None) -> str:
    """Create a summary of market data for the LLM, including multiple timeframes for trend context"""
    summary_parts = []
    
    # Primary timeframe (minute data)
    if df is not None and not df.empty:
        latest = df.iloc[-1]
        latest_time = latest['Open_time'] if 'Open_time' in df.columns else "Unknown"
        
        summary_parts.append(f"""
=== PRIMARY TIMEFRAME: {timeframe or '1m'} ===
Symbol: {symbol or 'N/A'}
Latest Candle Time: {latest_time}
Latest Price Data:
- Open: {latest.get('Open', 'N/A')}
- High: {latest.get('High', 'N/A')}
- Low: {latest.get('Low', 'N/A')}
- Close: {latest.get('Close', 'N/A')}
- Volume: {latest.get('Volume', 'N/A')}
""")
        
        # Add technical indicators if available
        if 'RSI14' in df.columns and not pd.isna(latest['RSI14']):
            summary_parts.append(f"- RSI14: {latest['RSI14']:.2f}\n")
        
        if 'MACD_Line' in df.columns and not pd.isna(latest['MACD_Line']):
            summary_parts.append(f"- MACD Line: {latest['MACD_Line']:.4f}\n")
            summary_parts.append(f"- MACD Signal: {latest['MACD_Signal']:.4f}\n")
            summary_parts.append(f"- MACD Histogram: {latest['MACD_Histogram']:.4f}\n")

        if 'ATR14' in df.columns and not pd.isna(latest['ATR14']):
            summary_parts.append(f"- ATR14: {latest['ATR14']:.4f}\n")
        
        # Add recent price action context (last 5 candles)
        if len(df) >= 5:
            recent_5 = df.tail(5)
            summary_parts.append(f"\nRecent 5 Candles ({timeframe or '1m'}):\n")
            for i, (_, candle) in enumerate(recent_5.iterrows()):
                candle_time = candle.get('Open_time', 'Unknown')
                close_price = candle.get('Close', 'N/A')
                summary_parts.append(f"- Candle {i+1}: {candle_time} - Close: {close_price}\n")
    
    # Hourly timeframe for trend context
    if df_hourly is not None and not df_hourly.empty:
        latest_h = df_hourly.iloc[-1]
        latest_time_h = latest_h['Open_time'] if 'Open_time' in df_hourly.columns else "Unknown"
        
        summary_parts.append(f"""
=== HOURLY TIMEFRAME (Trend Context) ===
Latest Hourly Candle Time: {latest_time_h}
Latest Hourly Price:
- Open: {latest_h.get('Open', 'N/A')}
- High: {latest_h.get('High', 'N/A')}
- Low: {latest_h.get('Low', 'N/A')}
- Close: {latest_h.get('Close', 'N/A')}
- Volume: {latest_h.get('Volume', 'N/A')}
""")
        
        # Add recent hourly candles for trend context (last 10)
        if len(df_hourly) >= 10:
            recent_10h = df_hourly.tail(10)
            summary_parts.append(f"\nRecent 10 Hourly Candles (Trend Context):\n")
            for i, (_, candle) in enumerate(recent_10h.iterrows()):
                candle_time = candle.get('Open_time', 'Unknown')
                close_price = candle.get('Close', 'N/A')
                summary_parts.append(f"- Hour {i+1}: {candle_time} - Close: {close_price}\n")
    
    # Daily timeframe for longer-term trend context
    if df_daily is not None and not df_daily.empty:
        latest_d = df_daily.iloc[-1]
        latest_time_d = latest_d['Open_time'] if 'Open_time' in df_daily.columns else "Unknown"
        
        summary_parts.append(f"""
=== DAILY TIMEFRAME (Long-term Trend Context) ===
Latest Daily Candle Time: {latest_time_d}
Latest Daily Price:
- Open: {latest_d.get('Open', 'N/A')}
- High: {latest_d.get('High', 'N/A')}
- Low: {latest_d.get('Low', 'N/A')}
- Close: {latest_d.get('Close', 'N/A')}
- Volume: {latest_d.get('Volume', 'N/A')}
""")
        
        # Add recent daily candles for trend context (last 10)
        if len(df_daily) >= 10:
            recent_10d = df_daily.tail(10)
            summary_parts.append(f"\nRecent 10 Daily Candles (Long-term Trend Context):\n")
            for i, (_, candle) in enumerate(recent_10d.iterrows()):
                candle_time = candle.get('Open_time', 'Unknown')
                close_price = candle.get('Close', 'N/A')
                summary_parts.append(f"- Day {i+1}: {candle_time} - Close: {close_price}\n")
    
    if not summary_parts:
        return "No market data available"
    
    return "".join(summary_parts)

def llm_trade_gate_decision(
    base_llm_output: Dict[str, Any],
    market_values: Dict[str, Any],
    checklist_passed: bool,
    invalidation_triggered: bool,
    triggered_conditions: list
) -> Dict[str, Any]:
    """Make trade gate decision using LLM"""
    emit_progress("Gate: Initializing LLM trade gate decision...")
    
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

    emit_progress(f"Gate: Sending request to LLM for trade gate decision (model: {LITELLM_TEXT_MODEL})...")
    try:
        response = litellm.completion(
            model=LITELLM_TEXT_MODEL,
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

def fetch_market_data(symbol, timeframe, limit=400):
    """Fetch raw klines data from MEXC API"""
    print(f"Fetching market data for {symbol} {timeframe} (limit={limit})")
    try:
        from pymexc import spot
        public_spot_client = spot.HTTP()
        klines = public_spot_client.klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        return klines
    except Exception as e:
        print(f"Error retrieving klines: {e}")
        return []

def fetch_market_dataframe(symbol, timeframe, limit=400):
    """Fetch market data and return as processed DataFrame"""
    klines = fetch_market_data(symbol, timeframe, limit=limit)
    
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
    indicator_name = condition.get('indicator')
    comparator = condition.get('comparator')
    threshold_value = condition.get('value')
    if indicator_name is None or comparator is None:
        return False
    
    if indicator_name == 'STOCH14_3_3' and isinstance(threshold_value, str):
        # Handle Stochastic comparisons like K vs D: e.g., "STOCH14_3_3 > d_percent"
        if 'STOCH_K' not in df.columns or 'STOCH_D' not in df.columns:
            return False
        k_val = df['STOCH_K'].iloc[-1]
        d_val = df['STOCH_D'].iloc[-1]
        compare_val = d_val if threshold_value.lower() == 'd_percent' else k_val
        return evaluate_comparison(k_val, comparator, compare_val)

    # Special-case derived indicators not stored as columns
    if indicator_name == 'PRICE':
        if df is None or df.empty or 'Close' not in df.columns:
            return False
        current_value = df['Close'].iloc[-1]
        return evaluate_comparison(current_value, comparator, threshold_value)

    if indicator_name == 'VOLUME':
        # Support ratio checks against average volume over lookback (default 20)
        if df is None or df.empty or 'Volume' not in df.columns:
            return False
        lookback = condition.get('lookback_candles', 20)
        baseline = (condition.get('baseline') or '').lower()
        current_vol = df['Volume'].iloc[-1]
        if baseline in ('average', 'average_volume', 'avg'):
            window = df['Volume'].iloc[-lookback:] if lookback and lookback > 0 else df['Volume']
            avg_vol = window.mean()
            if avg_vol is None or pd.isna(avg_vol) or avg_vol == 0:
                return False
            ratio = float(current_vol) / float(avg_vol)
            return evaluate_comparison(ratio, comparator, threshold_value)
        # Fallback: compare absolute volume
        return evaluate_comparison(current_vol, comparator, threshold_value)

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
    """Get list of technical indicators from LLM output.

    Supports both legacy schema with `opening_signal.checklist` and
    new schema with `opening_signal.core_checklist` and `opening_signal.secondary_checklist`.
    Returns concatenated core+secondary for new schema, or only items marked
    `technical_indicator` for legacy schema.
    """
    opening = llm_output.get('opening_signal', {}) or {}

    if 'core_checklist' in opening or 'secondary_checklist' in opening:
        core = opening.get('core_checklist', []) or []
        secondary = opening.get('secondary_checklist', []) or []
        def _supported(item):
            t = (item or {}).get('type')
            return t in ('indicator_threshold', 'indicator_crossover', 'indicator_condition', 'price_level')
        return [i for i in list(core) + list(secondary) if _supported(i)]

    technical_indicator_list = []
    for i in opening.get('checklist', []) or []:
        if i.get('technical_indicator') is True:
            technical_indicator_list.append(i)
    return technical_indicator_list

def indicator_checker(df, llm_output, emit_progress_fn=None):
    """Evaluate indicator checklist and return tuple:
    (all_core_met, num_core_met, total_core, any_secondary_met, details)

    New schema: uses `core_checklist` and `secondary_checklist`.
    Legacy: treats items from `checklist` with `technical_indicator=True` as core.
    """
    opening = llm_output.get('opening_signal', {}) or {}

    core_items = opening.get('core_checklist')
    secondary_items = opening.get('secondary_checklist')
    legacy_items = None
    if core_items is None and secondary_items is None:
        legacy_items = list_of_indicator_checker(llm_output)

    indicator_details = []
    num_core_met = 0
    total_core = 0
    any_secondary_met = False

    def eval_one(i):
        indicator_name = i.get('indicator')
        indicator_type = i.get('type') or 'indicator_threshold'
        condition_met = False
        current_value = None
        target_value = None
        condition_description = ""

        if indicator_type == 'indicator_threshold':
            condition_met = check_indicator_threshold(df, i)
            if indicator_name in df.columns:
                current_value = df[indicator_name].iloc[-1]
            elif indicator_name == 'MACD12_26_9' and 'MACD_Line' in df.columns:
                current_value = df['MACD_Line'].iloc[-1]
            elif indicator_name == 'STOCH14_3_3' and 'STOCH_K' in df.columns:
                current_value = df['STOCH_K'].iloc[-1]
            elif indicator_name == 'BB20_2_PercentB' and 'BB_PercentB' in df.columns:
                current_value = df['BB_PercentB'].iloc[-1]
            elif indicator_name == 'BB20_2_Bandwidth' and 'BB_Bandwidth' in df.columns:
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

        # Serialize values
        serializable_current_value = None
        if current_value is not None:
            try:
                if hasattr(current_value, 'isoformat'):
                    serializable_current_value = current_value.isoformat()
                elif hasattr(current_value, 'item'):
                    serializable_current_value = current_value.item()
                else:
                    serializable_current_value = float(current_value)
            except (ValueError, TypeError):
                serializable_current_value = str(current_value)

        serializable_target_value = None
        if target_value is not None:
            try:
                if hasattr(target_value, 'isoformat'):
                    serializable_target_value = target_value.isoformat()
                elif hasattr(target_value, 'item'):
                    serializable_target_value = target_value.item()
                else:
                    serializable_target_value = float(target_value)
            except (ValueError, TypeError):
                serializable_target_value = str(target_value)

        indicator_details.append({
            'name': indicator_name,
            'type': indicator_type,
            'condition': condition_description,
            'met': bool(condition_met),
            'current_value': serializable_current_value,
            'target_value': serializable_target_value
        })

        return condition_met

    if core_items is not None:
        total_core = len(core_items)
        for i in core_items:
            if eval_one(i):
                num_core_met += 1

    if secondary_items is not None:
        for i in secondary_items:
            if eval_one(i):
                any_secondary_met = True

    if legacy_items is not None:
        total_core = len(legacy_items)
        for i in legacy_items:
            if eval_one(i):
                num_core_met += 1

    all_core_met = (total_core > 0 and num_core_met == total_core)

    if emit_progress_fn and indicator_details:
        emit_progress_fn(json.dumps({
            "type": "indicator_details",
            "payload": make_json_serializable({
                "total_indicators": len(indicator_details),
                "core_total": total_core,
                "core_met": num_core_met,
                "secondary_any_met": any_secondary_met,
                "indicators": indicator_details
            })
        }))

    return all_core_met, num_core_met, total_core, any_secondary_met, indicator_details

def invalidation_checker(df, llm_output):
    """Check invalidation conditions from the LLM output"""
    opening = llm_output.get('opening_signal', {}) or {}
    invalidation_conditions = opening.get('invalidation', []) or []
    invalidation_triggered = False
    triggered_conditions = []
    
    for condition in invalidation_conditions:
        if not isinstance(condition, dict):
            continue
        condition_id = condition.get('id') or 'unknown'
        condition_type = condition.get('type')
        if not condition_type:
            # Skip invalid entries without a type
            continue
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
    """Comprehensive trading signal validation combining checklist and invalidation checks

    New pass rule: valid if all core met OR (>=2 core met AND strong pattern).
    Strong pattern: any pattern with confidence >= 0.75.
    """
    all_core_met, num_core_met, total_core, _, _ = indicator_checker(df, llm_output, emit_progress_fn)

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
        
        # Send notification for invalidated trade (to Telegram)
        try:
            trade_data = {
                "symbol": llm_output.get("symbol", "Unknown"),
                "current_price": market_values.get("current_price", 0),
                "triggered_conditions": triggered_conditions,
                "current_rsi": market_values.get("current_rsi", 0)
            }
            notification_results = notify_invalidated_trade(trade_data)
            if emit_progress_fn:
                emit_progress_fn(f"📱 Invalidation notifications sent - Telegram: {'✅' if notification_results.get('telegram') else '❌'}")
        except Exception as e:
            if emit_progress_fn:
                emit_progress_fn(f"⚠️ Invalidation notification failed: {str(e)}")

        # Attempt to close any open Hyperliquid position if enabled
        if os.getenv("HYPERLIQUID_ENABLE_ORDERS", "false").lower() in TRUTHY_VALUES:
            try:
                close_result = close_trade_with_hyperliquid(
                    llm_output.get("symbol", ""),
                    market_values.get("current_price"),
                )
                if emit_progress_fn:
                    if close_result.get("ok"):
                        message = close_result.get("message") or "Hyperliquid position close submitted"
                        emit_progress_fn(f"Hyperliquid: {message}")
                    else:
                        emit_progress_fn(f"Hyperliquid: Close failed - {close_result.get('error')}")
            except Exception as exc:
                if emit_progress_fn:
                    emit_progress_fn(f"Hyperliquid: Close exception - {str(exc)}")
        
        return False, "invalidated", triggered_conditions, market_values
    else:
        # Pattern strength check
        patterns = llm_output.get('pattern_analysis', []) or []
        strong_pattern = any((p.get('confidence') or 0) >= 0.75 for p in patterns)
        if all_core_met or (num_core_met >= 2 and strong_pattern):
            return True, "valid", [], market_values
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
        # Check if stop was requested
        if should_stop_analysis():
            if emit_progress_fn:
                emit_progress_fn("⏹️ Polling stopped by user request")
            set_analysis_running(False)
            send_telegram_status("⏹️ <b>Analysis Stopped</b>\n\nPolling was cancelled by user.")
            return False, "stopped", [], {}
        
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

        # Send periodic status updates
        if cycles % 5 == 0:  # Every 5 cycles
            elapsed_min = int((cycles * wait_seconds) / 60)
            send_telegram_status(f"⏳ <b>Still Polling...</b>\n\nCycle {cycles}\nElapsed: {elapsed_min}m\nStatus: {signal_status}")

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
        # Mark analysis as running
        set_analysis_running(True)
        send_telegram_status("🚀 <b>Analysis Started</b>\n\nPreparing market data...")
        
        # Step 1: Prepare market data BEFORE calling the LLM
        emit_progress("Step 1: Preparing market data (if provided)...", 1, 14)
        df = pd.DataFrame()
        df_hourly = pd.DataFrame()
        df_daily = pd.DataFrame()
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
            def _fetch_market_dataframe(sym: str, tf: str, limit: int = 400) -> pd.DataFrame:
                kl = public_spot_client_local.klines(symbol=sym, interval=tf, limit=limit)
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

            # Fetch primary timeframe data (400 candles)
            emit_progress(f"Step 1.1: Fetching {effective_timeframe} data (400 candles)...", 1, 14)
            df = _fetch_market_dataframe(effective_symbol, effective_timeframe, limit=400)
            
            # Fetch hourly data for trend context (100 candles)
            emit_progress("Step 1.2: Fetching hourly data (100 candles) for trend context...", 1, 14)
            try:
                df_hourly = _fetch_market_dataframe(effective_symbol, '60m', limit=100)
                if not df_hourly.empty:
                    emit_progress(f"Step 1.2: Fetched {len(df_hourly)} hourly candles", 1, 14)
            except Exception as e:
                emit_progress(f"Step 1.2 Warning: Failed to fetch hourly data: {e}", 1, 14)
            
            # Fetch daily data for longer-term trend context (100 candles)
            emit_progress("Step 1.3: Fetching daily data (100 candles) for long-term trend context...", 1, 14)
            try:
                df_daily = _fetch_market_dataframe(effective_symbol, '1d', limit=100)
                if not df_daily.empty:
                    emit_progress(f"Step 1.3: Fetched {len(df_daily)} daily candles", 1, 14)
            except Exception as e:
                emit_progress(f"Step 1.3 Warning: Failed to fetch daily data: {e}", 1, 14)
            
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
                emit_progress(f"Step 1 Complete: Market data prepared ({len(df)} {effective_timeframe} candles, {len(df_hourly)} hourly, {len(df_daily)} daily)", 1, 14)
            
        except Exception as e:
            emit_progress(f"Step 1 Warning: Market data prep failed: {e}. Proceeding image-only.", 1, 14)

        # Step 2: Analyze trading chart with multiple models in parallel
        emit_progress("Step 2: Analyzing trading chart with multiple models...", 2, 14)
        # Prepare market data dict with all timeframes
        market_data_dict = None
        if not df.empty:
            market_data_dict = {
                'minute': df,
                'hourly': df_hourly if not df_hourly.empty else None,
                'daily': df_daily if not df_daily.empty else None
            }
        multi_model_comparison = analyze_trading_chart_multi_model(
            image_path,
            symbol=effective_symbol,
            timeframe=effective_timeframe,
            df=market_data_dict,
        )
        emit_progress("Step 2 Complete: Multi-model analysis finished", 2, 14)
        
        # Select a model for the rest of the pipeline
        # Note: We don't select based on confidence - user will evaluate manually
        # For now, we use consensus if all models agree, otherwise first successful model
        llm_output = None
        selected_model = None
        selection_reason = None
        
        if multi_model_comparison.get("results"):
            results = multi_model_comparison["results"]
            summary = multi_model_comparison.get("summary", {})
            
            # Check if all models agree on direction (consensus)
            agreement = summary.get("agreement", False)
            consensus_direction = summary.get("consensus_direction")
            
            if agreement and consensus_direction:
                # All models agree - use first model (they're all the same direction anyway)
                selected_model = list(results.keys())[0]
                llm_output = results[selected_model]["result"]
                selection_reason = f"consensus (all models agree on {consensus_direction})"
                emit_progress(f"Step 2.1: Using {selected_model} - {selection_reason}", 2, 14)
            else:
                # Models disagree - use first successful model
                # User can review all results manually to determine which is best
                selected_model = list(results.keys())[0]
                llm_output = results[selected_model]["result"]
                selection_reason = "first successful model (models disagree - manual review recommended)"
                emit_progress(f"Step 2.1: Using {selected_model} - {selection_reason}", 2, 14)
                emit_progress(f"Step 2.1 Note: Models disagree on direction. Review all results to determine best model.", 2, 14)
        else:
            # No successful results - create fallback
            emit_progress("Step 2 Warning: No models succeeded, using fallback response", 2, 14)
            llm_output = create_fallback_llm_response(effective_symbol, effective_timeframe)
            selected_model = "fallback"
            selection_reason = "fallback (no models succeeded)"
        
        # Store multi-model comparison info in llm_output for reference
        llm_output["_multi_model_comparison"] = {
            "selected_model": selected_model,
            "selection_reason": selection_reason,
            "note": "Model selected for pipeline execution only - not based on confidence. Review all results to determine best model.",
            "comparison_summary": multi_model_comparison.get("summary", {}),
            "all_results": {k: {
                "direction": v["result"].get("opening_signal", {}).get("direction"), 
                "confidence": v["result"].get("validity_assessment", {}).get("core_alignment_score", 0),
                "elapsed_time": v["elapsed_time"],
                "stop_loss": v["result"].get("risk_management", {}).get("stop_loss", {}).get("price") if v["result"].get("risk_management", {}).get("stop_loss") else None,
                "take_profits": [{"price": tp.get("price"), "rr": tp.get("rr")} for tp in v["result"].get("risk_management", {}).get("take_profit", []) if tp.get("price") is not None],
                "max_rr": max([tp.get("rr", 0) for tp in v["result"].get("risk_management", {}).get("take_profit", []) if tp.get("rr") is not None], default=None)
            } 
                           for k, v in multi_model_comparison.get("results", {}).items()},
            "errors": multi_model_comparison.get("errors", {})
        }
        
        # Send initial analysis to Telegram immediately
        try:
            emit_progress("Step 2.5: Sending initial analysis to Telegram...", 2, 14)
            telegram_success = send_initial_analysis_to_telegram(llm_output)
            if telegram_success:
                emit_progress("Step 2.5 Complete: Initial analysis sent to Telegram ✅", 2, 14)
            else:
                emit_progress("Step 2.5 Warning: Failed to send initial analysis to Telegram", 2, 14)
        except Exception as e:
            emit_progress(f"Step 2.5 Warning: Telegram notification error: {str(e)}", 2, 14)

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
        
        # Update analysis state with symbol and direction
        direction = llm_output.get('opening_signal', {}).get('direction', 'Unknown')
        set_analysis_info(symbol, direction)
        
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

        # Send proposed trade and polling details to Telegram (only if not fallback)
        # This comes after "Preparing market data..." and before "Still Polling..." messages
        if not is_fallback:
            try:
                emit_progress("Step 4.5: Sending proposed trade and polling details to Telegram...", 4, 14)
                wait_seconds = _timeframe_seconds(timeframe)
                telegram_success = send_polling_start_to_telegram(llm_output, timeframe, wait_seconds)
                if telegram_success:
                    emit_progress("Step 4.5 Complete: Proposed trade and polling details sent to Telegram ✅", 4, 14)
                else:
                    emit_progress("Step 4.5 Warning: Failed to send proposed trade notification to Telegram", 4, 14)
            except Exception as e:
                emit_progress(f"Step 4.5 Warning: Telegram polling notification error: {str(e)}", 4, 14)

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
        gate_path = None  # Initialize gate_path to avoid reference errors
        
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
                    if os.getenv("MEXC_ENABLE_ORDERS", "false").lower() in TRUTHY_VALUES:
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

                # Mirror execution on Hyperliquid if enabled
                try:
                    if os.getenv("HYPERLIQUID_ENABLE_ORDERS", "false").lower() in TRUTHY_VALUES:
                        emit_progress("Hyperliquid: Placing futures order...")
                        hl_outcome = open_trade_with_hyperliquid(llm_output.get("symbol", ""), gate_result)
                        if hl_outcome.get("ok"):
                            emit_progress("Hyperliquid: Order placed successfully")
                        else:
                            emit_progress(f"Hyperliquid: Order placement error - {hl_outcome.get('error')}")
                except Exception as e:
                    emit_progress(f"Hyperliquid: Order placement exception - {str(e)}")

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
                        "take_profits": take_profits,
                        "_multi_model_comparison": llm_output.get("_multi_model_comparison")  # Include multi-model comparison data
                    }
                    notification_results = notify_valid_trade(trade_data)
                    emit_progress(f"📱 Notifications sent - Telegram: {'✅' if notification_results.get('telegram') else '❌'}", 14, 14)
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
                
                # Send Telegram notification for rejected trade
                try:
                    rejection_data = {
                        "symbol": llm_output.get("symbol", "Unknown"),
                        "direction": direction,
                        "current_price": market_values.get("current_price", 0),
                        "confidence": confidence,
                        "reasons": reasons,
                        "warnings": gate_result.get("warnings", []),
                        "checks": gate_result.get("checks", {})
                    }
                    notification_results = notify_rejected_trade(rejection_data)
                    emit_progress(f"📱 Rejection notifications sent - Telegram: {'✅' if notification_results.get('telegram') else '❌'}", 14, 14)
                except Exception as e:
                    emit_progress(f"⚠️ Rejection notification failed: {str(e)}", 14, 14)
        else:
            emit_progress(f"Step 11 Complete: Signal not valid (status: {signal_status}), skipping trade gate", 11, 14)

        # Mark analysis as complete
        set_analysis_running(False)
        send_telegram_status("✅ <b>Analysis Complete</b>\n\nThe analysis has finished.")

        return {
            "success": True,
            "llm_output": llm_output,
            "signal_status": signal_status,
            "signal_valid": signal_valid,
            "market_values": market_values,
            "gate_result": gate_result,
            "multi_model_comparison": multi_model_comparison,
            "selected_model": selected_model,
            "output_files": {
                "llm_output": output_path,
                "gate_result": gate_path if gate_result else None,
                "multi_model_comparison": multi_model_comparison.get("comparison_file_path") if multi_model_comparison else None
            }
        }

    except Exception as e:
        emit_progress(f"ERROR: Analysis failed with {type(e).__name__}: {str(e)}")
        # Mark analysis as stopped on error
        set_analysis_running(False)
        send_telegram_status(f"❌ <b>Analysis Failed</b>\n\nError: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.route('/api/upload-image', methods=['POST'])
def upload_image_api():
    """
    Enhanced API endpoint for image upload with optional analysis
    
    Parameters:
    - image: File (required) - The trading chart image
    - auto_analyze: Boolean (optional) - Whether to automatically start analysis (default: false)
    - save_permanently: Boolean (optional) - Whether to save the image permanently (default: false)
    - filename: String (optional) - Custom filename for saved image
    
    Returns:
    - success: Boolean
    - message: String
    - image_path: String (if saved permanently)
    - job_id: String (if auto_analyze is true)
    - metadata: Object with image details
    """
    try:
        # Validate image file
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided in request'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get optional parameters
        auto_analyze = request.form.get('auto_analyze', 'false').lower() == 'true'
        save_permanently = request.form.get('save_permanently', 'false').lower() == 'true'
        custom_filename = request.form.get('filename', '')
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_ext = file.filename.rsplit('.', 1)[1].lower()
        
        if custom_filename:
            # Sanitize custom filename
            safe_custom_name = secure_filename(custom_filename)
            if not safe_custom_name.endswith(f'.{original_ext}'):
                safe_custom_name = f"{safe_custom_name}.{original_ext}"
            filename = safe_custom_name
        else:
            filename = f"trading_chart_{timestamp}.{original_ext}"
        
        # Determine save path
        if save_permanently:
            # Save to project directory
            uploads_dir = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            save_path = os.path.join(uploads_dir, filename)
            file.save(save_path)
            image_path = save_path
            temp_file = False
        else:
            # Save as temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{original_ext}') as tmp_file:
                file.save(tmp_file.name)
                image_path = tmp_file.name
                temp_file = True
        
        # Get image metadata
        file_size = os.path.getsize(image_path)
        
        response_data = {
            'success': True,
            'message': 'Image uploaded successfully',
            'metadata': {
                'filename': filename,
                'size_bytes': file_size,
                'size_kb': round(file_size / 1024, 2),
                'timestamp': timestamp,
                'file_type': original_ext,
                'saved_permanently': save_permanently
            }
        }
        
        # Add image path if saved permanently
        if save_permanently:
            response_data['image_path'] = image_path
        
        # Send image to Telegram immediately upon upload
        try:
            telegram_success = send_image_to_telegram(
                image_path,
                caption=f"<b>📊 New Trading Chart Uploaded</b>\n\n<b>Filename:</b> {filename}\n<b>Size:</b> {round(file_size / 1024, 2)} KB\n<b>Time:</b> {timestamp}"
            )
            if telegram_success:
                response_data['telegram_sent'] = True
                print(f"✅ Image sent to Telegram successfully")
            else:
                response_data['telegram_sent'] = False
                print(f"⚠️ Failed to send image to Telegram")
        except Exception as e:
            response_data['telegram_sent'] = False
            print(f"⚠️ Telegram send exception: {str(e)}")
        
        # Start analysis if requested
        if auto_analyze:
            # Clear the progress queue before starting
            while not progress_queue.empty():
                progress_queue.get()
            
            job_id = f'analysis_{timestamp}'
            
            # Run analysis in background
            def run_analysis_and_cleanup():
                try:
                    # Run the complete trading analysis pipeline
                    # This includes: LLM analysis, market data fetching, signal validation,
                    # trade gate decision, and all notifications (Telegram)
                    result = run_trading_analysis(image_path)
                    
                    # run_trading_analysis() already handles all notifications internally:
                    # - notify_valid_trade() for approved trades (sent in Step 14)
                    # - notify_invalidated_trade() for invalidated signals (sent during polling)
                    # These notifications are sent to Telegram
                    
                    if result.get("success"):
                        print(f"✅ Analysis completed successfully - all notifications sent via run_trading_analysis()")
                    else:
                        print(f"⚠️ Analysis failed: {result.get('error', 'Unknown error')}")
                        # Send error notification to Telegram
                        try:
                            from services.notification_service import NotificationService
                            service = NotificationService()
                            error_message = f"""
<b>❌ Analysis Error</b>

<b>Image:</b> {filename}
<b>Error:</b> {result.get('error', 'Unknown error')}
<b>Time:</b> {timestamp}

Please check the logs for more details.
                            """.strip()
                            service.send_telegram_image(image_path, error_message)
                        except Exception as e:
                            print(f"⚠️ Failed to send error notification: {str(e)}")
                            
                except Exception as e:
                    print(f"❌ Analysis exception: {str(e)}")
                    # Send exception notification to Telegram
                    try:
                        from services.notification_service import NotificationService
                        service = NotificationService()
                        error_message = f"""
<b>❌ Analysis Exception</b>

<b>Image:</b> {filename}
<b>Exception:</b> {str(e)}
<b>Time:</b> {timestamp}

Please check the logs for more details.
                        """.strip()
                        service.send_telegram_image(image_path, error_message)
                    except Exception as telegram_error:
                        print(f"⚠️ Failed to send exception notification: {str(telegram_error)}")
                finally:
                    # Only delete temp files
                    if temp_file and os.path.exists(image_path):
                        os.unlink(image_path)
            
            analysis_thread = threading.Thread(target=run_analysis_and_cleanup)
            analysis_thread.start()
            
            response_data['job_id'] = job_id
            response_data['message'] = 'Image uploaded and analysis started - notifications will be sent to Telegram upon completion'
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/list-uploads')
def list_uploads():
    """List all uploaded images in the uploads directory"""
    try:
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        if not os.path.exists(uploads_dir):
            return jsonify({
                'success': True,
                'images': [],
                'count': 0
            })
        
        images = []
        for filename in os.listdir(uploads_dir):
            # Check if it's an image file
            if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
                file_path = os.path.join(uploads_dir, filename)
                file_stat = os.stat(file_path)
                
                images.append({
                    'filename': filename,
                    'filepath': file_path,
                    'size_bytes': file_stat.st_size,
                    'size_kb': round(file_stat.st_size / 1024, 2),
                    'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
                    'uploaded_at': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                    'modified_at': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sort by upload time (newest first)
        images.sort(key=lambda x: x['uploaded_at'], reverse=True)
        
        return jsonify({
            'success': True,
            'images': images,
            'count': len(images)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    werkzeug_run_main = os.environ.get("WERKZEUG_RUN_MAIN")
    if (werkzeug_run_main and werkzeug_run_main.lower() == "true") or not app.debug:
        ensure_telegram_bot_started()

    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
