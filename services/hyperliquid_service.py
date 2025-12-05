#!/usr/bin/env python3
"""
Hyperliquid Trading Service

Provides trading functionality for Hyperliquid exchange with:
- Limit orders only (to minimize fees)
- Leverage management
- Position monitoring
- MEXC price comparison for USDT/USDC differences
"""

import os
import json
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

class HyperliquidConfig:
    """Configuration for Hyperliquid trading"""
    
    @staticmethod
    def get_private_key() -> Optional[str]:
        return os.getenv("HYPERLIQUID_PRIVATE_KEY")
    
    @staticmethod
    def get_base_url() -> Optional[str]:
        """Get base URL - None means mainnet"""
        explicit_url = os.getenv("HYPERLIQUID_BASE_URL")
        if explicit_url:
            return explicit_url
        
        testnet = os.getenv("HYPERLIQUID_TESTNET", "false").lower() in ("true", "1", "yes")
        if testnet:
            return "https://api.hyperliquid-testnet.xyz"
        return None
    
    @staticmethod
    def is_testnet() -> bool:
        base_url = HyperliquidConfig.get_base_url()
        return base_url is not None and "testnet" in base_url.lower()
    
    @staticmethod
    def orders_enabled() -> bool:
        return os.getenv("HYPERLIQUID_ENABLE_ORDERS", "false").lower() in ("true", "1", "yes")
    
    @staticmethod
    def get_default_size() -> float:
        """Get default position size. If 0 or 'full', will use full account."""
        size_str = os.getenv("HYPERLIQUID_DEFAULT_SIZE", "0")
        if size_str.lower() in ("full", "max", "all"):
            return 0  # 0 means use full account
        try:
            return float(size_str)
        except:
            return 0
    
    @staticmethod
    def use_full_account() -> bool:
        """Check if should use full account balance for position sizing"""
        size_str = os.getenv("HYPERLIQUID_DEFAULT_SIZE", "0")
        if size_str.lower() in ("full", "max", "all", "0"):
            return True
        try:
            return float(size_str) <= 0
        except:
            return True
    
    @staticmethod
    def get_leverage() -> int:
        """Get configured leverage (default: 30)"""
        try:
            return int(os.getenv("HYPERLIQUID_LEVERAGE", "30"))
        except:
            return 30
    
    @staticmethod
    def get_max_leverage() -> int:
        """Get maximum allowed leverage (safety limit, default: 50)"""
        try:
            return int(os.getenv("HYPERLIQUID_MAX_LEVERAGE", "50"))
        except:
            return 50
    
    @staticmethod
    def get_slippage() -> float:
        """Get slippage tolerance for limit orders (percentage as decimal)"""
        try:
            return float(os.getenv("HYPERLIQUID_SLIPPAGE", "0.001"))  # 0.1% default for limit
        except:
            return 0.001
    
    @staticmethod
    def get_limit_tif() -> str:
        """Get time-in-force for limit orders: Gtc, Ioc, or Alo"""
        return os.getenv("HYPERLIQUID_LIMIT_TIF", "Gtc")


# ============================================================================
# Client Initialization (with caching for performance)
# ============================================================================

# Cached clients for faster repeated calls
_cached_clients: Optional[Dict[str, Any]] = None
_client_cache_time: float = 0
_CLIENT_CACHE_TTL = 300  # 5 minutes


def get_hyperliquid_clients(force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """Initialize and return Hyperliquid clients (cached for performance)"""
    global _cached_clients, _client_cache_time
    
    # Return cached clients if still valid
    if not force_refresh and _cached_clients is not None:
        if (time.time() - _client_cache_time) < _CLIENT_CACHE_TTL:
            return _cached_clients
    
    try:
        from hyperliquid.info import Info
        from hyperliquid.exchange import Exchange
        from eth_account import Account
    except ImportError as e:
        print(f"❌ Hyperliquid SDK not installed: {e}")
        print("   Run: pip install hyperliquid-python-sdk eth-account")
        return None
    
    base_url = HyperliquidConfig.get_base_url()
    
    # Public info client (no auth needed)
    try:
        info_client = Info(base_url=base_url, skip_ws=True)
    except Exception as e:
        print(f"❌ Failed to create Info client: {e}")
        return None
    
    # Private exchange client (needs private key)
    exchange_client = None
    wallet_address = None
    private_key = HyperliquidConfig.get_private_key()
    
    if private_key:
        try:
            account = Account.from_key(private_key)
            exchange_client = Exchange(account, base_url=base_url)
            wallet_address = account.address
        except Exception as e:
            print(f"⚠️ Could not initialize exchange client: {e}")
    
    _cached_clients = {
        "info": info_client,
        "exchange": exchange_client,
        "wallet_address": wallet_address,
        "base_url": base_url,
        "is_testnet": HyperliquidConfig.is_testnet(),
        "network": "TESTNET" if HyperliquidConfig.is_testnet() else "MAINNET"
    }
    _client_cache_time = time.time()
    
    return _cached_clients


# ============================================================================
# Symbol Conversion
# ============================================================================

def convert_symbol_to_hyperliquid(symbol: str) -> str:
    """
    Convert TradingView/MEXC symbol to Hyperliquid format.
    
    Examples:
        BTCUSDT.P -> BTC
        BTCUSDT -> BTC
        BTC/USDT -> BTC
        BTC -> BTC
    """
    if not symbol:
        return "BTC"
    
    # Remove common suffixes in order of priority
    cleaned = symbol.upper()
    
    # First remove .P suffix
    if cleaned.endswith(".P"):
        cleaned = cleaned[:-2]
    
    # Then remove quote currency suffixes
    for suffix in ["/USDT", "/USDC", "/USD", "USDT", "USDC", "USD"]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break
    
    return cleaned or "BTC"


# ============================================================================
# Price Utilities
# ============================================================================

def get_hyperliquid_price(coin: str = "BTC") -> Optional[float]:
    """Get current mid price from Hyperliquid"""
    clients = get_hyperliquid_clients()
    if not clients or not clients.get("info"):
        return None
    
    try:
        mids = clients["info"].all_mids()
        if mids and coin in mids:
            return float(mids[coin])
    except Exception as e:
        print(f"❌ Failed to get Hyperliquid price: {e}")
    return None


def get_mexc_price(symbol: str = "BTCUSDT") -> Optional[float]:
    """Get current price from MEXC for comparison"""
    try:
        from pymexc import spot
        public_client = spot.HTTP()
        ticker = public_client.ticker_price(symbol=symbol)
        if ticker and "price" in ticker:
            return float(ticker["price"])
    except Exception as e:
        print(f"⚠️ Failed to get MEXC price: {e}")
    return None


def get_price_difference(coin: str = "BTC") -> Dict[str, Any]:
    """
    Compare prices between Hyperliquid (USDC) and MEXC (USDT).
    Returns price info and percentage difference.
    """
    hl_price = get_hyperliquid_price(coin)
    mexc_symbol = f"{coin}USDT"
    mexc_price = get_mexc_price(mexc_symbol)
    
    result = {
        "hyperliquid_price": hl_price,
        "mexc_price": mexc_price,
        "difference": None,
        "difference_pct": None,
        "significant": False
    }
    
    if hl_price and mexc_price:
        diff = hl_price - mexc_price
        diff_pct = (diff / mexc_price) * 100
        result["difference"] = diff
        result["difference_pct"] = diff_pct
        # Flag if difference is > 0.1%
        result["significant"] = abs(diff_pct) > 0.1
    
    return result


# ============================================================================
# Leverage Management
# ============================================================================

def get_current_leverage(coin: str = "BTC") -> Optional[Dict[str, Any]]:
    """Get current leverage settings for a coin"""
    clients = get_hyperliquid_clients()
    if not clients or not clients.get("info") or not clients.get("wallet_address"):
        return None
    
    try:
        user_state = clients["info"].user_state(clients["wallet_address"])
        positions = user_state.get("assetPositions", [])
        
        for pos_data in positions:
            pos = pos_data.get("position", {})
            if pos.get("coin") == coin:
                leverage_info = pos.get("leverage", {})
                if isinstance(leverage_info, dict):
                    return {
                        "value": leverage_info.get("value", 1),
                        "type": leverage_info.get("type", "cross")
                    }
                return {"value": leverage_info, "type": "unknown"}
        
        # No position, return account default
        return {"value": 1, "type": "cross"}
    except Exception as e:
        print(f"❌ Failed to get leverage: {e}")
    return None


def set_leverage(coin: str, leverage: int, leverage_type: str = "cross") -> Dict[str, Any]:
    """
    Set leverage for a coin.
    
    Args:
        coin: The coin (e.g., "BTC")
        leverage: Leverage value (1-50 typically, depends on coin)
        leverage_type: "cross" or "isolated"
    
    Returns:
        Dict with success status and response
    """
    clients = get_hyperliquid_clients()
    if not clients or not clients.get("exchange"):
        return {"ok": False, "error": "Exchange client not initialized"}
    
    if not HyperliquidConfig.orders_enabled():
        return {"ok": False, "error": "Orders not enabled (HYPERLIQUID_ENABLE_ORDERS=false)"}
    
    # Safety check
    max_lev = HyperliquidConfig.get_max_leverage()
    if leverage > max_lev:
        return {"ok": False, "error": f"Leverage {leverage}x exceeds max allowed {max_lev}x"}
    
    try:
        exchange = clients["exchange"]
        # Hyperliquid uses update_leverage method
        response = exchange.update_leverage(
            leverage=leverage,
            name=coin,
            is_cross=leverage_type.lower() == "cross"
        )
        return {"ok": True, "response": response, "leverage": leverage, "type": leverage_type}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def check_and_set_leverage(coin: str) -> Dict[str, Any]:
    """
    Check current leverage and set to configured value if different.
    
    Returns:
        Dict with current leverage info and any changes made
    """
    current = get_current_leverage(coin)
    target_leverage = HyperliquidConfig.get_leverage()
    
    result = {
        "current_leverage": current,
        "target_leverage": target_leverage,
        "changed": False,
        "error": None
    }
    
    if current is None:
        result["error"] = "Could not get current leverage"
        return result
    
    current_value = current.get("value", 1)
    
    if current_value != target_leverage:
        # Need to change leverage
        set_result = set_leverage(coin, target_leverage)
        if set_result.get("ok"):
            result["changed"] = True
            result["new_leverage"] = target_leverage
        else:
            result["error"] = set_result.get("error")
    
    return result


# ============================================================================
# Account & Position Management
# ============================================================================

def get_account_info() -> Optional[Dict[str, Any]]:
    """Get account balance and margin info"""
    clients = get_hyperliquid_clients()
    if not clients or not clients.get("info") or not clients.get("wallet_address"):
        return None
    
    try:
        user_state = clients["info"].user_state(clients["wallet_address"])
        margin_summary = user_state.get("marginSummary", {})
        
        return {
            "account_value": float(margin_summary.get("accountValue", 0)),
            "total_margin_used": float(margin_summary.get("totalMarginUsed", 0)),
            "withdrawable": float(user_state.get("withdrawable", 0)),
            "network": clients["network"],
            "wallet_address": clients["wallet_address"]
        }
    except Exception as e:
        print(f"❌ Failed to get account info: {e}")
    return None


def get_open_positions() -> List[Dict[str, Any]]:
    """Get all open positions"""
    clients = get_hyperliquid_clients()
    if not clients or not clients.get("info") or not clients.get("wallet_address"):
        return []
    
    try:
        user_state = clients["info"].user_state(clients["wallet_address"])
        positions = user_state.get("assetPositions", [])
        
        active_positions = []
        for pos_data in positions:
            pos = pos_data.get("position", {})
            szi = float(pos.get("szi", 0))
            
            if abs(szi) > 1e-12:
                leverage_info = pos.get("leverage", {})
                if isinstance(leverage_info, dict):
                    lev_value = leverage_info.get("value", 1)
                else:
                    lev_value = leverage_info
                
                active_positions.append({
                    "coin": pos.get("coin", ""),
                    "direction": "LONG" if szi > 0 else "SHORT",
                    "size": abs(szi),
                    "entry_price": float(pos.get("entryPx", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "leverage": lev_value,
                    "margin_used": float(pos.get("marginUsed", 0)),
                    "liquidation_px": float(pos.get("liquidationPx", 0)) if pos.get("liquidationPx") else None
                })
        
        return active_positions
    except Exception as e:
        print(f"❌ Failed to get positions: {e}")
    return []


def get_position(coin: str) -> Optional[Dict[str, Any]]:
    """Get position for a specific coin"""
    positions = get_open_positions()
    for pos in positions:
        if pos["coin"] == coin:
            return pos
    return None


def get_positions_fast() -> Dict[str, Any]:
    """
    Get positions and account info in a single API call for speed.
    Returns both positions and account info from one user_state fetch.
    """
    clients = get_hyperliquid_clients()
    if not clients or not clients.get("info") or not clients.get("wallet_address"):
        return {"positions": [], "account": None, "error": "Client not initialized"}
    
    try:
        user_state = clients["info"].user_state(clients["wallet_address"])
        
        # Extract account info
        margin_summary = user_state.get("marginSummary", {})
        account = {
            "account_value": float(margin_summary.get("accountValue", 0)),
            "total_margin_used": float(margin_summary.get("totalMarginUsed", 0)),
            "withdrawable": float(user_state.get("withdrawable", 0)),
        }
        
        # Extract positions
        positions_data = user_state.get("assetPositions", [])
        positions = []
        
        for pos_data in positions_data:
            pos = pos_data.get("position", {})
            szi = float(pos.get("szi", 0))
            
            if abs(szi) > 1e-12:
                leverage_info = pos.get("leverage", {})
                if isinstance(leverage_info, dict):
                    lev_value = leverage_info.get("value", 1)
                else:
                    lev_value = leverage_info
                
                positions.append({
                    "coin": pos.get("coin", ""),
                    "direction": "LONG" if szi > 0 else "SHORT",
                    "size": abs(szi),
                    "entry_price": float(pos.get("entryPx", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "leverage": lev_value,
                    "margin_used": float(pos.get("marginUsed", 0)),
                    "liquidation_px": float(pos.get("liquidationPx", 0)) if pos.get("liquidationPx") else None
                })
        
        return {"positions": positions, "account": account, "error": None}
        
    except Exception as e:
        return {"positions": [], "account": None, "error": str(e)}


# ============================================================================
# Order Placement (Limit Orders Only)
# ============================================================================

def calculate_limit_price(
    coin: str,
    is_buy: bool,
    slippage: Optional[float] = None
) -> Optional[float]:
    """
    Calculate limit price based on current mid price and slippage.
    
    For buys: price = mid * (1 + slippage)  -- willing to pay slightly more
    For sells: price = mid * (1 - slippage)  -- willing to receive slightly less
    """
    current_price = get_hyperliquid_price(coin)
    if not current_price:
        return None
    
    if slippage is None:
        slippage = HyperliquidConfig.get_slippage()
    
    if is_buy:
        return current_price * (1 + slippage)
    else:
        return current_price * (1 - slippage)


def round_price(price: float, coin: str = "BTC") -> float:
    """
    Round price to appropriate tick size for the coin.
    
    Hyperliquid tick sizes:
    - BTC: $1 (whole dollars)
    - ETH: $0.1
    - Most others: $0.01 or $0.001
    """
    # Try to get tick size from exchange metadata
    clients = get_hyperliquid_clients()
    if clients and clients.get("info"):
        try:
            meta = clients["info"].meta()
            for asset in meta.get("universe", []):
                if asset.get("name") == coin:
                    # szDecimals is for size, but we can infer price precision
                    # BTC has szDecimals=5, price tick=$1
                    # For safety, use known tick sizes
                    break
        except:
            pass
    
    # Known tick sizes for Hyperliquid (price must be divisible by tick)
    tick_sizes = {
        "BTC": 1.0,      # $1 tick
        "ETH": 0.1,      # $0.10 tick
        "SOL": 0.01,     # $0.01 tick
        "DOGE": 0.0001,  # $0.0001 tick
        "XRP": 0.0001,
        "AVAX": 0.01,
        "LINK": 0.01,
        "ARB": 0.0001,
        "OP": 0.001,
        "MATIC": 0.0001,
        "APT": 0.01,
        "LTC": 0.01,
        "BCH": 0.1,
        "ATOM": 0.001,
        "DOT": 0.001,
        "UNI": 0.001,
        "FIL": 0.001,
        "NEAR": 0.001,
        "INJ": 0.01,
        "TIA": 0.001,
    }
    
    tick = tick_sizes.get(coin, 0.01)  # Default to $0.01
    
    # Round to nearest tick
    return round(price / tick) * tick


def round_size(size: float, coin: str = "BTC") -> float:
    """Round size to appropriate precision for the coin"""
    # Check meta for size decimals
    clients = get_hyperliquid_clients()
    if clients and clients.get("info"):
        try:
            meta = clients["info"].meta()
            for asset in meta.get("universe", []):
                if asset.get("name") == coin:
                    sz_decimals = asset.get("szDecimals", 3)
                    return round(size, sz_decimals)
        except:
            pass
    
    # Defaults
    if coin == "BTC":
        return round(size, 5)  # 0.00001 BTC minimum
    elif coin == "ETH":
        return round(size, 4)
    else:
        return round(size, 3)


def calculate_position_size(
    coin: str,
    leverage: int,
    account_fraction: float = 1.0,
    price: Optional[float] = None
) -> Optional[float]:
    """
    Calculate position size based on account balance and leverage.
    
    Args:
        coin: The coin to trade (e.g., "BTC")
        leverage: Leverage to use
        account_fraction: Fraction of account to use (1.0 = 100%, 0.5 = 50%)
        price: Current price (if None, fetches from exchange)
    
    Returns:
        Position size in coin units, or None if can't calculate
    """
    # Get account info
    account = get_account_info()
    if not account:
        return None
    
    account_value = account.get("account_value", 0)
    if account_value <= 0:
        return None
    
    # Get current price
    if price is None:
        price = get_hyperliquid_price(coin)
    if not price or price <= 0:
        return None
    
    # Calculate position value with leverage
    # Position Value = Account Value * Leverage * Account Fraction
    # Position Size = Position Value / Price
    usable_balance = account_value * account_fraction
    position_value = usable_balance * leverage
    position_size = position_value / price
    
    # Round to appropriate precision
    position_size = round_size(position_size, coin)
    
    return position_size


def get_max_position_size(coin: str) -> Dict[str, Any]:
    """
    Get the maximum position size that can be opened with current account.
    
    Returns:
        Dict with size info including:
        - max_size: Maximum position size in coin units
        - account_value: Current account value
        - leverage: Leverage being used
        - position_value: Total position value in USD
        - current_price: Current market price
    """
    leverage = HyperliquidConfig.get_leverage()
    account = get_account_info()
    price = get_hyperliquid_price(coin)
    
    if not account or not price:
        return {"error": "Could not fetch account or price info"}
    
    account_value = account.get("account_value", 0)
    max_size = calculate_position_size(coin, leverage, 1.0, price)
    
    return {
        "max_size": max_size,
        "account_value": account_value,
        "leverage": leverage,
        "position_value": account_value * leverage,
        "current_price": price,
        "coin": coin
    }


def open_position_limit(
    coin: str,
    direction: str,
    size: Optional[float] = None,
    price: Optional[float] = None,
    reduce_only: bool = False,
    account_fraction: float = 1.0
) -> Dict[str, Any]:
    """
    Open a position using a LIMIT order.
    
    Args:
        coin: The coin to trade (e.g., "BTC")
        direction: "long" or "short"
        size: Position size in coin units. If None or 0, uses full account with leverage
        price: Limit price. If None, calculates based on current price + slippage
        reduce_only: Whether this is a reduce-only order
        account_fraction: Fraction of account to use when calculating size (1.0 = 100%)
    
    Returns:
        Dict with order result
    """
    clients = get_hyperliquid_clients()
    if not clients or not clients.get("exchange"):
        return {"ok": False, "error": "Exchange client not initialized"}
    
    if not HyperliquidConfig.orders_enabled():
        return {"ok": False, "error": "Orders not enabled. Set HYPERLIQUID_ENABLE_ORDERS=true"}
    
    exchange = clients["exchange"]
    
    # Determine direction
    direction = direction.lower()
    if direction not in ("long", "short"):
        return {"ok": False, "error": f"Invalid direction: {direction}. Must be 'long' or 'short'"}
    
    is_buy = direction == "long"
    
    # Get leverage
    leverage = HyperliquidConfig.get_leverage()
    
    # Get size - either from parameter or calculate from full account
    if size is None or size <= 0:
        if HyperliquidConfig.use_full_account():
            # Calculate size based on full account balance with leverage
            size = calculate_position_size(coin, leverage, account_fraction, price)
            if size is None or size <= 0:
                return {"ok": False, "error": "Could not calculate position size from account balance"}
            print(f"📊 Using full account: {size:.6f} {coin} ({leverage}x leverage)")
        else:
            size = HyperliquidConfig.get_default_size()
    
    size = round_size(size, coin)
    
    if size <= 0:
        return {"ok": False, "error": f"Invalid size: {size}"}
    
    # Get/calculate limit price
    if price is None:
        price = calculate_limit_price(coin, is_buy)
        if price is None:
            return {"ok": False, "error": "Could not determine limit price"}
    
    price = round_price(price, coin)
    
    # Check and set leverage
    leverage_result = check_and_set_leverage(coin)
    if leverage_result.get("error"):
        print(f"⚠️ Leverage warning: {leverage_result['error']}")
    
    # Get TIF setting
    tif = HyperliquidConfig.get_limit_tif()
    
    try:
        response = exchange.order(
            name=coin,
            is_buy=is_buy,
            sz=size,
            limit_px=price,
            order_type={"limit": {"tif": tif}},
            reduce_only=reduce_only
        )
        
        result = {
            "ok": True,
            "response": response,
            "order_details": {
                "coin": coin,
                "direction": direction,
                "size": size,
                "limit_price": price,
                "tif": tif,
                "reduce_only": reduce_only
            },
            "leverage": leverage_result
        }
        
        # Parse response for order ID and status
        if response:
            status = response.get("status")
            result["order_status"] = status
            
            if status == "ok":
                statuses = response.get("response", {}).get("data", {}).get("statuses", [])
                for s in statuses:
                    if "resting" in s:
                        result["order_id"] = s["resting"].get("oid")
                        result["filled"] = False
                    elif "filled" in s:
                        result["filled"] = True
                        result["fill_info"] = s["filled"]
        
        # Send Telegram notification for limit orders (not reduce-only/close orders)
        if not reduce_only and result.get("ok"):
            try:
                _send_limit_order_notification(result, leverage)
            except Exception as notif_err:
                print(f"⚠️ Failed to send order notification: {notif_err}")
        
        return result
        
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _send_limit_order_notification(order_result: Dict[str, Any], leverage: int):
    """Send Telegram notification when a limit order is placed"""
    try:
        from telegram_bot import get_bot
        
        bot = get_bot()
        if not bot.bot_token or not bot.chat_id:
            return
        
        details = order_result.get("order_details", {})
        coin = details.get("coin", "?")
        direction = details.get("direction", "?").upper()
        size = details.get("size", 0)
        limit_price = details.get("limit_price", 0)
        
        direction_emoji = "🟢" if direction == "LONG" else "🔴"
        
        # Calculate position value
        position_value = size * limit_price
        
        msg = f"<b>📝 LIMIT ORDER PLACED</b>\n\n"
        msg += f"{direction_emoji} <b>{coin}</b> {direction}\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"<b>Size:</b> {size:.6f} {coin}\n"
        msg += f"<b>Limit Price:</b> ${limit_price:,.2f}\n"
        msg += f"<b>Position Value:</b> ${position_value:,.2f}\n"
        msg += f"<b>Leverage:</b> {leverage}x\n"
        
        if order_result.get("filled"):
            msg += f"\n✅ <b>Filled immediately!</b>"
        elif order_result.get("order_id"):
            msg += f"\n⏳ <b>Resting</b> (OID: {order_result['order_id']})"
        
        bot.send_message(msg)
        
    except Exception as e:
        print(f"⚠️ Notification error: {e}")


# ============================================================================
# Take Profit & Stop Loss Orders
# ============================================================================

def place_take_profit_order(
    coin: str,
    direction: str,
    size: float,
    tp_price: float,
    reduce_only: bool = True
) -> Dict[str, Any]:
    """
    Place a take profit order (limit order at TP price).
    
    Args:
        coin: The coin (e.g., "BTC")
        direction: Original position direction ("long" or "short")
        size: Size to close at TP
        tp_price: Take profit price
        reduce_only: Should be True for TP orders
    
    Returns:
        Dict with order result
    """
    clients = get_hyperliquid_clients()
    if not clients or not clients.get("exchange"):
        return {"ok": False, "error": "Exchange client not initialized"}
    
    if not HyperliquidConfig.orders_enabled():
        return {"ok": False, "error": "Orders not enabled"}
    
    exchange = clients["exchange"]
    
    # For TP: if LONG, we SELL at higher price. If SHORT, we BUY at lower price.
    is_buy = direction.lower() == "short"
    
    tp_price = round_price(tp_price, coin)
    size = round_size(size, coin)
    
    tif = HyperliquidConfig.get_limit_tif()
    
    try:
        print(f"🎯 Placing TP order: {'BUY' if is_buy else 'SELL'} {size} {coin} @ ${tp_price:,.2f}")
        
        response = exchange.order(
            name=coin,
            is_buy=is_buy,
            sz=size,
            limit_px=tp_price,
            order_type={"limit": {"tif": tif}},
            reduce_only=reduce_only
        )
        
        result = {
            "ok": True,
            "response": response,
            "tp_details": {
                "coin": coin,
                "direction": "BUY" if is_buy else "SELL",
                "size": size,
                "tp_price": tp_price
            }
        }
        
        # Parse response
        if response and response.get("status") == "ok":
            statuses = response.get("response", {}).get("data", {}).get("statuses", [])
            for s in statuses:
                if "resting" in s:
                    result["order_id"] = s["resting"].get("oid")
                    print(f"✅ TP order placed (OID: {result['order_id']})")
                elif "error" in s:
                    result["ok"] = False
                    result["error"] = s.get("error")
                    print(f"❌ TP order error: {result['error']}")
        
        return result
        
    except Exception as e:
        print(f"❌ TP order exception: {e}")
        return {"ok": False, "error": str(e)}


def place_stop_loss_order(
    coin: str,
    direction: str,
    size: float,
    sl_price: float,
    reduce_only: bool = True
) -> Dict[str, Any]:
    """
    Place a stop loss order using trigger order.
    
    Note: Hyperliquid uses trigger orders for stop losses.
    
    Args:
        coin: The coin (e.g., "BTC")
        direction: Original position direction ("long" or "short")
        size: Size to close at SL
        sl_price: Stop loss trigger price
        reduce_only: Should be True for SL orders
    
    Returns:
        Dict with order result
    """
    clients = get_hyperliquid_clients()
    if not clients or not clients.get("exchange"):
        return {"ok": False, "error": "Exchange client not initialized"}
    
    if not HyperliquidConfig.orders_enabled():
        return {"ok": False, "error": "Orders not enabled"}
    
    exchange = clients["exchange"]
    
    # For SL: if LONG, we SELL when price drops. If SHORT, we BUY when price rises.
    is_buy = direction.lower() == "short"
    
    sl_price = round_price(sl_price, coin)
    size = round_size(size, coin)
    
    # For stop loss, use trigger order
    # trigger_px is the price at which the order activates
    # For LONG stop loss: trigger when price <= sl_price, then market sell
    # For SHORT stop loss: trigger when price >= sl_price, then market buy
    
    try:
        print(f"🛡️ Placing SL order: {'BUY' if is_buy else 'SELL'} {size} {coin} @ ${sl_price:,.2f} (trigger)")
        
        # Use order with trigger
        # order_type for stop: {"trigger": {"triggerPx": str, "isMarket": bool, "tpsl": "sl"}}
        response = exchange.order(
            name=coin,
            is_buy=is_buy,
            sz=size,
            limit_px=sl_price,  # For market trigger, this is ignored but required
            order_type={
                "trigger": {
                    "triggerPx": str(sl_price),
                    "isMarket": True,  # Execute as market order when triggered
                    "tpsl": "sl"  # Mark as stop loss
                }
            },
            reduce_only=reduce_only
        )
        
        result = {
            "ok": True,
            "response": response,
            "sl_details": {
                "coin": coin,
                "direction": "BUY" if is_buy else "SELL",
                "size": size,
                "sl_price": sl_price
            }
        }
        
        # Parse response
        if response and response.get("status") == "ok":
            statuses = response.get("response", {}).get("data", {}).get("statuses", [])
            for s in statuses:
                if "resting" in s:
                    result["order_id"] = s["resting"].get("oid")
                    print(f"✅ SL order placed (OID: {result['order_id']})")
                elif "error" in s:
                    result["ok"] = False
                    result["error"] = s.get("error")
                    print(f"❌ SL order error: {result['error']}")
        
        return result
        
    except Exception as e:
        print(f"❌ SL order exception: {e}")
        return {"ok": False, "error": str(e)}


def place_tp_sl_orders(
    coin: str,
    direction: str,
    size: float,
    stop_loss: Optional[float] = None,
    take_profits: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Place all TP and SL orders for a position.
    
    Args:
        coin: The coin
        direction: Position direction
        size: Position size
        stop_loss: Stop loss price
        take_profits: List of TP dicts with 'price' key
    
    Returns:
        Dict with results for each order
    """
    results = {
        "stop_loss": None,
        "take_profits": []
    }
    
    # Place stop loss
    if stop_loss:
        sl_result = place_stop_loss_order(coin, direction, size, stop_loss)
        results["stop_loss"] = sl_result
    
    # Place take profits
    if take_profits:
        # Split size across TPs or use full size for each
        # For simplicity, use full size for first TP (will close entire position)
        for i, tp in enumerate(take_profits):
            tp_price = tp.get("price") if isinstance(tp, dict) else tp
            if tp_price:
                tp_result = place_take_profit_order(coin, direction, size, tp_price)
                results["take_profits"].append({
                    "tp_num": i + 1,
                    "price": tp_price,
                    "result": tp_result
                })
                # Only place first TP for now (as it will close the position)
                break
    
    return results


def close_position_limit(
    coin: str,
    price: Optional[float] = None,
    size: Optional[float] = None
) -> Dict[str, Any]:
    """
    Close an existing position using a LIMIT order.
    
    Args:
        coin: The coin to close (e.g., "BTC")
        price: Limit price. If None, calculates based on current price
        size: Size to close. If None, closes entire position
    
    Returns:
        Dict with order result
    """
    clients = get_hyperliquid_clients()
    if not clients or not clients.get("exchange"):
        return {"ok": False, "error": "Exchange client not initialized"}
    
    if not HyperliquidConfig.orders_enabled():
        return {"ok": False, "error": "Orders not enabled. Set HYPERLIQUID_ENABLE_ORDERS=true"}
    
    # Get current position
    position = get_position(coin)
    if not position:
        return {"ok": False, "error": f"No open position for {coin}"}
    
    exchange = clients["exchange"]
    
    # Determine close direction (opposite of position)
    is_buy = position["direction"] == "SHORT"  # Short -> buy to close
    
    # Determine size to close
    close_size = position["size"] if size is None else min(size, position["size"])
    close_size = round_size(close_size, coin)
    
    if close_size <= 0:
        return {"ok": False, "error": f"Invalid close size: {close_size}"}
    
    # Calculate limit price with slippage for faster fill
    if price is None:
        current_price = get_hyperliquid_price(coin)
        if not current_price:
            return {"ok": False, "error": "Could not get current price"}
        
        slippage = 0.003  # 0.3% slippage for closes to ensure fill
        if is_buy:
            price = current_price * (1 + slippage)  # Buy slightly higher
        else:
            price = current_price * (1 - slippage)  # Sell slightly lower
    
    price = round_price(price, coin)
    
    # Get TIF setting
    tif = HyperliquidConfig.get_limit_tif()
    
    try:
        print(f"📤 Closing {coin}: {'BUY' if is_buy else 'SELL'} {close_size} @ ${price:,.2f} (reduce_only=True)")
        
        response = exchange.order(
            name=coin,
            is_buy=is_buy,
            sz=close_size,
            limit_px=price,
            order_type={"limit": {"tif": tif}},
            reduce_only=True
        )
        
        result = {
            "ok": True,
            "response": response,
            "close_details": {
                "coin": coin,
                "direction": "BUY" if is_buy else "SELL",
                "size": close_size,
                "limit_price": price,
                "original_direction": position["direction"],
                "entry_price": position["entry_price"],
                "unrealized_pnl": position["unrealized_pnl"]
            }
        }
        
        # Parse response for status
        if response:
            status = response.get("status")
            result["order_status"] = status
            
            if status == "ok":
                statuses = response.get("response", {}).get("data", {}).get("statuses", [])
                for s in statuses:
                    if "resting" in s:
                        result["order_id"] = s["resting"].get("oid")
                        result["filled"] = False
                        print(f"✅ Close order placed (resting): OID {result.get('order_id')}")
                    elif "filled" in s:
                        result["filled"] = True
                        result["fill_info"] = s["filled"]
                        print(f"✅ Close order filled immediately")
                    elif "error" in s:
                        result["ok"] = False
                        result["error"] = s.get("error", "Unknown order error")
                        print(f"❌ Close order error: {result['error']}")
            else:
                result["ok"] = False
                result["error"] = f"Order status: {status}"
                print(f"❌ Order failed with status: {status}")
        
        return result
        
    except Exception as e:
        print(f"❌ Close order exception: {e}")
        return {"ok": False, "error": str(e)}


def close_position_market(coin: str) -> Dict[str, Any]:
    """
    Close a position using market order (for immediate execution).
    
    Note: This uses higher fees but guarantees execution.
    Use close_position_limit for better fees when time permits.
    """
    clients = get_hyperliquid_clients()
    if not clients or not clients.get("exchange"):
        return {"ok": False, "error": "Exchange client not initialized"}
    
    if not HyperliquidConfig.orders_enabled():
        return {"ok": False, "error": "Orders not enabled"}
    
    position = get_position(coin)
    if not position:
        return {"ok": False, "error": f"No open position for {coin}"}
    
    exchange = clients["exchange"]
    is_buy = position["direction"] == "SHORT"
    size = position["size"]
    
    try:
        current_price = get_hyperliquid_price(coin)
        if not current_price:
            return {"ok": False, "error": "Could not get current price"}
        
        slippage = 0.01  # 1% for market orders
        
        print(f"🚨 Emergency close {coin}: {'BUY' if is_buy else 'SELL'} {size} @ market (slippage: {slippage*100}%)")
        
        response = exchange.market_open(
            name=coin,
            is_buy=is_buy,
            sz=size,
            px=current_price,
            slippage=slippage,
            reduce_only=True
        )
        
        result = {
            "ok": True,
            "response": response,
            "close_details": {
                "coin": coin,
                "direction": "BUY" if is_buy else "SELL",
                "size": size,
                "original_position": position["direction"],
                "method": "market"
            }
        }
        
        # Check response status
        if response:
            status = response.get("status")
            if status == "ok":
                statuses = response.get("response", {}).get("data", {}).get("statuses", [])
                for s in statuses:
                    if "filled" in s:
                        result["filled"] = True
                        result["fill_info"] = s["filled"]
                        print(f"✅ Emergency close filled")
                    elif "error" in s:
                        result["ok"] = False
                        result["error"] = s.get("error", "Unknown error")
                        print(f"❌ Emergency close error: {result['error']}")
            else:
                result["ok"] = False
                result["error"] = f"Order status: {status}"
        
        return result
        
    except Exception as e:
        print(f"❌ Emergency close exception: {e}")
        return {"ok": False, "error": str(e)}


# ============================================================================
# Trade Execution from Gate Result
# ============================================================================

def execute_trade_from_gate(
    symbol: str,
    gate_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a trade based on the LLM gate result.
    Uses LIMIT orders only.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT.P")
        gate_result: The gate decision result containing direction, execution params, etc.
    
    Returns:
        Dict with trade execution result
    """
    coin = convert_symbol_to_hyperliquid(symbol)
    
    direction = (gate_result.get("direction") or "").lower()
    if direction not in ("long", "short"):
        return {"ok": False, "error": f"Invalid direction in gate result: {direction}"}
    
    execution = gate_result.get("execution", {})
    entry_price = execution.get("entry_price")
    stop_loss = execution.get("stop_loss")
    take_profits = execution.get("take_profits", [])
    
    # Check price difference between MEXC and Hyperliquid
    price_diff = get_price_difference(coin)
    if price_diff.get("significant"):
        print(f"⚠️ Significant price difference detected:")
        print(f"   Hyperliquid (USDC): ${price_diff['hyperliquid_price']:,.2f}")
        print(f"   MEXC (USDT): ${price_diff['mexc_price']:,.2f}")
        print(f"   Difference: {price_diff['difference_pct']:.3f}%")
    
    # Execute the opening order
    result = open_position_limit(
        coin=coin,
        direction=direction,
        price=entry_price  # Use gate's entry price as limit
    )
    
    if result.get("ok"):
        result["symbol"] = symbol
        result["coin"] = coin
        result["gate_execution"] = execution
        result["price_comparison"] = price_diff
        
        # TODO: Set up stop loss and take profit orders if supported
        if stop_loss:
            result["stop_loss_pending"] = stop_loss
        if take_profits:
            result["take_profits_pending"] = take_profits
    
    return result


# ============================================================================
# Position Monitoring
# ============================================================================

class PositionMonitor:
    """
    Background monitor for open positions.
    Checks MEXC prices periodically to help manage trades.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._running = False
                    cls._instance._thread = None
                    cls._instance._check_interval = 60  # seconds
                    cls._instance._callbacks = []
                    cls._instance._tracked_positions = {}
        return cls._instance
    
    def add_callback(self, callback):
        """Add a callback function to be called on position updates"""
        self._callbacks.append(callback)
    
    def set_check_interval(self, seconds: int):
        """Set how often to check positions"""
        self._check_interval = max(10, seconds)  # Minimum 10 seconds
    
    def track_position(self, coin: str, entry_price: float, stop_loss: Optional[float] = None, 
                      take_profit: Optional[float] = None):
        """Start tracking a position for monitoring"""
        self._tracked_positions[coin] = {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "opened_at": datetime.now().isoformat()
        }
    
    def untrack_position(self, coin: str):
        """Stop tracking a position"""
        if coin in self._tracked_positions:
            del self._tracked_positions[coin]
    
    def _check_positions(self):
        """Internal method to check all positions"""
        positions = get_open_positions()
        
        for pos in positions:
            coin = pos["coin"]
            tracked = self._tracked_positions.get(coin, {})
            
            # Get both prices for comparison
            hl_price = get_hyperliquid_price(coin)
            mexc_price = get_mexc_price(f"{coin}USDT")
            
            update = {
                "position": pos,
                "hyperliquid_price": hl_price,
                "mexc_price": mexc_price,
                "tracked_info": tracked,
                "timestamp": datetime.now().isoformat()
            }
            
            # Check stop loss based on MEXC price (since analysis was on MEXC)
            if tracked.get("stop_loss") and mexc_price:
                if pos["direction"] == "LONG" and mexc_price <= tracked["stop_loss"]:
                    update["alert"] = "STOP_LOSS_HIT"
                elif pos["direction"] == "SHORT" and mexc_price >= tracked["stop_loss"]:
                    update["alert"] = "STOP_LOSS_HIT"
            
            # Check take profit
            if tracked.get("take_profit") and mexc_price:
                if pos["direction"] == "LONG" and mexc_price >= tracked["take_profit"]:
                    update["alert"] = "TAKE_PROFIT_HIT"
                elif pos["direction"] == "SHORT" and mexc_price <= tracked["take_profit"]:
                    update["alert"] = "TAKE_PROFIT_HIT"
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(update)
                except Exception as e:
                    print(f"⚠️ Position monitor callback error: {e}")
    
    def start(self):
        """Start the position monitor"""
        if self._running:
            return
        
        self._running = True
        
        def monitor_loop():
            print("✅ Position monitor started")
            while self._running:
                try:
                    self._check_positions()
                except Exception as e:
                    print(f"❌ Position monitor error: {e}")
                time.sleep(self._check_interval)
            print("🛑 Position monitor stopped")
        
        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the position monitor"""
        self._running = False


def get_position_monitor() -> PositionMonitor:
    """Get the singleton position monitor instance"""
    return PositionMonitor()


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_long(coin: str = "BTC", size: Optional[float] = None) -> Dict[str, Any]:
    """Quickly open a long position"""
    return open_position_limit(coin, "long", size)


def quick_short(coin: str = "BTC", size: Optional[float] = None) -> Dict[str, Any]:
    """Quickly open a short position"""
    return open_position_limit(coin, "short", size)


def quick_close(coin: str = "BTC") -> Dict[str, Any]:
    """Quickly close a position (limit order)"""
    return close_position_limit(coin)


def emergency_close(coin: str = "BTC") -> Dict[str, Any]:
    """Emergency close with market order"""
    return close_position_market(coin)


def status() -> Dict[str, Any]:
    """Get complete status including account, positions, and config"""
    return {
        "config": {
            "network": "TESTNET" if HyperliquidConfig.is_testnet() else "MAINNET",
            "orders_enabled": HyperliquidConfig.orders_enabled(),
            "default_size": HyperliquidConfig.get_default_size(),
            "leverage": HyperliquidConfig.get_leverage(),
            "max_leverage": HyperliquidConfig.get_max_leverage(),
            "slippage": HyperliquidConfig.get_slippage()
        },
        "account": get_account_info(),
        "positions": get_open_positions(),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Main - Test when run directly
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hyperliquid Trading Service - Status Check")
    print("=" * 60)
    
    s = status()
    
    print(f"\n📊 Configuration:")
    print(f"   Network: {s['config']['network']}")
    print(f"   Orders Enabled: {s['config']['orders_enabled']}")
    print(f"   Default Size: {s['config']['default_size']}")
    print(f"   Leverage: {s['config']['leverage']}x")
    
    if s['account']:
        print(f"\n💰 Account:")
        print(f"   Value: ${s['account']['account_value']:,.2f}")
        print(f"   Margin Used: ${s['account']['total_margin_used']:,.2f}")
        print(f"   Withdrawable: ${s['account']['withdrawable']:,.2f}")
    else:
        print(f"\n⚠️ Could not get account info")
    
    if s['positions']:
        print(f"\n📍 Open Positions:")
        for pos in s['positions']:
            pnl_emoji = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
            print(f"   {pos['coin']} {pos['direction']}: {pos['size']:.6f} @ ${pos['entry_price']:,.2f}")
            print(f"      {pnl_emoji} P&L: ${pos['unrealized_pnl']:.2f} | Leverage: {pos['leverage']}x")
    else:
        print(f"\n📍 No open positions")
    
    # Test price comparison
    print(f"\n📊 Price Comparison (BTC):")
    diff = get_price_difference("BTC")
    if diff['hyperliquid_price'] and diff['mexc_price']:
        print(f"   Hyperliquid (USDC): ${diff['hyperliquid_price']:,.2f}")
        print(f"   MEXC (USDT): ${diff['mexc_price']:,.2f}")
        print(f"   Difference: {diff['difference_pct']:.4f}%")
    else:
        print(f"   Could not fetch prices")

