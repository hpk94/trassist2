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
    
    @staticmethod
    def auto_sl_tp_enabled() -> bool:
        """Check if automatic SL/TP monitoring and execution is enabled"""
        return os.getenv("HYPERLIQUID_AUTO_SL_TP", "true").lower() in ("true", "1", "yes")
    
    @staticmethod
    def place_backup_sl_orders() -> bool:
        """Check if backup SL orders should be placed on Hyperliquid as safety net"""
        return os.getenv("HYPERLIQUID_BACKUP_SL_ORDERS", "false").lower() in ("true", "1", "yes")
    
    @staticmethod
    def get_sl_tp_check_interval() -> int:
        """Get interval (seconds) for SL/TP monitoring checks"""
        try:
            return int(os.getenv("HYPERLIQUID_SL_TP_CHECK_INTERVAL", "10"))
        except:
            return 10


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
        print(f"📝 Placing LIMIT order: {'BUY' if is_buy else 'SELL'} {size} {coin} @ ${price:,.2f}")
        
        response = exchange.order(
            name=coin,
            is_buy=is_buy,
            sz=size,
            limit_px=price,
            order_type={"limit": {"tif": tif}},
            reduce_only=reduce_only
        )
        
        result = {
            "ok": False,  # Will set to True only if order is confirmed
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
            print(f"   Response status: {status}")
            
            if status == "ok":
                statuses = response.get("response", {}).get("data", {}).get("statuses", [])
                print(f"   Order statuses: {statuses}")
                
                for s in statuses:
                    if "resting" in s:
                        result["order_id"] = s["resting"].get("oid")
                        result["filled"] = False
                        result["ok"] = True  # Confirmed: order is resting
                        print(f"   ✅ Order resting (OID: {result['order_id']})")
                    elif "filled" in s:
                        result["filled"] = True
                        result["fill_info"] = s["filled"]
                        result["ok"] = True  # Confirmed: order filled
                        print(f"   ✅ Order filled immediately")
                    elif "error" in s:
                        result["error"] = s.get("error", "Unknown order error")
                        print(f"   ❌ Order error: {result['error']}")
                
                # If we didn't find resting or filled status, check if there was no status at all
                if not result.get("ok") and not result.get("error"):
                    result["error"] = f"No order confirmation received. Statuses: {statuses}"
                    print(f"   ❌ No order confirmation in response")
            else:
                result["error"] = f"API returned status: {status}"
                print(f"   ❌ API error status: {status}")
        else:
            result["error"] = "No response received from exchange"
            print(f"   ❌ No response from exchange")
        
        # Send Telegram notification for limit orders (not reduce-only/close orders)
        if not reduce_only and result.get("ok"):
            try:
                _send_limit_order_notification(result, leverage)
            except Exception as notif_err:
                print(f"⚠️ Failed to send order notification: {notif_err}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Order exception: {e}")
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
                "unrealized_pnl": position["unrealized_pnl"],
                "leverage": position.get("leverage", 1)
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

def _adjust_price_for_hl(mexc_price: float, price_diff: Dict[str, Any]) -> float:
    """
    Adjust a MEXC price to equivalent Hyperliquid price.
    
    Uses the current price ratio to convert MEXC (USDT) prices to HL (USDC) prices.
    This accounts for USDT/USDC differences and exchange-specific spreads.
    """
    if not price_diff.get("mexc_price") or not price_diff.get("hyperliquid_price"):
        return mexc_price  # Can't adjust, return as-is
    
    # Calculate ratio: HL_price / MEXC_price
    ratio = price_diff["hyperliquid_price"] / price_diff["mexc_price"]
    
    # Apply ratio to the target price
    adjusted = mexc_price * ratio
    
    return adjusted


def execute_trade_from_gate(
    symbol: str,
    gate_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a trade based on the LLM gate result.
    Uses LIMIT orders only.
    
    Features:
    - Opens position with limit order
    - Sets up position monitoring for auto SL/TP based on MEXC prices
    - Optionally places backup SL order on Hyperliquid (price-adjusted)
    
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
    stop_loss = execution.get("stop_loss")  # MEXC price
    take_profits = execution.get("take_profits", [])
    
    # Get primary take profit
    take_profit = None
    if take_profits:
        take_profit = take_profits[0].get("price") if isinstance(take_profits[0], dict) else take_profits[0]
    
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
        
        # Get the actual position size from the order
        position_size = result.get("order_details", {}).get("size")
        actual_entry = result.get("order_details", {}).get("limit_price", entry_price)
        
        # =====================================================================
        # Set up position monitoring with auto SL/TP (monitors MEXC prices)
        # =====================================================================
        if HyperliquidConfig.auto_sl_tp_enabled():
            print(f"\n📊 Setting up SL/TP monitoring (based on MEXC prices)...")
            monitor = get_position_monitor()
            monitor.track_position(
                coin=coin,
                entry_price=entry_price,  # MEXC entry price
                direction=direction,
                stop_loss=stop_loss,  # MEXC SL price
                take_profit=take_profit,  # MEXC TP price  
                take_profits=take_profits,
                size=position_size
            )
            
            # Start monitor if not already running
            if not monitor.is_running():
                monitor.start()
            
            result["sl_tp_monitoring"] = {
                "enabled": True,
                "stop_loss_mexc": stop_loss,
                "take_profit_mexc": take_profit,
                "check_interval": HyperliquidConfig.get_sl_tp_check_interval()
            }
            print(f"   ✅ Monitoring active - checking every {HyperliquidConfig.get_sl_tp_check_interval()}s")
        
        # =====================================================================
        # Optionally place backup SL order on Hyperliquid (safety net)
        # =====================================================================
        if HyperliquidConfig.place_backup_sl_orders() and stop_loss and position_size:
            print(f"\n🛡️ Placing backup SL order on Hyperliquid...")
            
            # Adjust SL price for HL (account for USDT/USDC difference)
            adjusted_sl = _adjust_price_for_hl(stop_loss, price_diff)
            
            # Add small buffer for backup SL (1% worse than adjusted)
            # This ensures the MEXC-monitored SL triggers first
            if direction == "long":
                backup_sl = adjusted_sl * 0.99  # Slightly lower for longs
            else:
                backup_sl = adjusted_sl * 1.01  # Slightly higher for shorts
            
            print(f"   MEXC SL: ${stop_loss:,.2f}")
            print(f"   Adjusted HL SL: ${adjusted_sl:,.2f}")
            print(f"   Backup HL SL: ${backup_sl:,.2f} (safety buffer)")
            
            sl_result = place_stop_loss_order(
                coin=coin,
                direction=direction,
                size=position_size,
                sl_price=backup_sl
            )
            
            result["backup_sl_order"] = {
                "mexc_sl": stop_loss,
                "adjusted_sl": adjusted_sl,
                "backup_sl_price": backup_sl,
                "result": sl_result
            }
            
            if sl_result.get("ok"):
                print(f"   ✅ Backup SL order placed (OID: {sl_result.get('order_id')})")
            else:
                print(f"   ⚠️ Backup SL order failed: {sl_result.get('error')}")
        
        # Store SL/TP info in result
        result["sl_tp_config"] = {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "take_profits": take_profits,
            "monitoring_enabled": HyperliquidConfig.auto_sl_tp_enabled(),
            "backup_orders_enabled": HyperliquidConfig.place_backup_sl_orders()
        }
    
    return result


# ============================================================================
# Position Monitoring with Auto SL/TP Execution
# ============================================================================

class PositionMonitor:
    """
    Background monitor for open positions with automatic SL/TP execution.
    
    Key features:
    - Monitors MEXC prices (where analysis was done)
    - Auto-closes positions on Hyperliquid when MEXC prices hit SL/TP levels
    - Sends Telegram notifications on SL/TP triggers
    - Handles price differences between MEXC (USDT) and Hyperliquid (USDC)
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
                    cls._instance._check_interval = HyperliquidConfig.get_sl_tp_check_interval()
                    cls._instance._callbacks = []
                    cls._instance._tracked_positions = {}
                    cls._instance._auto_execute = HyperliquidConfig.auto_sl_tp_enabled()
                    cls._instance._processed_alerts = set()  # Prevent duplicate executions
        return cls._instance
    
    def add_callback(self, callback):
        """Add a callback function to be called on position updates"""
        self._callbacks.append(callback)
    
    def set_check_interval(self, seconds: int):
        """Set how often to check positions"""
        self._check_interval = max(5, seconds)  # Minimum 5 seconds for SL/TP
    
    def set_auto_execute(self, enabled: bool):
        """Enable or disable automatic SL/TP execution"""
        self._auto_execute = enabled
        print(f"{'✅' if enabled else '⏸️'} Auto SL/TP execution {'enabled' if enabled else 'disabled'}")
    
    def track_position(self, coin: str, entry_price: float, direction: str,
                       stop_loss: Optional[float] = None, 
                       take_profit: Optional[float] = None,
                       take_profits: Optional[List[Dict]] = None,
                       size: Optional[float] = None):
        """
        Start tracking a position for SL/TP monitoring.
        
        Args:
            coin: The coin (e.g., "BTC")
            entry_price: Entry price (from MEXC analysis)
            direction: "long" or "short"
            stop_loss: Stop loss price (MEXC price)
            take_profit: Primary take profit price (MEXC price)
            take_profits: List of TP dicts with 'price' key for multiple TPs
            size: Position size (for partial TP closes)
        """
        # Extract primary TP from take_profits list if not provided
        if take_profit is None and take_profits:
            take_profit = take_profits[0].get("price") if isinstance(take_profits[0], dict) else take_profits[0]
        
        self._tracked_positions[coin] = {
            "entry_price": entry_price,
            "direction": direction.upper(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "take_profits": take_profits or [],
            "size": size,
            "opened_at": datetime.now().isoformat(),
            "sl_triggered": False,
            "tp_triggered": False
        }
        
        print(f"📊 Tracking {coin} {direction.upper()}")
        print(f"   Entry: ${entry_price:,.2f} (MEXC)")
        if stop_loss:
            print(f"   SL: ${stop_loss:,.2f}")
        if take_profit:
            print(f"   TP: ${take_profit:,.2f}")
    
    def untrack_position(self, coin: str):
        """Stop tracking a position"""
        if coin in self._tracked_positions:
            del self._tracked_positions[coin]
            # Clear processed alerts for this coin
            self._processed_alerts = {a for a in self._processed_alerts if not a.startswith(coin)}
            print(f"🛑 Stopped tracking {coin}")
    
    def get_tracked_positions(self) -> Dict[str, Any]:
        """Get all tracked positions"""
        return self._tracked_positions.copy()
    
    def _send_sl_tp_notification(self, coin: str, trigger_type: str, mexc_price: float, 
                                  hl_price: float, close_result: Dict[str, Any]):
        """Send Telegram notification for SL/TP trigger"""
        try:
            from services.notification_service import send_telegram_message
            
            tracked = self._tracked_positions.get(coin, {})
            direction = tracked.get("direction", "UNKNOWN")
            entry = tracked.get("entry_price", 0)
            
            # Calculate PnL
            if direction == "LONG":
                pnl_pct = ((mexc_price - entry) / entry) * 100
            else:
                pnl_pct = ((entry - mexc_price) / entry) * 100
            
            emoji = "🛡️" if trigger_type == "STOP_LOSS" else "🎯"
            result_emoji = "✅" if close_result.get("ok") else "❌"
            
            message = f"""
{emoji} <b>{trigger_type.replace('_', ' ')} TRIGGERED</b>

<b>Coin:</b> {coin}
<b>Direction:</b> {direction}
<b>Entry:</b> ${entry:,.2f}
<b>MEXC Price:</b> ${mexc_price:,.2f}
<b>HL Price:</b> ${hl_price:,.2f}
<b>PnL:</b> {pnl_pct:+.2f}%

{result_emoji} <b>Close Result:</b> {'Success' if close_result.get('ok') else close_result.get('error', 'Failed')}
"""
            send_telegram_message(message)
            
        except Exception as e:
            print(f"⚠️ Failed to send SL/TP notification: {e}")
    
    def _execute_sl_tp_close(self, coin: str, trigger_type: str, mexc_price: float, 
                              hl_price: float) -> Dict[str, Any]:
        """Execute position close for SL/TP trigger"""
        alert_key = f"{coin}_{trigger_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Prevent duplicate execution within same minute
        if alert_key in self._processed_alerts:
            return {"ok": False, "error": "Already processed", "skipped": True}
        
        self._processed_alerts.add(alert_key)
        
        print(f"\n{'🛡️' if trigger_type == 'STOP_LOSS' else '🎯'} {trigger_type} triggered for {coin}!")
        print(f"   MEXC Price: ${mexc_price:,.2f}")
        print(f"   HL Price: ${hl_price:,.2f}")
        print(f"   Diff: {((hl_price - mexc_price) / mexc_price * 100):+.3f}%")
        
        if not self._auto_execute:
            print(f"   ⏸️ Auto-execute disabled - manual close required")
            return {"ok": False, "error": "Auto-execute disabled", "manual_required": True}
        
        if not HyperliquidConfig.orders_enabled():
            print(f"   ⚠️ Orders disabled - cannot close position")
            return {"ok": False, "error": "Orders disabled"}
        
        # Use market order for immediate execution on SL/TP
        print(f"   🚀 Executing market close on Hyperliquid...")
        result = close_position_market(coin)
        
        if result.get("ok"):
            print(f"   ✅ Position closed successfully")
            # Mark as triggered to prevent further checks
            if coin in self._tracked_positions:
                self._tracked_positions[coin][f"{trigger_type.lower()}_triggered"] = True
        else:
            print(f"   ❌ Close failed: {result.get('error')}")
        
        # Send notification
        self._send_sl_tp_notification(coin, trigger_type, mexc_price, hl_price, result)
        
        return result
    
    def _check_positions(self):
        """Internal method to check all positions for SL/TP triggers"""
        positions = get_open_positions()
        open_coins = {pos["coin"] for pos in positions}
        
        # Clean up tracked positions that no longer exist
        closed_coins = set(self._tracked_positions.keys()) - open_coins
        for coin in closed_coins:
            print(f"📤 Position {coin} closed externally, removing from tracking")
            self.untrack_position(coin)
        
        for pos in positions:
            coin = pos["coin"]
            tracked = self._tracked_positions.get(coin, {})
            
            # Skip if not tracked or already triggered
            if not tracked:
                continue
            if tracked.get("sl_triggered") or tracked.get("tp_triggered"):
                continue
            
            # Get both prices for comparison
            hl_price = get_hyperliquid_price(coin)
            mexc_price = get_mexc_price(f"{coin}USDT")
            
            if not mexc_price or not hl_price:
                continue
            
            direction = tracked.get("direction", pos.get("direction", "UNKNOWN"))
            stop_loss = tracked.get("stop_loss")
            take_profit = tracked.get("take_profit")
            
            update = {
                "position": pos,
                "hyperliquid_price": hl_price,
                "mexc_price": mexc_price,
                "tracked_info": tracked,
                "timestamp": datetime.now().isoformat(),
                "alert": None
            }
            
            # Check stop loss based on MEXC price (since analysis was on MEXC)
            if stop_loss:
                sl_hit = False
                if direction == "LONG" and mexc_price <= stop_loss:
                    sl_hit = True
                elif direction == "SHORT" and mexc_price >= stop_loss:
                    sl_hit = True
                
                if sl_hit:
                    update["alert"] = "STOP_LOSS_HIT"
                    close_result = self._execute_sl_tp_close(coin, "STOP_LOSS", mexc_price, hl_price)
                    update["close_result"] = close_result
                    if close_result.get("ok"):
                        self.untrack_position(coin)
                        continue
            
            # Check take profit based on MEXC price
            if take_profit and not update.get("alert"):
                tp_hit = False
                if direction == "LONG" and mexc_price >= take_profit:
                    tp_hit = True
                elif direction == "SHORT" and mexc_price <= take_profit:
                    tp_hit = True
                
                if tp_hit:
                    update["alert"] = "TAKE_PROFIT_HIT"
                    close_result = self._execute_sl_tp_close(coin, "TAKE_PROFIT", mexc_price, hl_price)
                    update["close_result"] = close_result
                    if close_result.get("ok"):
                        self.untrack_position(coin)
                        continue
            
            # Notify callbacks (for external monitoring/logging)
            for callback in self._callbacks:
                try:
                    callback(update)
                except Exception as e:
                    print(f"⚠️ Position monitor callback error: {e}")
    
    def start(self):
        """Start the position monitor"""
        if self._running:
            print("⚠️ Position monitor already running")
            return
        
        self._running = True
        
        def monitor_loop():
            print(f"✅ Position monitor started (interval: {self._check_interval}s, auto-execute: {self._auto_execute})")
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
        print("🛑 Stopping position monitor...")
    
    def is_running(self) -> bool:
        """Check if monitor is running"""
        return self._running


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

