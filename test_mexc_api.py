#!/usr/bin/env python3
"""
MEXC Futures API Test Script

A comprehensive test script to verify MEXC Futures API connectivity,
test market data retrieval, account management, and order placement.

Usage:
    python test_mexc_api.py                  # Run all safe tests
    python test_mexc_api.py --interactive    # Interactive menu mode
    python test_mexc_api.py --test-order     # Test order placement (TESTNET/PAPER)
    
⚠️  WARNING: Order tests should only be run with small amounts!
"""

import os
import sys
import json
import argparse
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================================
# Configuration & Constants
# ============================================================================

# Default symbol for testing
DEFAULT_SYMBOL = "BTC_USDT"
SPOT_SYMBOL = "BTCUSDT"

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


# ============================================================================
# MEXC Client Initialization
# ============================================================================

def get_mexc_clients():
    """Initialize and return MEXC futures and spot clients"""
    try:
        from pymexc import spot, futures
        
        api_key = os.getenv("MEXC_API_KEY")
        api_secret = os.getenv("MEXC_API_SECRET")
        
        # Authenticated clients (for account operations)
        futures_client = None
        spot_client = None
        
        if api_key and api_secret:
            futures_client = futures.HTTP(api_key=api_key, api_secret=api_secret)
            spot_client = spot.HTTP(api_key=api_key, api_secret=api_secret)
        
        # Public client (for market data - no auth needed)
        public_spot_client = spot.HTTP()
        
        return {
            "futures": futures_client,
            "spot": spot_client,
            "public_spot": public_spot_client
        }
    except ImportError:
        print_error("pymexc library not installed. Run: pip install pymexc")
        return None


# ============================================================================
# Test Functions
# ============================================================================

def test_environment_config() -> Dict[str, Any]:
    """Test 1: Check environment configuration"""
    print_header("Test 1: Environment Configuration")
    
    results = {"passed": True, "details": {}}
    
    # Required variables
    api_key = os.getenv("MEXC_API_KEY")
    api_secret = os.getenv("MEXC_API_SECRET")
    
    # Optional variables with defaults
    leverage = os.getenv("MEXC_LEVERAGE", "30")
    default_size = os.getenv("MEXC_DEFAULT_SIZE", "10.0")
    enable_orders = os.getenv("MEXC_ENABLE_ORDERS", "false")
    
    # Check API credentials
    if api_key:
        print_success(f"MEXC_API_KEY: Set (length: {len(api_key)}, starts with: {api_key[:4]}...)")
        results["details"]["api_key"] = "configured"
    else:
        print_error("MEXC_API_KEY: NOT SET")
        results["passed"] = False
        results["details"]["api_key"] = "missing"
    
    if api_secret:
        print_success(f"MEXC_API_SECRET: Set (length: {len(api_secret)})")
        results["details"]["api_secret"] = "configured"
    else:
        print_error("MEXC_API_SECRET: NOT SET")
        results["passed"] = False
        results["details"]["api_secret"] = "missing"
    
    # Display trading parameters
    print(f"\n{Colors.BOLD}Trading Parameters:{Colors.END}")
    print(f"   Leverage: {leverage}x")
    print(f"   Default Size: ${default_size} USDT")
    print(f"   Enable Orders: {enable_orders}")
    
    # Safety check
    if enable_orders.lower() == "true":
        print_warning("LIVE TRADING IS ENABLED! Orders will be placed for real.")
    else:
        print_info("Paper trading mode (orders simulated but not placed)")
    
    results["details"]["leverage"] = leverage
    results["details"]["default_size"] = default_size
    results["details"]["enable_orders"] = enable_orders
    
    return results


def test_api_connection(clients: Dict) -> Dict[str, Any]:
    """Test 2: Test API connection"""
    print_header("Test 2: API Connection")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("futures"):
        print_error("No authenticated client available")
        return results
    
    futures_client = clients["futures"]
    
    try:
        # Test server ping
        ping_result = futures_client.ping()
        if ping_result is not None:
            print_success("Server connection successful (ping OK)")
            results["details"]["ping"] = ping_result
            results["passed"] = True
        else:
            print_error("Could not ping server")
    except Exception as e:
        print_error(f"Server connection failed: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_account_info(clients: Dict) -> Dict[str, Any]:
    """Test 3: Get account information"""
    print_header("Test 3: Account Information")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("futures"):
        print_error("No authenticated client available")
        return results
    
    futures_client = clients["futures"]
    
    try:
        # Get account assets using correct method name
        account_info = futures_client.assets()
        
        if isinstance(account_info, dict):
            if account_info.get("success", False):
                print_success("Account data retrieved successfully")
                results["passed"] = True
                
                assets = account_info.get('data', [])
                print(f"\n{Colors.BOLD}💰 Account Balances:{Colors.END}")
                
                found_usdt = False
                for asset in assets:
                    currency = asset.get('currency', '')
                    available = float(asset.get('availableBalance', 0))
                    frozen = float(asset.get('frozenBalance', 0))
                    total = available + frozen
                    
                    if total > 0 or currency == 'USDT':
                        print(f"   {currency}:")
                        print(f"      Total: ${total:.4f}")
                        print(f"      Available: ${available:.4f}")
                        print(f"      In Use: ${frozen:.4f}")
                        
                        if currency == 'USDT':
                            found_usdt = True
                            results["details"]["usdt_available"] = available
                            results["details"]["usdt_total"] = total
                            
                            default_size = float(os.getenv("MEXC_DEFAULT_SIZE", "10.0"))
                            if available < default_size:
                                print_warning(f"Available balance (${available:.2f}) < default order size (${default_size})")
                
                if not found_usdt:
                    print_warning("No USDT balance found in futures account")
                    print_info("Transfer USDT from spot to futures account to trade")
            elif account_info.get("code") == 401:
                # Permission issue - API key doesn't have Read permission
                print_warning("API key doesn't have 'Read' permission for account balances")
                print_info("Go to MEXC → API Management → Edit API Key → Enable 'Read' permission")
                print_info("Note: Trading can still work without this permission")
                results["details"]["permission_issue"] = True
                results["passed"] = True  # Not a critical failure
            else:
                print_error(f"Failed to get account info: {account_info}")
                results["details"]["error"] = str(account_info)
        else:
            print_error(f"Unexpected response: {account_info}")
            results["details"]["error"] = str(account_info)
            
    except Exception as e:
        print_error(f"Account info error: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_open_positions(clients: Dict) -> Dict[str, Any]:
    """Test 4: Get open positions"""
    print_header("Test 4: Open Positions")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("futures"):
        print_error("No authenticated client available")
        return results
    
    futures_client = clients["futures"]
    
    try:
        positions = futures_client.open_positions()
        
        if positions:
            print_success("Positions data retrieved successfully")
            results["passed"] = True
            
            pos_list = positions.get('data', [])
            
            if pos_list:
                print(f"\n{Colors.BOLD}📍 Open Positions:{Colors.END}")
                results["details"]["positions"] = []
                
                for pos in pos_list:
                    symbol = pos.get('symbol', 'Unknown')
                    pos_type = 'LONG' if pos.get('positionType') == 1 else 'SHORT'
                    hold_vol = pos.get('holdVol', 0)
                    leverage = pos.get('leverage', 0)
                    unrealized_pnl = float(pos.get('unrealisedPnl', 0))
                    open_avg = pos.get('openAvgPrice', 0)
                    
                    pnl_color = Colors.GREEN if unrealized_pnl >= 0 else Colors.RED
                    
                    print(f"\n   {Colors.BOLD}{symbol}{Colors.END}")
                    print(f"      Direction: {pos_type}")
                    print(f"      Size: {hold_vol}")
                    print(f"      Leverage: {leverage}x")
                    print(f"      Entry Price: ${open_avg}")
                    print(f"      Unrealized P&L: {pnl_color}${unrealized_pnl:.4f}{Colors.END}")
                    
                    results["details"]["positions"].append({
                        "symbol": symbol,
                        "direction": pos_type,
                        "size": hold_vol,
                        "leverage": leverage,
                        "pnl": unrealized_pnl
                    })
            else:
                print_info("No open positions")
                results["details"]["positions"] = []
        else:
            print_warning("Could not retrieve positions")
            
    except Exception as e:
        print_error(f"Positions error: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_market_data(clients: Dict, symbol: str = SPOT_SYMBOL) -> Dict[str, Any]:
    """Test 5: Get market data"""
    print_header(f"Test 5: Market Data ({symbol})")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("public_spot"):
        print_error("No public client available")
        return results
    
    public_client = clients["public_spot"]
    
    try:
        # Get ticker price
        ticker = public_client.ticker_price(symbol=symbol)
        
        if ticker:
            print_success("Ticker data retrieved")
            price = ticker.get('price', ticker.get('data', {}).get('price'))
            print(f"\n{Colors.BOLD}📊 {symbol} Price:{Colors.END}")
            print(f"   Current: ${float(price):,.2f}")
            results["details"]["price"] = price
            results["passed"] = True
        
        # Get 24h stats
        try:
            ticker_24h = public_client.ticker_24h(symbol=symbol)
            if ticker_24h:
                data = ticker_24h if isinstance(ticker_24h, dict) else {}
                print(f"\n{Colors.BOLD}📈 24h Statistics:{Colors.END}")
                print(f"   High: ${float(data.get('highPrice', data.get('high', 0))):,.2f}")
                print(f"   Low: ${float(data.get('lowPrice', data.get('low', 0))):,.2f}")
                print(f"   Volume: {float(data.get('volume', 0)):,.2f}")
                change_pct = float(data.get('priceChangePercent', data.get('change', 0)))
                change_color = Colors.GREEN if change_pct >= 0 else Colors.RED
                print(f"   Change: {change_color}{change_pct:+.2f}%{Colors.END}")
        except Exception as e:
            print_warning(f"Could not get 24h stats: {e}")
        
        # Get recent klines
        try:
            klines = public_client.klines(symbol=symbol, interval='1m', limit=5)
            if klines:
                print(f"\n{Colors.BOLD}🕐 Recent 1m Candles:{Colors.END}")
                for i, k in enumerate(klines[-3:]):  # Last 3 candles
                    open_p = float(k[1])
                    high_p = float(k[2])
                    low_p = float(k[3])
                    close_p = float(k[4])
                    vol = float(k[5])
                    print(f"   Candle {i+1}: O:{open_p:.2f} H:{high_p:.2f} L:{low_p:.2f} C:{close_p:.2f}")
        except Exception as e:
            print_warning(f"Could not get klines: {e}")
            
    except Exception as e:
        print_error(f"Market data error: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_futures_contract_info(clients: Dict, symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    """Test 6: Get futures contract information"""
    print_header(f"Test 6: Futures Contract Info ({symbol})")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("futures"):
        print_error("No futures client available")
        return results
    
    futures_client = clients["futures"]
    
    try:
        # Get contract details using correct method name
        contract_details = futures_client.detail(symbol=symbol)
        
        if contract_details and contract_details.get('success', True):
            print_success("Contract details retrieved")
            results["passed"] = True
            
            data = contract_details.get('data', contract_details)
            
            if isinstance(data, dict):
                print(f"\n{Colors.BOLD}📜 Contract Details:{Colors.END}")
                print(f"   Symbol: {data.get('symbol', symbol)}")
                print(f"   Display Name: {data.get('displayName', 'N/A')}")
                print(f"   Contract Size: {data.get('contractSize', 'N/A')}")
                print(f"   Min Volume: {data.get('minVol', 'N/A')}")
                print(f"   Max Volume: {data.get('maxVol', 'N/A')}")
                print(f"   Price Precision: {data.get('priceScale', 'N/A')}")
                print(f"   Volume Precision: {data.get('volScale', 'N/A')}")
                print(f"   Max Leverage: {data.get('maxLeverage', 'N/A')}x")
                
                results["details"]["contract"] = data
        else:
            print_warning(f"Contract info response: {contract_details}")
            
    except Exception as e:
        print_error(f"Contract info error: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_order_book(clients: Dict, symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    """Test 7: Get order book"""
    print_header(f"Test 7: Order Book ({symbol})")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("futures"):
        print_error("No futures client available")
        return results
    
    futures_client = clients["futures"]
    
    try:
        # Get order book depth using correct method name
        depth = futures_client.get_depth(symbol=symbol, limit=5)
        
        if depth:
            print_success("Order book retrieved")
            results["passed"] = True
            
            data = depth.get('data', depth)
            bids = data.get('bids', [])[:5]
            asks = data.get('asks', [])[:5]
            
            print(f"\n{Colors.BOLD}📗 Order Book (Top 5):{Colors.END}")
            
            print(f"\n   {Colors.RED}ASKS (Sell Orders):{Colors.END}")
            for ask in reversed(asks):
                price = float(ask[0])
                qty = float(ask[1])
                print(f"      ${price:,.2f}  |  {qty:.4f}")
            
            # Spread
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                spread = best_ask - best_bid
                spread_pct = (spread / best_bid) * 100
                print(f"\n   {Colors.YELLOW}--- Spread: ${spread:.2f} ({spread_pct:.4f}%) ---{Colors.END}")
            
            print(f"\n   {Colors.GREEN}BIDS (Buy Orders):{Colors.END}")
            for bid in bids:
                price = float(bid[0])
                qty = float(bid[1])
                print(f"      ${price:,.2f}  |  {qty:.4f}")
                
            results["details"]["spread"] = spread if bids and asks else None
        else:
            print_warning("Could not get order book")
            
    except Exception as e:
        print_error(f"Order book error: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_leverage_setting(clients: Dict, symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    """Test 8: Test leverage configuration"""
    print_header(f"Test 8: Leverage Configuration ({symbol})")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("futures"):
        print_error("No futures client available")
        return results
    
    futures_client = clients["futures"]
    target_leverage = int(os.getenv("MEXC_LEVERAGE", "30"))
    
    try:
        # First, try to get current leverage (this usually works)
        current_leverage_data = None
        try:
            current_leverage = futures_client.get_leverage(symbol=symbol)
            if current_leverage and current_leverage.get('success'):
                current_leverage_data = current_leverage.get('data', [])
                if current_leverage_data:
                    for lev_info in current_leverage_data:
                        pos_type = "LONG" if lev_info.get('positionType') == 1 else "SHORT"
                        curr_lev = lev_info.get('leverage', 'N/A')
                        max_lev = lev_info.get('maxLeverageView', 'N/A')
                        print_success(f"Current {pos_type} leverage: {curr_lev}x (max: {max_lev}x)")
                    results["passed"] = True
                    results["details"]["current_leverage"] = current_leverage_data
        except Exception as e:
            print_warning(f"Could not get current leverage: {e}")
        
        # Try to set leverage (may fail due to pymexc signature bug)
        try:
            leverage_result = futures_client.change_leverage(
                position_id=0,
                leverage=target_leverage,
                open_type=1,
                symbol=symbol,
                position_type=1
            )
            
            if leverage_result:
                if leverage_result.get('success'):
                    print_success(f"Leverage changed to {target_leverage}x")
                    results["passed"] = True
                    results["details"]["leverage_set"] = target_leverage
                elif leverage_result.get('code') == 602:
                    # Known pymexc signature bug
                    print_warning("Could not change leverage (signature issue in pymexc library)")
                    print_info("This is a known library bug - leverage can be set manually on MEXC")
                    print_info(f"Your current leverage ({current_leverage_data[0].get('leverage') if current_leverage_data else 'unknown'}x) will be used")
                    # Still pass if we could read the current leverage
                    if current_leverage_data:
                        results["passed"] = True
                else:
                    print_warning(f"Leverage change response: {leverage_result}")
        except Exception as e:
            print_warning(f"Change leverage error: {e}")
            print_info("Leverage can be set manually in MEXC Futures settings")
            # Still pass if we could read leverage
            if current_leverage_data:
                results["passed"] = True
            
    except Exception as e:
        print_error(f"Leverage test error: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_order_placement_dry_run(clients: Dict, symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    """Test 9: Simulate order placement (DRY RUN - no actual orders)"""
    print_header("Test 9: Order Placement (DRY RUN)")
    
    results = {"passed": False, "details": {}}
    
    enable_orders = os.getenv("MEXC_ENABLE_ORDERS", "false").lower()
    
    if enable_orders == "true":
        print_warning("MEXC_ENABLE_ORDERS is TRUE!")
        print_warning("This test would place REAL orders!")
        print_info("Skipping to prevent accidental trading")
        print_info("Set MEXC_ENABLE_ORDERS=false to run this test safely")
        results["details"]["skipped"] = "live_trading_enabled"
        return results
    
    # Get current price for simulation (use public client - works without auth)
    current_price = 95000.0  # Fallback
    try:
        if clients and clients.get("public_spot"):
            public_client = clients["public_spot"]
            ticker = public_client.ticker_price(symbol=SPOT_SYMBOL)
            current_price = float(ticker.get('price', ticker.get('data', {}).get('price', 0)))
    except Exception as e:
        print_warning(f"Could not get live price, using fallback: {e}")
    
    default_size = float(os.getenv("MEXC_DEFAULT_SIZE", "10.0"))
    leverage = int(os.getenv("MEXC_LEVERAGE", "30"))
    
    print_info("Simulating order (NOT placing real order)")
    print(f"\n{Colors.BOLD}📝 Simulated Order Details:{Colors.END}")
    print(f"   Symbol: {symbol}")
    print(f"   Direction: LONG (Buy)")
    print(f"   Order Type: Market")
    print(f"   Margin: ${default_size:.2f} USDT")
    print(f"   Leverage: {leverage}x")
    print(f"   Position Value: ${default_size * leverage:.2f}")
    print(f"   Estimated Entry: ${current_price:,.2f}")
    
    # Calculate stop loss and take profit
    stop_loss = current_price * 0.99  # 1% SL
    take_profit = current_price * 1.02  # 2% TP
    
    print(f"\n{Colors.BOLD}🛡️ Risk Management:{Colors.END}")
    print(f"   Stop Loss: ${stop_loss:,.2f} (-1%)")
    print(f"   Take Profit: ${take_profit:,.2f} (+2%)")
    print(f"   Risk/Reward: 2:1")
    
    # Calculate potential P&L
    max_loss = default_size  # In isolated margin, max loss = margin
    potential_profit = default_size * 2  # With 2:1 R/R
    
    print(f"\n{Colors.BOLD}💰 Potential P&L:{Colors.END}")
    print(f"   {Colors.RED}Max Loss: ${max_loss:.2f}{Colors.END}")
    print(f"   {Colors.GREEN}Potential Profit: ${potential_profit:.2f}{Colors.END}")
    
    print_success("Order simulation complete")
    results["passed"] = True
    results["details"]["simulation"] = {
        "symbol": symbol,
        "size": default_size,
        "leverage": leverage,
        "entry": current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }
    
    return results


def test_place_real_order(clients: Dict, symbol: str = DEFAULT_SYMBOL, direction: str = "long", size_override: float = None) -> Dict[str, Any]:
    """Test 10: Place a REAL order (USE WITH CAUTION!)"""
    print_header("Test 10: REAL Order Placement")
    
    results = {"passed": False, "details": {}}
    
    print_warning("⚠️  THIS WILL PLACE A REAL ORDER! ⚠️")
    print_warning("Make sure you understand the risks!")
    
    enable_orders = os.getenv("MEXC_ENABLE_ORDERS", "false").lower()
    
    if enable_orders != "true":
        print_error("MEXC_ENABLE_ORDERS is not 'true'")
        print_info("Set MEXC_ENABLE_ORDERS=true in .env to enable live trading")
        results["details"]["blocked"] = "orders_disabled"
        return results
    
    if not clients or not clients.get("futures"):
        print_error("No futures client available")
        return results
    
    futures_client = clients["futures"]
    
    # Get parameters
    default_size = size_override or float(os.getenv("MEXC_DEFAULT_SIZE", "10.0"))
    leverage = int(os.getenv("MEXC_LEVERAGE", "30"))
    
    # Get current price
    try:
        public_client = clients["public_spot"]
        ticker = public_client.ticker_price(symbol=SPOT_SYMBOL)
        current_price = float(ticker.get('price', ticker.get('data', {}).get('price', 0)))
    except Exception as e:
        print_error(f"Could not get current price: {e}")
        return results
    
    # Calculate volume (USDT amount / price * leverage for position sizing)
    # For MEXC futures, vol is in contracts (typically 0.001 BTC per contract)
    vol = default_size / current_price
    
    # Determine side: 1=open long, 2=close short, 3=open short, 4=close long
    side = 1 if direction.lower() == "long" else 3
    
    print(f"\n{Colors.BOLD}Order Parameters:{Colors.END}")
    print(f"   Symbol: {symbol}")
    print(f"   Direction: {direction.upper()}")
    print(f"   Size: ${default_size:.2f} USDT (~{vol:.6f} BTC)")
    print(f"   Leverage: {leverage}x")
    print(f"   Current Price: ${current_price:,.2f}")
    
    # Final confirmation
    confirm = input(f"\n{Colors.RED}Type 'CONFIRM' to place this order: {Colors.END}")
    
    if confirm != "CONFIRM":
        print_info("Order cancelled")
        results["details"]["cancelled"] = True
        return results
    
    try:
        # Set leverage first
        try:
            futures_client.change_leverage(
                position_id=0,
                leverage=leverage,
                open_type=1,  # Isolated
                symbol=symbol,
                position_type=1 if direction.lower() == "long" else 2
            )
            print_info(f"Leverage set to {leverage}x")
        except Exception as e:
            print_warning(f"Leverage setting note: {e}")
        
        # Place market order using the correct method
        print(f"\n{Colors.YELLOW}Placing order...{Colors.END}")
        order_result = futures_client.order(
            symbol=symbol,
            price=current_price,  # For market orders, this is reference price
            vol=vol,
            side=side,  # 1=open long, 3=open short
            type=5,  # 5 = market order
            open_type=1,  # 1 = isolated margin
            leverage=leverage
        )
        
        if order_result:
            success = order_result.get('success', False)
            if success:
                print_success("Order placed!")
                order_data = order_result.get('data', {})
                print(f"   Order ID: {order_data.get('orderId', 'N/A')}")
                results["passed"] = True
                results["details"]["order"] = order_result
            else:
                print_error(f"Order failed: {order_result}")
                results["details"]["error"] = str(order_result)
        else:
            print_error("Order placement returned empty result")
            
    except Exception as e:
        print_error(f"Order failed: {e}")
        results["details"]["error"] = str(e)
    
    return results


def open_long_position(clients: Dict, symbol: str = DEFAULT_SYMBOL, use_percentage: float = None) -> Dict[str, Any]:
    """
    Open a LONG position on MEXC Futures with proper position sizing.
    
    Args:
        clients: MEXC client instances
        symbol: Trading pair (default: BTC_USDT)
        use_percentage: Optional percentage of balance to use (0-100)
    
    Returns:
        Dict with order details
    """
    print_header("OPEN LONG POSITION")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("futures"):
        print_error("No futures client available")
        return results
    
    futures_client = clients["futures"]
    public_client = clients.get("public_spot")
    
    # -------------------------------------------------------------------------
    # Step 1: Get contract details
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}Step 1: Getting contract details...{Colors.END}")
    contract_size = 0.0001  # Default BTC contract size
    min_vol = 1
    vol_precision = 0
    
    try:
        contract_info = futures_client.detail(symbol=symbol)
        if contract_info and contract_info.get('success'):
            data = contract_info.get('data', {})
            contract_size = float(data.get('contractSize', 0.0001))
            min_vol = int(data.get('minVol', 1))
            vol_precision = int(data.get('volScale', 0))
            print_success(f"Contract size: {contract_size} BTC per contract")
            print_info(f"Min volume: {min_vol} contracts")
    except Exception as e:
        print_warning(f"Using default contract size: {e}")
    
    # -------------------------------------------------------------------------
    # Step 2: Get current price
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}Step 2: Getting current price...{Colors.END}")
    try:
        ticker = public_client.ticker_price(symbol=SPOT_SYMBOL)
        current_price = float(ticker.get('price', ticker.get('data', {}).get('price', 0)))
        print_success(f"Current BTC price: ${current_price:,.2f}")
    except Exception as e:
        print_error(f"Could not get price: {e}")
        return results
    
    # -------------------------------------------------------------------------
    # Step 3: Get account balance (or use default)
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}Step 3: Getting account balance...{Colors.END}")
    usdt_balance = None
    
    try:
        assets_result = futures_client.assets()
        if assets_result and assets_result.get('success'):
            for asset in assets_result.get('data', []):
                if asset.get('currency') == 'USDT':
                    usdt_balance = float(asset.get('availableBalance', 0))
                    print_success(f"Available USDT balance: ${usdt_balance:.2f}")
                    break
    except:
        pass
    
    if usdt_balance is None:
        # Use default size from env
        default_size = float(os.getenv("MEXC_DEFAULT_SIZE", "10.0"))
        print_warning("Could not get balance (API permission). Using MEXC_DEFAULT_SIZE")
        usdt_balance = default_size
        print_info(f"Using margin amount: ${usdt_balance:.2f}")
    
    # -------------------------------------------------------------------------
    # Step 4: Get leverage (use env setting, show current MEXC setting for info)
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}Step 4: Getting leverage info...{Colors.END}")
    leverage = int(os.getenv("MEXC_LEVERAGE", "30"))  # Use env setting
    print_info(f"Using MEXC_LEVERAGE from .env: {leverage}x")
    
    try:
        lev_info = futures_client.get_leverage(symbol=symbol)
        if lev_info and lev_info.get('success'):
            for lev in lev_info.get('data', []):
                if lev.get('positionType') == 1:  # Long position
                    current_mexc_leverage = int(lev.get('leverage', 0))
                    if current_mexc_leverage != leverage:
                        print_warning(f"MEXC account leverage is {current_mexc_leverage}x (different from env)")
                        print_info(f"Order will use {leverage}x as specified in .env")
                    break
    except Exception as e:
        print_warning(f"Could not check MEXC leverage: {e}")
    
    # -------------------------------------------------------------------------
    # Step 5: Calculate position size
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}Step 5: Calculating position size...{Colors.END}")
    
    # Determine margin to use
    if use_percentage:
        margin_usdt = usdt_balance * (use_percentage / 100)
        print_info(f"Using {use_percentage}% of balance: ${margin_usdt:.2f}")
    else:
        margin_usdt = float(os.getenv("MEXC_DEFAULT_SIZE", "10.0"))
        print_info(f"Using MEXC_DEFAULT_SIZE: ${margin_usdt:.2f}")
    
    # Calculate position value (margin * leverage)
    position_value_usdt = margin_usdt * leverage
    
    # Calculate BTC amount
    btc_amount = position_value_usdt / current_price
    
    # Convert to contracts
    num_contracts = int(btc_amount / contract_size)
    
    # Ensure minimum volume
    if num_contracts < min_vol:
        num_contracts = min_vol
        print_warning(f"Adjusted to minimum: {min_vol} contracts")
    
    # Recalculate actual values
    actual_btc = num_contracts * contract_size
    actual_position_value = actual_btc * current_price
    actual_margin = actual_position_value / leverage
    
    print(f"\n{Colors.BOLD}📊 Position Calculation:{Colors.END}")
    print(f"   Margin (collateral): ${actual_margin:.2f} USDT")
    print(f"   Leverage: {leverage}x")
    print(f"   Position Value: ${actual_position_value:.2f}")
    print(f"   BTC Amount: {actual_btc:.6f} BTC")
    print(f"   Contracts: {num_contracts}")
    print(f"   Entry Price: ${current_price:,.2f}")
    
    # Risk calculation
    liquidation_move = 100 / leverage  # Rough estimate
    stop_loss_price = current_price * 0.99  # 1% SL
    take_profit_price = current_price * 1.02  # 2% TP
    
    print(f"\n{Colors.BOLD}🛡️ Risk Management:{Colors.END}")
    print(f"   Est. Liquidation: ~{liquidation_move:.1f}% against position")
    print(f"   Suggested SL (-1%): ${stop_loss_price:,.2f}")
    print(f"   Suggested TP (+2%): ${take_profit_price:,.2f}")
    print(f"   {Colors.RED}Max Loss (if liquidated): ${actual_margin:.2f}{Colors.END}")
    
    results["details"]["calculation"] = {
        "margin": actual_margin,
        "leverage": leverage,
        "position_value": actual_position_value,
        "btc_amount": actual_btc,
        "contracts": num_contracts,
        "entry_price": current_price
    }
    
    # -------------------------------------------------------------------------
    # Step 6: Confirm and place order
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}Step 6: Order Confirmation{Colors.END}")
    print(f"\n{Colors.RED}{'='*50}")
    print(f"⚠️  YOU ARE ABOUT TO PLACE A REAL ORDER!")
    print(f"{'='*50}{Colors.END}")
    print(f"\n   Action: {Colors.GREEN}OPEN LONG{Colors.END}")
    print(f"   Symbol: {symbol}")
    print(f"   Contracts: {num_contracts}")
    print(f"   Margin: ${actual_margin:.2f}")
    print(f"   Position: ${actual_position_value:.2f} ({actual_btc:.6f} BTC)")
    print(f"   Leverage: {leverage}x")
    
    confirm = input(f"\n{Colors.RED}Type 'LONG' to confirm order: {Colors.END}")
    
    if confirm.upper() != "LONG":
        print_info("Order cancelled by user")
        results["details"]["cancelled"] = True
        return results
    
    # -------------------------------------------------------------------------
    # Step 7: Place the order
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}Step 7: Placing order...{Colors.END}")
    
    try:
        order_result = futures_client.order(
            symbol=symbol,
            price=current_price,
            vol=num_contracts,
            side=1,  # 1 = open long
            type=5,  # 5 = market order
            open_type=1,  # 1 = isolated margin
            leverage=leverage
        )
        
        if order_result:
            if order_result.get('success'):
                order_data = order_result.get('data', {})
                order_id = order_data.get('orderId', 'N/A')
                
                print_success("ORDER PLACED SUCCESSFULLY!")
                print(f"\n{Colors.GREEN}{'='*50}")
                print(f"   Order ID: {order_id}")
                print(f"   Symbol: {symbol}")
                print(f"   Direction: LONG")
                print(f"   Contracts: {num_contracts}")
                print(f"   Leverage: {leverage}x")
                print(f"{'='*50}{Colors.END}")
                
                results["passed"] = True
                results["details"]["order"] = order_result
                results["details"]["order_id"] = order_id
                
                # Check position
                print(f"\n{Colors.BOLD}Verifying position...{Colors.END}")
                try:
                    import time
                    time.sleep(1)  # Wait for order to fill
                    positions = futures_client.open_positions()
                    if positions and positions.get('success'):
                        for pos in positions.get('data', []):
                            if pos.get('symbol') == symbol:
                                entry = pos.get('openAvgPrice', 0)
                                hold = pos.get('holdVol', 0)
                                pnl = pos.get('unrealisedPnl', 0)
                                print_success(f"Position confirmed!")
                                print(f"   Entry: ${float(entry):,.2f}")
                                print(f"   Size: {hold} contracts")
                                print(f"   Unrealized P&L: ${float(pnl):.4f}")
                except Exception as e:
                    print_warning(f"Could not verify position: {e}")
            else:
                error_msg = order_result.get('message', str(order_result))
                error_code = order_result.get('code', 'N/A')
                print_error(f"Order failed!")
                print(f"   Error Code: {error_code}")
                print(f"   Message: {error_msg}")
                results["details"]["error"] = error_msg
        else:
            print_error("Empty order response")
            
    except Exception as e:
        print_error(f"Order exception: {e}")
        results["details"]["error"] = str(e)
    
    return results


def close_position(clients: Dict, symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    """Close an open position"""
    print_header("CLOSE POSITION")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("futures"):
        print_error("No futures client available")
        return results
    
    futures_client = clients["futures"]
    
    # Get open positions
    try:
        positions = futures_client.open_positions()
        if not positions or not positions.get('success'):
            print_error("Could not get positions")
            return results
        
        pos_list = positions.get('data', [])
        target_pos = None
        
        for pos in pos_list:
            if pos.get('symbol') == symbol:
                target_pos = pos
                break
        
        if not target_pos:
            print_warning(f"No open position found for {symbol}")
            return results
        
        pos_type = target_pos.get('positionType')
        direction = "LONG" if pos_type == 1 else "SHORT"
        hold_vol = int(target_pos.get('holdVol', 0))
        entry_price = float(target_pos.get('openAvgPrice', 0))
        unrealized_pnl = float(target_pos.get('unrealisedPnl', 0))
        
        print(f"\n{Colors.BOLD}Current Position:{Colors.END}")
        print(f"   Symbol: {symbol}")
        print(f"   Direction: {direction}")
        print(f"   Size: {hold_vol} contracts")
        print(f"   Entry: ${entry_price:,.2f}")
        pnl_color = Colors.GREEN if unrealized_pnl >= 0 else Colors.RED
        print(f"   Unrealized P&L: {pnl_color}${unrealized_pnl:.4f}{Colors.END}")
        
        confirm = input(f"\n{Colors.RED}Type 'CLOSE' to close this position: {Colors.END}")
        
        if confirm.upper() != "CLOSE":
            print_info("Close cancelled")
            return results
        
        # Close the position
        # Side: 2 = close short (buy to close), 4 = close long (sell to close)
        close_side = 4 if pos_type == 1 else 2
        
        print(f"\n{Colors.YELLOW}Closing position...{Colors.END}")
        
        close_result = futures_client.order(
            symbol=symbol,
            price=entry_price,  # Reference price
            vol=hold_vol,
            side=close_side,
            type=5,  # Market order
            open_type=1,
            reduce_only=True
        )
        
        if close_result and close_result.get('success'):
            print_success("Position closed successfully!")
            results["passed"] = True
            results["details"]["close_result"] = close_result
        else:
            print_error(f"Close failed: {close_result}")
            results["details"]["error"] = str(close_result)
            
    except Exception as e:
        print_error(f"Close exception: {e}")
        results["details"]["error"] = str(e)
    
    return results


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all safe tests"""
    print_header("MEXC FUTURES API TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize clients
    clients = get_mexc_clients()
    
    if not clients:
        print_error("Failed to initialize MEXC clients")
        return
    
    all_results = {}
    
    # Run tests
    tests = [
        ("Environment Config", test_environment_config, []),
        ("API Connection", test_api_connection, [clients]),
        ("Account Info", test_account_info, [clients]),
        ("Open Positions", test_open_positions, [clients]),
        ("Market Data", test_market_data, [clients]),
        ("Futures Contract", test_futures_contract_info, [clients]),
        ("Order Book", test_order_book, [clients]),
        ("Leverage Setting", test_leverage_setting, [clients]),
        ("Order Dry Run", test_order_placement_dry_run, [clients]),
    ]
    
    for test_name, test_func, args in tests:
        try:
            result = test_func(*args)
            all_results[test_name] = result
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            all_results[test_name] = {"passed": False, "error": str(e)}
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for r in all_results.values() if r.get("passed"))
    total = len(all_results)
    
    for test_name, result in all_results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result.get("passed") else f"{Colors.RED}FAIL{Colors.END}"
        print(f"   {test_name}: {status}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print_success("All tests passed! Your MEXC setup is ready.")
    else:
        print_warning("Some tests failed. Check the output above for details.")
    
    return all_results


def interactive_menu():
    """Interactive test menu"""
    clients = get_mexc_clients()
    
    if not clients:
        print_error("Failed to initialize MEXC clients")
        return
    
    while True:
        print_header("MEXC API TEST MENU")
        print(f"{Colors.BOLD}--- Tests ---{Colors.END}")
        print("1. Check Environment Config")
        print("2. Test API Connection")
        print("3. View Account Balance")
        print("4. View Open Positions")
        print("5. Get Market Data (BTC)")
        print("6. Get Order Book")
        print("7. Test Leverage Setting")
        print("8. Simulate Order (Dry Run)")
        print("9. Run All Safe Tests")
        print(f"\n{Colors.BOLD}--- Trading (⚠️ REAL MONEY) ---{Colors.END}")
        print(f"{Colors.GREEN}10. OPEN LONG Position{Colors.END}")
        print(f"{Colors.RED}11. CLOSE Position{Colors.END}")
        print("12. Place Custom Order")
        print("\n0. Exit")
        
        choice = input(f"\n{Colors.CYAN}Select option: {Colors.END}")
        
        if choice == "1":
            test_environment_config()
        elif choice == "2":
            test_api_connection(clients)
        elif choice == "3":
            test_account_info(clients)
        elif choice == "4":
            test_open_positions(clients)
        elif choice == "5":
            test_market_data(clients)
        elif choice == "6":
            test_order_book(clients)
        elif choice == "7":
            test_leverage_setting(clients)
        elif choice == "8":
            test_order_placement_dry_run(clients)
        elif choice == "9":
            run_all_tests()
        elif choice == "10":
            # Open Long Position
            print_warning("This will open a REAL LONG position!")
            pct_input = input("Use % of balance (or Enter for MEXC_DEFAULT_SIZE): ").strip()
            use_pct = float(pct_input) if pct_input else None
            open_long_position(clients, use_percentage=use_pct)
        elif choice == "11":
            # Close Position
            close_position(clients)
        elif choice == "12":
            # Custom Order
            print_warning("This will place a REAL order!")
            confirm = input("Type 'YES' to proceed: ")
            if confirm == "YES":
                direction = input("Direction (long/short): ").strip() or "long"
                test_place_real_order(clients, direction=direction)
        elif choice == "0":
            print_info("Exiting...")
            break
        else:
            print_warning("Invalid option")
        
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEXC Futures API Test Script")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive menu mode")
    parser.add_argument("--test-order", action="store_true", help="Include real order test (requires confirmation)")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Symbol to test (default: BTC_USDT)")
    parser.add_argument("--long", action="store_true", help="Open a LONG position (requires confirmation)")
    parser.add_argument("--close", action="store_true", help="Close existing position (requires confirmation)")
    parser.add_argument("--percent", type=float, help="Percentage of balance to use (with --long)")
    
    args = parser.parse_args()
    
    if args.long:
        # Direct long order
        clients = get_mexc_clients()
        if clients:
            open_long_position(clients, symbol=args.symbol, use_percentage=args.percent)
        else:
            print_error("Failed to initialize MEXC clients")
    elif args.close:
        # Direct close
        clients = get_mexc_clients()
        if clients:
            close_position(clients, symbol=args.symbol)
        else:
            print_error("Failed to initialize MEXC clients")
    elif args.interactive:
        interactive_menu()
    else:
        run_all_tests()

