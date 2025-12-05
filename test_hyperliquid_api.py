#!/usr/bin/env python3
"""
Hyperliquid API Test Script

A comprehensive test script to verify Hyperliquid API connectivity,
test market data retrieval, account management, and order placement.

Hyperliquid is a decentralized exchange - NO regional blocks!

Usage:
    python test_hyperliquid_api.py                  # Run all safe tests
    python test_hyperliquid_api.py --interactive    # Interactive menu mode
    python test_hyperliquid_api.py --long           # Open a LONG position
    python test_hyperliquid_api.py --close          # Close existing position
    python test_hyperliquid_api.py --testnet        # Force testnet mode
    
⚠️  WARNING: Order tests will use REAL funds unless using testnet!
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

DEFAULT_COIN = "BTC"

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
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
# Hyperliquid Client Initialization
# ============================================================================

def get_hyperliquid_base_url(force_testnet: bool = False) -> Optional[str]:
    """Get Hyperliquid base URL based on testnet setting."""
    if force_testnet:
        return "https://api.hyperliquid-testnet.xyz"
    
    explicit_base_url = os.getenv("HYPERLIQUID_BASE_URL")
    if explicit_base_url:
        return explicit_base_url
    
    testnet_enabled = os.getenv("HYPERLIQUID_TESTNET", "false").lower() in ("true", "1", "yes")
    if testnet_enabled:
        return "https://api.hyperliquid-testnet.xyz"
    
    return None  # Mainnet


def get_hyperliquid_clients(force_testnet: bool = False) -> Optional[Dict]:
    """Initialize and return Hyperliquid clients"""
    try:
        from hyperliquid.info import Info
        from hyperliquid.exchange import Exchange
        from eth_account import Account
        
        base_url = get_hyperliquid_base_url(force_testnet)
        is_testnet = base_url is not None and "testnet" in base_url.lower()
        
        # Public info client (no auth needed)
        info_client = Info(base_url=base_url, skip_ws=True)
        
        # Private exchange client (needs private key)
        exchange_client = None
        wallet_address = None
        private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
        
        if private_key:
            try:
                account = Account.from_key(private_key)
                exchange_client = Exchange(account, base_url=base_url)
                wallet_address = account.address
            except Exception as e:
                print_warning(f"Could not initialize exchange client: {e}")
        
        return {
            "info": info_client,
            "exchange": exchange_client,
            "wallet_address": wallet_address,
            "base_url": base_url,
            "is_testnet": is_testnet,
            "network": "TESTNET" if is_testnet else "MAINNET"
        }
    except ImportError as e:
        print_error(f"Required library not installed: {e}")
        print_info("Run: pip install hyperliquid-python-sdk eth-account")
        return None


# ============================================================================
# Test Functions
# ============================================================================

def test_environment_config() -> Dict[str, Any]:
    """Test 1: Check environment configuration"""
    print_header("Test 1: Environment Configuration")
    
    results = {"passed": True, "details": {}}
    
    # Check private key
    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    testnet = os.getenv("HYPERLIQUID_TESTNET", "false")
    default_size = os.getenv("HYPERLIQUID_DEFAULT_SIZE", "0.001")
    enable_orders = os.getenv("HYPERLIQUID_ENABLE_ORDERS", "false")
    
    if private_key:
        # Mask the key for display
        masked = f"{private_key[:6]}...{private_key[-4:]}" if len(private_key) > 10 else "***"
        print_success(f"HYPERLIQUID_PRIVATE_KEY: Set ({masked})")
        results["details"]["private_key"] = "configured"
        
        # Derive wallet address
        try:
            from eth_account import Account
            account = Account.from_key(private_key)
            print_success(f"Wallet Address: {account.address}")
            results["details"]["wallet_address"] = account.address
        except Exception as e:
            print_error(f"Invalid private key: {e}")
            results["passed"] = False
    else:
        print_error("HYPERLIQUID_PRIVATE_KEY: NOT SET")
        print_info("See HOW_TO_GET_HYPERLIQUID_PRIVATE_KEY.md for setup instructions")
        results["passed"] = False
        results["details"]["private_key"] = "missing"
    
    # Display settings
    print(f"\n{Colors.BOLD}Trading Parameters:{Colors.END}")
    is_testnet = testnet.lower() in ("true", "1", "yes")
    network = f"{Colors.MAGENTA}TESTNET{Colors.END}" if is_testnet else f"{Colors.GREEN}MAINNET{Colors.END}"
    print(f"   Network: {network}")
    print(f"   Default Size: {default_size} BTC")
    print(f"   Enable Orders: {enable_orders}")
    
    if enable_orders.lower() == "true":
        if is_testnet:
            print_info("Orders enabled on TESTNET (safe for testing)")
        else:
            print_warning("LIVE TRADING ENABLED on MAINNET! Real funds at risk!")
    else:
        print_info("Paper trading mode (orders simulated)")
    
    results["details"]["testnet"] = is_testnet
    results["details"]["default_size"] = default_size
    results["details"]["enable_orders"] = enable_orders
    
    return results


def test_api_connection(clients: Dict) -> Dict[str, Any]:
    """Test 2: Test API connection"""
    print_header("Test 2: API Connection")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("info"):
        print_error("No info client available")
        return results
    
    info_client = clients["info"]
    network = clients.get("network", "UNKNOWN")
    
    try:
        # Test by fetching metadata
        meta = info_client.meta()
        if meta:
            print_success(f"Connected to Hyperliquid {network}")
            universe = meta.get("universe", [])
            print_info(f"Available coins: {len(universe)}")
            results["passed"] = True
            results["details"]["coins_available"] = len(universe)
        else:
            print_error("Could not fetch metadata")
    except Exception as e:
        print_error(f"Connection failed: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_account_info(clients: Dict) -> Dict[str, Any]:
    """Test 3: Get account information"""
    print_header("Test 3: Account Information")
    
    results = {"passed": False, "details": {}}
    
    if not clients:
        print_error("No clients available")
        return results
    
    info_client = clients.get("info")
    wallet_address = clients.get("wallet_address")
    
    if not wallet_address:
        print_warning("No wallet address - private key not configured")
        print_info("Set HYPERLIQUID_PRIVATE_KEY in .env to view account info")
        results["details"]["no_wallet"] = True
        return results
    
    try:
        # Get user state
        user_state = info_client.user_state(wallet_address)
        
        if user_state:
            print_success("Account data retrieved successfully")
            results["passed"] = True
            
            # Extract balance info
            margin_summary = user_state.get("marginSummary", {})
            account_value = float(margin_summary.get("accountValue", 0))
            total_margin = float(margin_summary.get("totalMarginUsed", 0))
            withdrawable = float(user_state.get("withdrawable", 0))
            
            print(f"\n{Colors.BOLD}💰 Account Balance:{Colors.END}")
            print(f"   Account Value: ${account_value:,.2f}")
            print(f"   Margin Used: ${total_margin:,.2f}")
            print(f"   Withdrawable: ${withdrawable:,.2f}")
            
            results["details"]["account_value"] = account_value
            results["details"]["margin_used"] = total_margin
            results["details"]["withdrawable"] = withdrawable
            
            # Check if enough balance
            default_size = float(os.getenv("HYPERLIQUID_DEFAULT_SIZE", "0.001"))
            if account_value < 10:
                print_warning(f"Low balance (${account_value:.2f})")
                if clients.get("is_testnet"):
                    print_info("Get testnet tokens from https://hyperliquidfaucet.com/")
        else:
            print_warning("Empty account state response")
            
    except Exception as e:
        print_error(f"Account info error: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_open_positions(clients: Dict) -> Dict[str, Any]:
    """Test 4: Get open positions"""
    print_header("Test 4: Open Positions")
    
    results = {"passed": False, "details": {}}
    
    if not clients:
        print_error("No clients available")
        return results
    
    info_client = clients.get("info")
    wallet_address = clients.get("wallet_address")
    
    if not wallet_address:
        print_warning("No wallet address configured")
        return results
    
    try:
        user_state = info_client.user_state(wallet_address)
        
        if user_state:
            print_success("Positions data retrieved successfully")
            results["passed"] = True
            
            positions = user_state.get("assetPositions", [])
            active_positions = []
            
            for pos_data in positions:
                pos = pos_data.get("position", {})
                coin = pos.get("coin", "")
                szi = float(pos.get("szi", 0))
                
                if abs(szi) > 1e-12:
                    entry_px = float(pos.get("entryPx", 0))
                    unrealized_pnl = float(pos.get("unrealizedPnl", 0))
                    leverage = pos.get("leverage", {})
                    lev_value = leverage.get("value", 1) if isinstance(leverage, dict) else leverage
                    
                    direction = "LONG" if szi > 0 else "SHORT"
                    pnl_color = Colors.GREEN if unrealized_pnl >= 0 else Colors.RED
                    
                    active_positions.append({
                        "coin": coin,
                        "direction": direction,
                        "size": abs(szi),
                        "entry": entry_px,
                        "pnl": unrealized_pnl,
                        "leverage": lev_value
                    })
            
            if active_positions:
                print(f"\n{Colors.BOLD}📍 Open Positions:{Colors.END}")
                for pos in active_positions:
                    pnl_color = Colors.GREEN if pos["pnl"] >= 0 else Colors.RED
                    print(f"\n   {Colors.BOLD}{pos['coin']}{Colors.END}")
                    print(f"      Direction: {pos['direction']}")
                    print(f"      Size: {pos['size']:.6f}")
                    print(f"      Entry: ${pos['entry']:,.2f}")
                    print(f"      Leverage: {pos['leverage']}x")
                    print(f"      Unrealized P&L: {pnl_color}${pos['pnl']:.2f}{Colors.END}")
                
                results["details"]["positions"] = active_positions
            else:
                print_info("No open positions")
                results["details"]["positions"] = []
        else:
            print_warning("Could not retrieve positions")
            
    except Exception as e:
        print_error(f"Positions error: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_market_data(clients: Dict, coin: str = DEFAULT_COIN) -> Dict[str, Any]:
    """Test 5: Get market data"""
    print_header(f"Test 5: Market Data ({coin})")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("info"):
        print_error("No info client available")
        return results
    
    info_client = clients["info"]
    
    try:
        # Get all mids (mid prices)
        mids = info_client.all_mids()
        
        if mids and coin in mids:
            price = float(mids[coin])
            print_success(f"Market data retrieved")
            print(f"\n{Colors.BOLD}📊 {coin} Price:{Colors.END}")
            print(f"   Mid Price: ${price:,.2f}")
            results["details"]["price"] = price
            results["passed"] = True
        
        # Get recent candles
        try:
            now = int(datetime.now().timestamp() * 1000)
            start = now - (60 * 60 * 1000)  # Last hour
            candles = info_client.candles_snapshot(coin, "1m", start, now)
            
            if candles and len(candles) > 0:
                print(f"\n{Colors.BOLD}🕐 Recent 1m Candles:{Colors.END}")
                for c in candles[-3:]:
                    o = float(c.get("o", 0))
                    h = float(c.get("h", 0))
                    l = float(c.get("l", 0))
                    close = float(c.get("c", 0))
                    print(f"   O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{close:.2f}")
                    
                results["details"]["candles"] = len(candles)
        except Exception as e:
            print_warning(f"Could not get candles: {e}")
        
        # Get 24h stats if available
        try:
            meta = info_client.meta()
            for asset in meta.get("universe", []):
                if asset.get("name") == coin:
                    print(f"\n{Colors.BOLD}📈 Contract Info:{Colors.END}")
                    print(f"   Max Leverage: {asset.get('maxLeverage', 'N/A')}x")
                    print(f"   Size Decimals: {asset.get('szDecimals', 'N/A')}")
                    results["details"]["max_leverage"] = asset.get("maxLeverage")
                    break
        except:
            pass
            
    except Exception as e:
        print_error(f"Market data error: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_order_book(clients: Dict, coin: str = DEFAULT_COIN) -> Dict[str, Any]:
    """Test 6: Get order book"""
    print_header(f"Test 6: Order Book ({coin})")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("info"):
        print_error("No info client available")
        return results
    
    info_client = clients["info"]
    
    try:
        # Get L2 order book
        l2 = info_client.l2_snapshot(coin)
        
        if l2:
            print_success("Order book retrieved")
            results["passed"] = True
            
            levels = l2.get("levels", [[], []])
            bids = levels[0][:5] if len(levels) > 0 else []
            asks = levels[1][:5] if len(levels) > 1 else []
            
            print(f"\n{Colors.BOLD}📗 Order Book (Top 5):{Colors.END}")
            
            print(f"\n   {Colors.RED}ASKS (Sell Orders):{Colors.END}")
            for ask in reversed(asks):
                price = float(ask.get("px", 0))
                size = float(ask.get("sz", 0))
                print(f"      ${price:,.2f}  |  {size:.4f}")
            
            if bids and asks:
                best_bid = float(bids[0].get("px", 0))
                best_ask = float(asks[0].get("px", 0))
                spread = best_ask - best_bid
                spread_pct = (spread / best_bid) * 100 if best_bid > 0 else 0
                print(f"\n   {Colors.YELLOW}--- Spread: ${spread:.2f} ({spread_pct:.4f}%) ---{Colors.END}")
                results["details"]["spread"] = spread
            
            print(f"\n   {Colors.GREEN}BIDS (Buy Orders):{Colors.END}")
            for bid in bids:
                price = float(bid.get("px", 0))
                size = float(bid.get("sz", 0))
                print(f"      ${price:,.2f}  |  {size:.4f}")
        else:
            print_warning("Could not get order book")
            
    except Exception as e:
        print_error(f"Order book error: {e}")
        results["details"]["error"] = str(e)
    
    return results


def test_order_dry_run(clients: Dict, coin: str = DEFAULT_COIN) -> Dict[str, Any]:
    """Test 7: Simulate order placement"""
    print_header("Test 7: Order Placement (DRY RUN)")
    
    results = {"passed": False, "details": {}}
    
    enable_orders = os.getenv("HYPERLIQUID_ENABLE_ORDERS", "false").lower()
    
    if enable_orders == "true" and not clients.get("is_testnet"):
        print_warning("HYPERLIQUID_ENABLE_ORDERS is TRUE on MAINNET!")
        print_warning("This test would place REAL orders!")
        print_info("Skipping to prevent accidental trading")
        print_info("Set HYPERLIQUID_ENABLE_ORDERS=false or use testnet")
        results["details"]["skipped"] = "live_trading_enabled"
        return results
    
    # Get current price
    current_price = 95000.0  # Fallback
    try:
        if clients and clients.get("info"):
            mids = clients["info"].all_mids()
            if mids and coin in mids:
                current_price = float(mids[coin])
    except:
        pass
    
    default_size = float(os.getenv("HYPERLIQUID_DEFAULT_SIZE", "0.001"))
    
    print_info("Simulating order (NOT placing real order)")
    print(f"\n{Colors.BOLD}📝 Simulated Order Details:{Colors.END}")
    print(f"   Coin: {coin}")
    print(f"   Direction: LONG (Buy)")
    print(f"   Order Type: Market")
    print(f"   Size: {default_size} {coin}")
    print(f"   Position Value: ${current_price * default_size:,.2f}")
    print(f"   Estimated Entry: ${current_price:,.2f}")
    
    # Risk management
    stop_loss = current_price * 0.99
    take_profit = current_price * 1.02
    
    print(f"\n{Colors.BOLD}🛡️ Risk Management:{Colors.END}")
    print(f"   Stop Loss (-1%): ${stop_loss:,.2f}")
    print(f"   Take Profit (+2%): ${take_profit:,.2f}")
    print(f"   Risk/Reward: 2:1")
    
    # P&L calculation
    loss_amount = current_price * default_size * 0.01
    profit_amount = current_price * default_size * 0.02
    
    print(f"\n{Colors.BOLD}💰 Potential P&L:{Colors.END}")
    print(f"   {Colors.RED}Max Loss: ${loss_amount:.2f}{Colors.END}")
    print(f"   {Colors.GREEN}Potential Profit: ${profit_amount:.2f}{Colors.END}")
    
    print_success("Order simulation complete")
    results["passed"] = True
    results["details"]["simulation"] = {
        "coin": coin,
        "size": default_size,
        "entry": current_price,
        "value": current_price * default_size
    }
    
    return results


def open_long_position(clients: Dict, coin: str = DEFAULT_COIN, size_override: float = None) -> Dict[str, Any]:
    """Open a LONG position on Hyperliquid"""
    print_header("OPEN LONG POSITION")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("exchange"):
        print_error("No exchange client available")
        print_info("Set HYPERLIQUID_PRIVATE_KEY in .env")
        return results
    
    exchange = clients["exchange"]
    info_client = clients["info"]
    network = clients.get("network", "UNKNOWN")
    is_testnet = clients.get("is_testnet", False)
    
    # Step 1: Get current price
    print(f"\n{Colors.BOLD}Step 1: Getting current price...{Colors.END}")
    try:
        mids = info_client.all_mids()
        current_price = float(mids.get(coin, 0))
        print_success(f"Current {coin} price: ${current_price:,.2f}")
    except Exception as e:
        print_error(f"Could not get price: {e}")
        return results
    
    # Step 2: Get account balance
    print(f"\n{Colors.BOLD}Step 2: Getting account balance...{Colors.END}")
    wallet_address = clients.get("wallet_address")
    account_value = 0
    try:
        user_state = info_client.user_state(wallet_address)
        margin_summary = user_state.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0))
        print_success(f"Account value: ${account_value:,.2f}")
    except Exception as e:
        print_warning(f"Could not get balance: {e}")
    
    # Step 3: Determine size
    print(f"\n{Colors.BOLD}Step 3: Determining position size...{Colors.END}")
    size = size_override or float(os.getenv("HYPERLIQUID_DEFAULT_SIZE", "0.001"))
    position_value = size * current_price
    
    print(f"\n{Colors.BOLD}📊 Order Details:{Colors.END}")
    print(f"   Coin: {coin}")
    print(f"   Direction: LONG")
    print(f"   Size: {size} {coin}")
    print(f"   Position Value: ${position_value:,.2f}")
    print(f"   Current Price: ${current_price:,.2f}")
    print(f"   Network: {Colors.MAGENTA if is_testnet else Colors.GREEN}{network}{Colors.END}")
    
    # Step 4: Confirmation
    print(f"\n{Colors.RED}{'='*50}")
    if is_testnet:
        print(f"⚠️  PLACING ORDER ON TESTNET")
    else:
        print(f"⚠️  PLACING REAL ORDER ON MAINNET!")
    print(f"{'='*50}{Colors.END}")
    
    confirm = input(f"\n{Colors.RED}Type 'LONG' to confirm: {Colors.END}")
    
    if confirm.upper() != "LONG":
        print_info("Order cancelled by user")
        results["details"]["cancelled"] = True
        return results
    
    # Step 5: Place order
    print(f"\n{Colors.BOLD}Step 4: Placing order...{Colors.END}")
    
    try:
        slippage = float(os.getenv("HYPERLIQUID_SLIPPAGE", "0.01"))
        
        response = exchange.market_open(
            name=coin,
            is_buy=True,
            sz=size,
            px=current_price,
            slippage=slippage
        )
        
        if response:
            status = response.get("status", "unknown")
            if status == "ok":
                print_success("ORDER PLACED SUCCESSFULLY!")
                
                # Get fill info
                statuses = response.get("response", {}).get("data", {}).get("statuses", [])
                for s in statuses:
                    if "filled" in s:
                        filled = s["filled"]
                        print(f"\n{Colors.GREEN}{'='*50}")
                        print(f"   Coin: {coin}")
                        print(f"   Direction: LONG")
                        print(f"   Filled Size: {filled.get('totalSz', size)}")
                        print(f"   Avg Price: ${float(filled.get('avgPx', current_price)):,.2f}")
                        print(f"{'='*50}{Colors.END}")
                
                results["passed"] = True
                results["details"]["order"] = response
            else:
                print_error(f"Order failed: {response}")
                results["details"]["error"] = str(response)
        else:
            print_error("Empty order response")
            
    except Exception as e:
        print_error(f"Order exception: {e}")
        results["details"]["error"] = str(e)
    
    return results


def close_position(clients: Dict, coin: str = DEFAULT_COIN) -> Dict[str, Any]:
    """Close an open position"""
    print_header("CLOSE POSITION")
    
    results = {"passed": False, "details": {}}
    
    if not clients or not clients.get("exchange"):
        print_error("No exchange client available")
        return results
    
    exchange = clients["exchange"]
    info_client = clients["info"]
    wallet_address = clients.get("wallet_address")
    
    try:
        user_state = info_client.user_state(wallet_address)
        positions = user_state.get("assetPositions", [])
        
        target_pos = None
        for pos_data in positions:
            pos = pos_data.get("position", {})
            if pos.get("coin") == coin:
                szi = float(pos.get("szi", 0))
                if abs(szi) > 1e-12:
                    target_pos = pos
                    break
        
        if not target_pos:
            print_warning(f"No open position found for {coin}")
            return results
        
        szi = float(target_pos.get("szi", 0))
        direction = "LONG" if szi > 0 else "SHORT"
        size = abs(szi)
        entry = float(target_pos.get("entryPx", 0))
        pnl = float(target_pos.get("unrealizedPnl", 0))
        
        pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
        
        print(f"\n{Colors.BOLD}Current Position:{Colors.END}")
        print(f"   Coin: {coin}")
        print(f"   Direction: {direction}")
        print(f"   Size: {size}")
        print(f"   Entry: ${entry:,.2f}")
        print(f"   Unrealized P&L: {pnl_color}${pnl:.2f}{Colors.END}")
        
        confirm = input(f"\n{Colors.RED}Type 'CLOSE' to close: {Colors.END}")
        
        if confirm.upper() != "CLOSE":
            print_info("Close cancelled")
            return results
        
        print(f"\n{Colors.YELLOW}Closing position...{Colors.END}")
        
        # Close position
        is_buy = szi < 0  # If short, buy to close. If long, sell to close
        slippage = float(os.getenv("HYPERLIQUID_SLIPPAGE", "0.01"))
        
        mids = info_client.all_mids()
        current_price = float(mids.get(coin, 0))
        
        response = exchange.market_open(
            name=coin,
            is_buy=is_buy,
            sz=size,
            px=current_price,
            slippage=slippage,
            reduce_only=True
        )
        
        if response and response.get("status") == "ok":
            print_success("Position closed successfully!")
            results["passed"] = True
            results["details"]["response"] = response
        else:
            print_error(f"Close failed: {response}")
            results["details"]["error"] = str(response)
            
    except Exception as e:
        print_error(f"Close exception: {e}")
        results["details"]["error"] = str(e)
    
    return results


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests(force_testnet: bool = False):
    """Run all safe tests"""
    print_header("HYPERLIQUID API TEST SUITE")
    
    clients = get_hyperliquid_clients(force_testnet)
    
    if not clients:
        print_error("Failed to initialize Hyperliquid clients")
        return
    
    network = clients.get("network", "UNKNOWN")
    print(f"Network: {Colors.MAGENTA if clients.get('is_testnet') else Colors.GREEN}{network}{Colors.END}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    tests = [
        ("Environment Config", test_environment_config, []),
        ("API Connection", test_api_connection, [clients]),
        ("Account Info", test_account_info, [clients]),
        ("Open Positions", test_open_positions, [clients]),
        ("Market Data", test_market_data, [clients]),
        ("Order Book", test_order_book, [clients]),
        ("Order Dry Run", test_order_dry_run, [clients]),
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
        print_success("All tests passed! Your Hyperliquid setup is ready.")
    else:
        print_warning("Some tests failed. Check the output above for details.")
    
    return all_results


def interactive_menu(force_testnet: bool = False):
    """Interactive test menu"""
    clients = get_hyperliquid_clients(force_testnet)
    
    if not clients:
        print_error("Failed to initialize Hyperliquid clients")
        return
    
    while True:
        network = clients.get("network", "UNKNOWN")
        print_header(f"HYPERLIQUID API TEST MENU ({network})")
        print(f"{Colors.BOLD}--- Tests ---{Colors.END}")
        print("1. Check Environment Config")
        print("2. Test API Connection")
        print("3. View Account Balance")
        print("4. View Open Positions")
        print("5. Get Market Data (BTC)")
        print("6. Get Order Book")
        print("7. Simulate Order (Dry Run)")
        print("8. Run All Safe Tests")
        print(f"\n{Colors.BOLD}--- Trading ---{Colors.END}")
        print(f"{Colors.GREEN}9.  OPEN LONG Position{Colors.END}")
        print(f"{Colors.RED}10. CLOSE Position{Colors.END}")
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
            test_order_dry_run(clients)
        elif choice == "8":
            run_all_tests(force_testnet)
        elif choice == "9":
            open_long_position(clients)
        elif choice == "10":
            close_position(clients)
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
    parser = argparse.ArgumentParser(description="Hyperliquid API Test Script")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive menu mode")
    parser.add_argument("--testnet", "-t", action="store_true", help="Force testnet mode")
    parser.add_argument("--long", action="store_true", help="Open a LONG position")
    parser.add_argument("--close", action="store_true", help="Close existing position")
    parser.add_argument("--coin", default=DEFAULT_COIN, help="Coin to trade (default: BTC)")
    parser.add_argument("--size", type=float, help="Position size override")
    
    args = parser.parse_args()
    
    if args.long:
        clients = get_hyperliquid_clients(args.testnet)
        if clients:
            open_long_position(clients, coin=args.coin, size_override=args.size)
        else:
            print_error("Failed to initialize clients")
    elif args.close:
        clients = get_hyperliquid_clients(args.testnet)
        if clients:
            close_position(clients, coin=args.coin)
        else:
            print_error("Failed to initialize clients")
    elif args.interactive:
        interactive_menu(args.testnet)
    else:
        run_all_tests(args.testnet)

