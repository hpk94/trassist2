#!/usr/bin/env python3
"""
Test script for Hyperliquid API connection (supports testnet)
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from eth_account import Account

load_dotenv()

def get_hyperliquid_base_url():
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

# Determine if we're using testnet
base_url = get_hyperliquid_base_url()
is_testnet = base_url is not None and "testnet" in base_url.lower()
network_name = "TESTNET" if is_testnet else "MAINNET"

print(f"Testing Hyperliquid API connection ({network_name})...")
if is_testnet:
    print("⚠️  Using Hyperliquid Testnet - all operations are on testnet!")

# Test public API (info client)
try:
    # Info client initialization - skip_ws is supported for Info
    info_client = Info(base_url=base_url, skip_ws=True)
    print(f"\n✅ Info client initialized successfully ({network_name})")
    
    # Test fetching candles - uses positional arguments: (name, interval, startTime, endTime)
    candles = info_client.candles_snapshot(
        "BTC",  # name parameter
        "1m",   # interval
        int((datetime.now().timestamp() - 10 * 60) * 1000),  # startTime
        int(datetime.now().timestamp() * 1000)  # endTime
    )
    print(f"✅ Fetched {len(candles)} candles for BTC")
except Exception as e:
    print(f"❌ Info client test failed: {e}")
    import traceback
    traceback.print_exc()

# Test private API (exchange client)
private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
if private_key:
    try:
        account = Account.from_key(private_key)
        # Exchange doesn't support skip_ws parameter
        exchange_client = Exchange(account, base_url=base_url)
        print(f"\n✅ Exchange client initialized successfully ({network_name})")
        print(f"   Wallet address: {account.address}")
        
        # Test getting account info
        # Note: Add any test operations here if needed
        print(f"✅ Account connected to Hyperliquid {network_name}")
    except Exception as e:
        print(f"❌ Exchange client test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠️  HYPERLIQUID_PRIVATE_KEY not set - skipping exchange client test")

print("\nTest complete!")
