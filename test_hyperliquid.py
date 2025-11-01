#!/usr/bin/env python3
"""
Test script for Hyperliquid API connection
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from eth_account import Account

load_dotenv()

print("Testing Hyperliquid API connection...")

# Test public API (info client)
try:
    info_client = Info(skip_ws=True)
    print("\n✅ Info client initialized successfully")
    
    # Test fetching candles
    candles = info_client.candles_snapshot(
        coin="BTC",
        interval="1m",
        startTime=int((datetime.now().timestamp() - 10 * 60) * 1000),
        endTime=int(datetime.now().timestamp() * 1000)
    )
    print(f"✅ Fetched {len(candles)} candles for BTC")
except Exception as e:
    print(f"❌ Info client test failed: {e}")

# Test private API (exchange client)
private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
if private_key:
    try:
        account = Account.from_key(private_key)
        exchange_client = Exchange(account, skip_ws=True)
        print(f"\n✅ Exchange client initialized successfully")
        print(f"   Wallet address: {account.address}")
        
        # Test getting account info
        # Note: Add any test operations here if needed
        print("✅ Account connected to Hyperliquid")
    except Exception as e:
        print(f"❌ Exchange client test failed: {e}")
else:
    print("\n⚠️  HYPERLIQUID_PRIVATE_KEY not set - skipping exchange client test")

print("\nTest complete!")
