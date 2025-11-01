#!/usr/bin/env python3
"""
Test MEXC Futures Connection and Trading Setup
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("MEXC FUTURES CONFIGURATION TEST")
print("="*60 + "\n")

# Check environment variables
print("1. Checking Environment Variables:")
api_key = os.getenv("MEXC_API_KEY")
api_secret = os.getenv("MEXC_API_SECRET")
leverage = os.getenv("MEXC_LEVERAGE", "30")
size = os.getenv("MEXC_DEFAULT_SIZE", "10.0")
enabled = os.getenv("MEXC_ENABLE_ORDERS", "false")

if api_key:
    print(f"   ✅ MEXC_API_KEY: Set (length: {len(api_key)})")
else:
    print("   ❌ MEXC_API_KEY: NOT SET")

if api_secret:
    print(f"   ✅ MEXC_API_SECRET: Set (length: {len(api_secret)})")
else:
    print("   ❌ MEXC_API_SECRET: NOT SET")

print(f"   ℹ️  MEXC_LEVERAGE: {leverage}x")
print(f"   ℹ️  MEXC_DEFAULT_SIZE: ${size} USDT")
print(f"   {'🟢' if enabled.lower() == 'true' else '🔴'} MEXC_ENABLE_ORDERS: {enabled}")

if not api_key or not api_secret:
    print("\n❌ Please set MEXC_API_KEY and MEXC_API_SECRET in your .env file")
    exit(1)

# Test API connection
print("\n2. Testing MEXC Futures API Connection:")
try:
    from pymexc import futures
    print("   ✅ pymexc library imported successfully")
    
    client = futures.HTTP(api_key=api_key, api_secret=api_secret)
    print("   ✅ Futures client initialized")
    
    # Test API connection by getting account info
    try:
        account_info = client.account_assets()
        print("   ✅ API connection successful!")
        print(f"\n   📊 Account Information:")
        
        if isinstance(account_info, dict):
            # Find USDT balance
            assets = account_info.get('data', [])
            for asset in assets:
                if asset.get('currency') == 'USDT':
                    available = float(asset.get('availableBalance', 0))
                    frozen = float(asset.get('frozenBalance', 0))
                    total = available + frozen
                    print(f"      💰 USDT Balance: ${total:.2f}")
                    print(f"         Available: ${available:.2f}")
                    print(f"         In Use: ${frozen:.2f}")
                    
                    if available < float(size):
                        print(f"\n      ⚠️  Warning: Available balance (${available:.2f}) is less than")
                        print(f"         default order size (${size} USDT)")
                    break
        else:
            print(f"      {account_info}")
            
    except Exception as e:
        print(f"   ❌ API connection failed: {e}")
        print("\n   Possible issues:")
        print("   - API keys are incorrect")
        print("   - Futures trading not enabled on your account")
        print("   - IP not whitelisted")
        exit(1)
    
    # Check open positions
    try:
        positions = client.open_positions()
        print(f"\n   📍 Open Positions:")
        if positions and isinstance(positions, dict):
            pos_list = positions.get('data', [])
            if pos_list:
                for pos in pos_list:
                    symbol = pos.get('symbol')
                    side = 'LONG' if pos.get('positionType') == 1 else 'SHORT'
                    size_pos = pos.get('holdVol', 0)
                    leverage_pos = pos.get('leverage', 0)
                    unrealized_pnl = pos.get('unrealisedPnl', 0)
                    print(f"      • {symbol}: {side} {size_pos} @ {leverage_pos}x")
                    print(f"        Unrealized P&L: ${unrealized_pnl}")
            else:
                print("      No open positions")
        else:
            print("      No open positions")
    except Exception as e:
        print(f"      Could not fetch positions: {e}")
    
except ImportError:
    print("   ❌ pymexc library not installed")
    print("   Run: pip install pymexc")
    exit(1)
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Test trading parameters
print("\n3. Trading Parameters:")
print(f"   Symbol: BTCUSDT (BTC_USDT perpetual)")
print(f"   Leverage: {leverage}x")
print(f"   Order Size: ${size} USDT")
print(f"   Margin Mode: Isolated")
print(f"   Order Type: Market")

# Calculate position details
try:
    lev = int(leverage)
    sz = float(size)
    position_value = sz * lev
    print(f"\n   📊 Position Calculation:")
    print(f"      Margin Required: ${sz:.2f}")
    print(f"      Position Value: ${position_value:.2f}")
    print(f"      Max Loss: ${sz:.2f} (if stop loss hit)")
except:
    pass

# Safety check
print("\n4. Safety Status:")
if enabled.lower() == "true":
    print("   🚨 LIVE TRADING ENABLED!")
    print("   ⚠️  Real orders WILL be placed!")
    print("\n   Make sure you:")
    print("   • Have tested with small amounts")
    print("   • Understand the risks")
    print("   • Have set appropriate stop losses")
else:
    print("   ✅ Paper Trading Mode (MEXC_ENABLE_ORDERS=false)")
    print("   📝 Orders will be simulated but NOT placed")
    print("\n   To enable live trading:")
    print("   1. Test thoroughly in paper mode")
    print("   2. Set MEXC_ENABLE_ORDERS=true in .env")
    print("   3. Restart your server")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)

if enabled.lower() != "true":
    print("\n💡 Tip: Start with paper trading (MEXC_ENABLE_ORDERS=false)")
    print("   Once confident, enable with MEXC_ENABLE_ORDERS=true")
print()

