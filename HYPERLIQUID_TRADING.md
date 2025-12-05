# Hyperliquid Trading Integration

This document describes the Hyperliquid trading integration for the Trading Assistant.

## Overview

The system uses **Hyperliquid** for executing trades with the following key features:

- **Limit Orders Only**: All trades are executed using limit orders to minimize fees
- **MEXC Price Monitoring**: Since analysis is based on MEXC BTC/USDT-P, the system monitors MEXC prices to manage trades
- **Leverage Management**: Automatic leverage checking and configuration
- **Telegram Trade Buttons**: Interactive buttons for managing trades directly from Telegram

## Important Note: USDT vs USDC

- **TradingView Analysis**: Based on MEXC BTC/USDT-P (USDT margined)
- **Hyperliquid Trading**: Uses USDC as collateral

The system monitors both exchanges and alerts you when there's a significant price difference (>0.1%) between them.

## Environment Variables

Add these to your `.env` file:

```bash
# === Hyperliquid Configuration ===

# Your Ethereum wallet private key (required for trading)
HYPERLIQUID_PRIVATE_KEY=0x...

# Enable/disable live trading (default: false)
# Set to "true" to enable actual order placement
HYPERLIQUID_ENABLE_ORDERS=false

# Network selection
# Set to "true" for testnet, "false" for mainnet
HYPERLIQUID_TESTNET=false

# Or explicitly set the base URL (overrides HYPERLIQUID_TESTNET)
# HYPERLIQUID_BASE_URL=https://api.hyperliquid-testnet.xyz

# === Order Configuration ===

# Default position size in coin units (e.g., 0.001 BTC)
HYPERLIQUID_DEFAULT_SIZE=0.001

# Leverage setting (default: 1)
HYPERLIQUID_LEVERAGE=1

# Maximum allowed leverage (safety limit, default: 10)
HYPERLIQUID_MAX_LEVERAGE=10

# Slippage for limit orders (as decimal, 0.001 = 0.1%)
HYPERLIQUID_SLIPPAGE=0.001

# Time-in-force for limit orders: Gtc, Ioc, or Alo (default: Gtc)
HYPERLIQUID_LIMIT_TIF=Gtc

# Slippage for close orders (as decimal, default: 0.002)
HYPERLIQUID_CLOSE_SLIPPAGE=0.002

# === Optional Advanced Settings ===

# Skip WebSocket connection (default: true)
HYPERLIQUID_SKIP_WS=true

# Custom account address (for sub-accounts)
# HYPERLIQUID_ACCOUNT_ADDRESS=0x...

# Vault address (for vault trading)
# HYPERLIQUID_VAULT_ADDRESS=0x...
```

## Telegram Trade Management

When a trade is opened, you'll receive a Telegram notification with interactive buttons:

### Trade Notification Buttons

| Button | Action |
|--------|--------|
| 📊 Check P&L | Shows current profit/loss |
| 📈 Prices | Shows Hyperliquid vs MEXC prices |
| 🔒 Close 50% | Closes half the position (limit order) |
| 🔒 Close 100% | Closes entire position (limit order) |
| 🚨 Emergency Close | Market order close (higher fees) |

### New Bot Commands

| Command | Description |
|---------|-------------|
| `/positions` | View all open positions with close buttons |
| `/close` | Close all positions (with confirmation) |
| `/status` | Check analysis status |
| `/stop` | Stop running analysis |
| `/help` | Show available commands |

## How It Works

### 1. Trade Execution Flow

1. Chart analysis identifies a trade signal
2. Trade gate approves the trade
3. System checks price difference between MEXC and Hyperliquid
4. Leverage is verified/set according to configuration
5. **Limit order** is placed on Hyperliquid
6. Telegram notification with management buttons is sent
7. Position monitor starts tracking the trade

### 2. Position Monitoring

The position monitor runs in the background and:
- Checks positions at regular intervals
- Monitors MEXC prices (since analysis was based on MEXC)
- Alerts when stop loss or take profit levels are hit (based on MEXC price)
- Allows you to close trades via Telegram buttons

### 3. Limit Orders vs Market Orders

| Order Type | Use Case | Fees |
|------------|----------|------|
| Limit | Normal opens/closes | Lower (maker) |
| Market | Emergency close only | Higher (taker) |

The system defaults to limit orders for all operations. Market orders are only used for emergency closes when immediate execution is required.

## Price Difference Handling

Since MEXC uses USDT and Hyperliquid uses USDC:

1. **Price Comparison**: Before each trade, the system fetches prices from both exchanges
2. **Difference Alert**: If difference > 0.1%, a warning is logged/shown
3. **Stop Loss Monitoring**: Stop loss checks use MEXC prices (matching your analysis)
4. **Limit Price Calculation**: Based on Hyperliquid mid price + slippage

Example log output:
```
Hyperliquid: Checking price difference (MEXC USDT vs Hyperliquid USDC)...
   HL (USDC): $97,500.00 | MEXC (USDT): $97,450.00 | Diff: 0.0513%
```

## Safety Features

1. **Orders Disabled by Default**: `HYPERLIQUID_ENABLE_ORDERS=false`
2. **Maximum Leverage Limit**: Configurable safety cap
3. **Testnet Support**: Test with fake funds first
4. **Confirmation for Close All**: Telegram asks for confirmation

## Quick Start

1. **Set up your private key:**
   ```bash
   # Add to .env
   HYPERLIQUID_PRIVATE_KEY=0x...
   ```

2. **Test on testnet first:**
   ```bash
   HYPERLIQUID_TESTNET=true
   ```

3. **Enable orders when ready:**
   ```bash
   HYPERLIQUID_ENABLE_ORDERS=true
   ```

4. **Configure position size:**
   ```bash
   HYPERLIQUID_DEFAULT_SIZE=0.001  # 0.001 BTC
   HYPERLIQUID_LEVERAGE=1          # 1x leverage
   ```

## Testing the Service

Run the service status check:
```bash
python services/hyperliquid_service.py
```

This will show:
- Configuration status
- Account balance
- Open positions
- Price comparison

## Troubleshooting

### "Orders not enabled"
Set `HYPERLIQUID_ENABLE_ORDERS=true` in your `.env` file.

### "Exchange client not initialized"
Check that `HYPERLIQUID_PRIVATE_KEY` is set correctly.

### "Could not determine limit price"
The system couldn't fetch the current market price. Check network connectivity.

### Large price difference
If you see >0.5% difference between MEXC and Hyperliquid, there may be a stablecoin depeg. Exercise caution.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Chart Upload → LLM Analysis → Trade Signal               │
│                         ↓                                     │
│  2. MEXC Price Data → Gate Decision → Approved/Rejected      │
│                         ↓                                     │
│  3. Price Comparison (MEXC USDT vs Hyperliquid USDC)         │
│                         ↓                                     │
│  4. Hyperliquid LIMIT Order → Position Opened                │
│                         ↓                                     │
│  5. Telegram Notification + Buttons                          │
│                         ↓                                     │
│  6. Position Monitor (MEXC price based)                      │
│                         ↓                                     │
│  7. Close via Telegram Button (LIMIT order)                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
services/
├── hyperliquid_service.py    # Main trading service
├── notification_service.py   # Telegram notifications
└── indicator_service.py      # Technical indicators

telegram_bot.py               # Bot with inline buttons
app.py                        # Main pipeline (uses Hyperliquid)
web_app.py                    # Web interface (uses Hyperliquid)
```

