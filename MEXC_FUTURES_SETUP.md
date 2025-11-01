# MEXC Futures Trading Setup Guide

Your trading bot now supports **automated MEXC futures trading** with 30x leverage on BTCUSDT and other pairs!

## 🚀 Quick Start

### 1. Get Your MEXC API Keys

1. Log into [MEXC](https://www.mexc.com/)
2. Go to **API Management**
3. Create a new API key (sub-account keys work!)
4. **Important**: Enable **Futures Trading** permission
5. **Optional**: Whitelist your IP for extra security
6. Save your API Key and Secret Key

### 2. Configure Environment Variables

Edit your `.env` file:

```bash
# MEXC Futures Trading
MEXC_API_KEY=your_api_key_here
MEXC_API_SECRET=your_secret_key_here

# Trading Parameters
MEXC_LEVERAGE=30                    # Leverage (1-200x)
MEXC_DEFAULT_SIZE=10.0             # Order size in USDT
MEXC_ENABLE_ORDERS=false           # IMPORTANT: false = paper trading
```

### 3. Test Your Setup

```bash
python test_mexc_futures.py
```

This will:
- ✅ Verify your API credentials
- ✅ Check your USDT balance
- ✅ Show any open positions
- ✅ Calculate position details
- ✅ Confirm safety settings

### 4. Start Paper Trading

With `MEXC_ENABLE_ORDERS=false` (default):
- Analysis runs normally
- Trade signals are generated
- Orders are **simulated but NOT placed**
- You see what WOULD have been traded

### 5. Enable Live Trading (When Ready)

```bash
# In .env file:
MEXC_ENABLE_ORDERS=true
```

**⚠️  WARNING: Real money will be traded!**

## 📊 How It Works

### Automatic Trading Flow

```
1. Upload Chart → Analysis runs
                 ↓
2. Signal Valid → Trade gate approves
                 ↓
3. If MEXC_ENABLE_ORDERS=true:
   ├─ Set 30x leverage
   ├─ Place market order
   ├─ Set stop loss
   └─ Set take profit(s)
                 ↓
4. Telegram notification with order details
```

### Order Details

**Symbol Format**: BTC_USDT (MEXC perpetual format)
**Order Type**: Market (instant execution)
**Margin Mode**: Isolated (safer - only position at risk)
**Leverage**: 30x (configurable 1-200x)

### Position Sizing Example

```
MEXC_DEFAULT_SIZE = $10.00 USDT
MEXC_LEVERAGE = 30x

Your margin: $10.00
Position value: $300.00 (10 × 30)
Max loss: $10.00 (if stop loss hit)
```

## ⚙️ Configuration Options

### Leverage Settings

```bash
# Conservative (lower risk)
MEXC_LEVERAGE=5

# Moderate (balanced)
MEXC_LEVERAGE=20

# Aggressive (higher risk)
MEXC_LEVERAGE=50

# Maximum (very risky!)
MEXC_LEVERAGE=125
```

**Recommendation**: Start with 10-30x until you're comfortable.

### Position Size

```bash
# Small positions (testing)
MEXC_DEFAULT_SIZE=5.0

# Medium positions
MEXC_DEFAULT_SIZE=25.0

# Large positions
MEXC_DEFAULT_SIZE=100.0
```

**Formula**: Your risk = MEXC_DEFAULT_SIZE
(This is your margin, not total position value)

## 🛡️ Risk Management

### Built-in Safety Features

✅ **Stop Loss**: Automatically placed on every trade
✅ **Take Profit**: Multiple TP levels set automatically
✅ **Isolated Margin**: Only your position margin at risk
✅ **Market Orders**: No slippage from failed limit orders
✅ **Paper Trading**: Test without risk first

### Manual Safety Checklist

Before enabling live trading:

- [ ] Tested with `MEXC_ENABLE_ORDERS=false`
- [ ] Verified API keys work with `test_mexc_futures.py`
- [ ] Understand 30x leverage means 30x gains AND losses
- [ ] Have sufficient USDT balance (> MEXC_DEFAULT_SIZE)
- [ ] Know how to manually close positions on MEXC
- [ ] Comfortable with your position size
- [ ] Stop loss levels make sense

## 📱 Telegram Notifications

You'll receive detailed notifications:

### When Order Placed
```
🚀 TRADE APPROVED: LONG

Symbol: BTCUSDT
Direction: LONG
Entry Price: $67,850.00
Stop Loss: $67,200.00
Take Profit 1: $68,500.00
Leverage: 30x
Position Size: $10.00 USDT
```

### Order Confirmation
```
✅ Order placed successfully!
Order ID: 123456789
Leverage: 30x
Stop Loss set at $67,200.00
Take Profit 1 set at $68,500.00
```

## 🧪 Testing Workflow

### Phase 1: Paper Trading (1-2 weeks)

```bash
MEXC_ENABLE_ORDERS=false
MEXC_DEFAULT_SIZE=10.0
```

- Monitor all signals
- Track hypothetical P&L
- Verify analysis quality
- Check stop loss placement

### Phase 2: Small Live Trades (1 week)

```bash
MEXC_ENABLE_ORDERS=true
MEXC_DEFAULT_SIZE=5.0      # Small size
MEXC_LEVERAGE=10           # Lower leverage
```

- Start with 5-10 USDT positions
- Use 10x leverage max
- Monitor closely
- Learn the execution

### Phase 3: Normal Trading

```bash
MEXC_ENABLE_ORDERS=true
MEXC_DEFAULT_SIZE=10.0     # Your comfort size
MEXC_LEVERAGE=30          # Your preference
```

## 🔍 Monitoring Positions

### Check via MEXC App/Website

1. Go to **Futures** tab
2. View **Positions**
3. See all open positions
4. Manually close if needed

### Via Python Script

```python
from pymexc import futures
import os

client = futures.HTTP(
    api_key=os.getenv("MEXC_API_KEY"),
    api_secret=os.getenv("MEXC_API_SECRET")
)

# Get open positions
positions = client.open_positions()
print(positions)

# Get account balance
balance = client.account_assets()
print(balance)
```

## ❌ Troubleshooting

### "API key invalid"
- Check API key and secret are correct
- Ensure Futures permission is enabled
- Try regenerating the API key

### "Insufficient balance"
- Check USDT balance in Futures account
- Transfer from Spot to Futures if needed
- Reduce MEXC_DEFAULT_SIZE

### "Leverage not allowed"
- Some pairs have max leverage limits
- Check MEXC for specific pair limits
- BTCUSDT usually allows up to 125x

### "Order placement failed"
- Check if market is open
- Verify symbol format (BTC_USDT)
- Ensure sufficient balance
- Check API key permissions

## 🎯 Example Scenarios

### Scenario 1: Conservative Trader

```bash
MEXC_LEVERAGE=10
MEXC_DEFAULT_SIZE=10.0
MEXC_ENABLE_ORDERS=true

Result:
- $10 margin = $100 position
- Max loss: $10
- Lower risk, lower reward
```

### Scenario 2: Balanced Trader

```bash
MEXC_LEVERAGE=30
MEXC_DEFAULT_SIZE=20.0
MEXC_ENABLE_ORDERS=true

Result:
- $20 margin = $600 position
- Max loss: $20
- Standard risk/reward
```

### Scenario 3: Aggressive Trader

```bash
MEXC_LEVERAGE=50
MEXC_DEFAULT_SIZE=50.0
MEXC_ENABLE_ORDERS=true

Result:
- $50 margin = $2,500 position
- Max loss: $50
- High risk, high reward
```

## 📚 Additional Resources

### MEXC Documentation
- [Futures API Docs](https://mexcdevelop.github.io/apidocs/contract_v1_en/)
- [Futures Trading Guide](https://www.mexc.com/support/sections/360000329711)

### Important Notes

1. **Tax Implications**: Keep records of all trades
2. **Risk Warning**: Leverage amplifies both gains and losses
3. **Start Small**: Test with amounts you can afford to lose
4. **Stop Losses**: Always use them - they're auto-set!
5. **Monitor Positions**: Check MEXC app regularly

## 🆘 Emergency Actions

### If Things Go Wrong

1. **Close All Positions**:
   - Go to MEXC Futures
   - Click "Close All"
   - Or close positions individually

2. **Disable Auto-Trading**:
   ```bash
   # In .env:
   MEXC_ENABLE_ORDERS=false
   ```
   Then restart server

3. **Revoke API Key**:
   - Go to MEXC API Management
   - Delete the API key
   - Bot can't trade without it

## ✅ Ready to Start?

1. Run test script: `python test_mexc_futures.py`
2. Verify everything is green ✅
3. Start with paper trading (`MEXC_ENABLE_ORDERS=false`)
4. Monitor for 1-2 weeks
5. Enable live trading when confident
6. Start small and scale up!

---

**Remember**: Trading with leverage is risky. Only trade with money you can afford to lose! 🚨

