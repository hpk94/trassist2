# Environment Variables Migration Guide

## MEXC to Hyperliquid Migration

### Variables to Remove
```bash
# Remove these from your .env file:
MEXC_API_KEY=...
MEXC_API_SECRET=...
MEXC_ENABLE_ORDERS=...
```

### Variables to Add
```bash
# Add these to your .env file:

# Your Ethereum wallet private key (starts with 0x)
HYPERLIQUID_PRIVATE_KEY=0x...

# Enable/disable live trading (default: false)
HYPERLIQUID_ENABLE_ORDERS=false

# Optional: Order size in USD (default: 10.0)
HYPERLIQUID_DEFAULT_SIZE=10.0

# Optional: Leverage for perpetual positions (default: 1)
HYPERLIQUID_LEVERAGE=1
```

### Complete .env Example
```bash
# LiteLLM / OpenAI Configuration
LITELLM_VISION_MODEL=gpt-4o
LITELLM_TEXT_MODEL=gpt-4o
OPENAI_API_KEY=sk-...

# Hyperliquid Configuration
HYPERLIQUID_PRIVATE_KEY=0x...
HYPERLIQUID_ENABLE_ORDERS=false
HYPERLIQUID_DEFAULT_SIZE=10.0
HYPERLIQUID_LEVERAGE=1

# Telegram Notifications
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Optional: Email Notifications
EMAIL_FROM=...
EMAIL_TO=...
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=...
SMTP_PASSWORD=...

# Optional: Pushover Notifications
PUSHOVER_USER_KEY=...
PUSHOVER_API_TOKEN=...
```

## Important Security Notes

⚠️ **HYPERLIQUID_PRIVATE_KEY** contains your wallet's private key:
- Keep it secret and never commit to version control
- This key has full access to your Hyperliquid account
- Consider using a dedicated wallet for trading
- Start with small amounts for testing

⚠️ **HYPERLIQUID_ENABLE_ORDERS** safety:
- Keep this set to `false` during testing
- Only set to `true` when you're ready for live trading
- The application will only execute trades when this is `true`

## Getting Your Hyperliquid Private Key

1. **If you have an existing Ethereum wallet**:
   - Export your private key from MetaMask, Trust Wallet, etc.
   - Format: `0x` followed by 64 hexadecimal characters
   
2. **If you need to create a new wallet**:
   ```python
   from eth_account import Account
   account = Account.create()
   print(f"Address: {account.address}")
   print(f"Private Key: {account.key.hex()}")
   ```

3. **Fund your wallet**:
   - Transfer USDC to your wallet address on Arbitrum network
   - Hyperliquid operates on Arbitrum L2

## Testing Your Configuration

```bash
# Test Hyperliquid API connection
python test_hyperliquid.py

# Run full diagnostics
python diagnose_analysis.py

# Start the web application
python web_app.py
```

## Troubleshooting

### "Import hyperliquid could not be resolved"
```bash
pip install hyperliquid-python-sdk eth-account
```

### "No private key for trading"
- Verify HYPERLIQUID_PRIVATE_KEY is set in .env
- Ensure it starts with `0x`
- Check there are no extra spaces or quotes

### "Failed to init exchange client"
- Verify private key format is correct
- Check your internet connection
- Ensure Arbitrum network is accessible

### Trading not executing
- Check HYPERLIQUID_ENABLE_ORDERS is set to "true"
- Verify your wallet has sufficient USDC balance
- Check logs for specific error messages

## Migration Checklist

- [ ] Create backup of current .env file
- [ ] Remove MEXC_* variables
- [ ] Add HYPERLIQUID_* variables
- [ ] Set HYPERLIQUID_ENABLE_ORDERS=false initially
- [ ] Test with `python test_hyperliquid.py`
- [ ] Test with `python diagnose_analysis.py`
- [ ] Upload a test chart via web interface
- [ ] Verify analysis completes successfully
- [ ] Only after validation, set HYPERLIQUID_ENABLE_ORDERS=true if desired

---

**Last Updated:** October 31, 2025

