# Hyperliquid Testnet Setup

## Overview
This document describes how to use Hyperliquid testnet with the trading assistant application.

## Configuration

### Environment Variables

To enable Hyperliquid testnet, add the following to your `.env` file:

```bash
# Enable Hyperliquid testnet
HYPERLIQUID_TESTNET=true

# Your testnet wallet private key (starts with 0x)
HYPERLIQUID_PRIVATE_KEY=0x...

# Optional: Explicitly set base URL (overrides HYPERLIQUID_TESTNET)
# HYPERLIQUID_BASE_URL=https://api.hyperliquid-testnet.xyz
```

### Testnet Base URL
- **Testnet**: `https://api.hyperliquid-testnet.xyz`
- **Mainnet**: `None` (default, uses mainnet)

## How It Works

1. **Priority Order**:
   - If `HYPERLIQUID_BASE_URL` is set, it takes precedence
   - Otherwise, if `HYPERLIQUID_TESTNET=true`, testnet base URL is used
   - Otherwise, mainnet is used (default)

2. **Functions Updated**:
   - `_get_hyperliquid_base_url()`: Determines the correct base URL
   - `_init_hyperliquid_exchange()`: Uses testnet when enabled
   - `_init_hyperliquid_info()`: New helper for Info client with testnet support

## Testing

### Test Script
Run the test script to verify testnet connection:

```bash
# Set testnet in .env first
HYPERLIQUID_TESTNET=true

# Run test
python test_hyperliquid.py
```

The test script will:
- Show whether it's connecting to TESTNET or MAINNET
- Test Info client (public API)
- Test Exchange client (private API, if private key is set)

## Getting Testnet Tokens

See the detailed guide: [HOW_TO_GET_TESTNET_TOKENS.md](HOW_TO_GET_TESTNET_TOKENS.md)

**Quick Steps**:
1. Get your wallet address (from your private key)
2. Visit the [Hyperliquid Faucet](https://hyperliquidfaucet.com/)
3. Connect your wallet (add testnet network if needed)
4. Claim mock USDC (1,000 every 4 hours)
5. Trade for HYPE tokens on testnet if needed

**Alternative Faucets**:
- Community faucets (instant, smaller amounts)
- Chainstack faucet (requires 0.08 ETH on mainnet)
- QuickNode faucet (requires 0.001 ETH on mainnet)

For detailed instructions, see [HOW_TO_GET_TESTNET_TOKENS.md](HOW_TO_GET_TESTNET_TOKENS.md)

## Important Notes

⚠️ **Testnet vs Mainnet**:
- Testnet uses separate testnet tokens (no real value)
- Testnet data and positions are separate from mainnet
- Always verify you're on testnet before testing trading operations

⚠️ **Private Key Security**:
- Use a separate testnet wallet for testing
- Never use your mainnet private key on testnet
- Testnet private keys can be shared for testing (but still be cautious)

## Switching Between Testnet and Mainnet

To switch back to mainnet:
```bash
# Remove or set to false
HYPERLIQUID_TESTNET=false
# Or remove the variable entirely
```

## Code Changes

### New Functions
- `_get_hyperliquid_base_url()`: Centralized base URL determination
- `_init_hyperliquid_info()`: Info client initialization with testnet support

### Updated Functions
- `_init_hyperliquid_exchange()`: Now uses `_get_hyperliquid_base_url()`

### Updated Files
- `web_app.py`: Added testnet support functions
- `test_hyperliquid.py`: Updated to show testnet/mainnet status

