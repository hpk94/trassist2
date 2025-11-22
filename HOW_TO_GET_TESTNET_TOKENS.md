# How to Get Hyperliquid Testnet Tokens

## Overview
To test your trading bot on Hyperliquid testnet, you need testnet tokens. These are free tokens with no real value, used only for testing.

## Step 1: Get Your Wallet Address

First, you need your wallet address. You can get it from your private key:

```bash
# Using Python (in your venv)
source venv/bin/activate
python3 -c "from eth_account import Account; import os; from dotenv import load_dotenv; load_dotenv(); acc = Account.from_key(os.getenv('HYPERLIQUID_PRIVATE_KEY')); print(f'Wallet Address: {acc.address}')"
```

Or if you're using a Hyperliquid email account, you can find it in the Hyperliquid app settings.

## Step 2: Choose a Faucet

### Option 1: Official Hyperliquid Faucet (Recommended)

**URL**: [https://hyperliquidfaucet.com/](https://hyperliquidfaucet.com/)

**Steps**:
1. Visit the faucet website
2. **Connect your wallet** to Hyperliquid testnet
   - You may need to add the testnet network to your wallet first (see below)
3. **Claim Mock USDC**: Request 1,000 mock USDC
   - Can claim every 4 hours
4. **Trade for HYPE tokens**: Use the mock USDC to purchase HYPE tokens on the testnet platform

**Pros**: Official, reliable, gives you mock USDC to trade with
**Cons**: 4-hour cooldown between claims

### Option 2: Community Faucet (Faster)

**URL**: Community faucets (search for "Hyperliquid testnet faucet")

**Steps**:
1. Visit the community faucet
2. Connect your wallet
3. Claim HYPE tokens directly (usually 0.1 HYPE instantly)

**Pros**: Instant, no waiting
**Cons**: Smaller amounts, may require mainnet ETH balance

### Option 3: Chainstack Faucet

**URL**: [https://faucet.chainstack.com/hyperliquid-testnet-faucet](https://faucet.chainstack.com/hyperliquid-testnet-faucet)

**Requirements**:
- Must hold at least 0.08 ETH on Ethereum mainnet
- Can claim up to 1 HYPE token every 24 hours

**Steps**:
1. Enter your wallet address
2. Verify eligibility
3. Claim tokens

### Option 4: QuickNode Faucet

**URL**: [https://faucet.quicknode.com/hyperliquid](https://faucet.quicknode.com/hyperliquid)

**Requirements**:
- Must hold at least 0.001 ETH on Ethereum mainnet

**Steps**:
1. Connect your wallet
2. Claim testnet HYPE tokens

## Step 3: Add Hyperliquid Testnet to Your Wallet (If Needed)

If you're using MetaMask or another wallet, you may need to add the testnet network:

### MetaMask:
1. Open MetaMask
2. Click network dropdown (top)
3. Click "Add Network" → "Add a network manually"
4. Enter these details:
   - **Network Name**: Hyperliquid Testnet
   - **RPC URL**: `https://api.hyperliquid-testnet.xyz`
   - **Chain ID**: `998`
   - **Currency Symbol**: `HYPE`
   - **Block Explorer**: `https://testnet.purrsec.com` (optional)

### Other Wallets:
Similar process - add custom network with the RPC URL: `https://api.hyperliquid-testnet.xyz`

## Step 4: Verify You Have Tokens

After claiming tokens, verify they're in your wallet:

```bash
# Test your connection
source venv/bin/activate
python test_hyperliquid.py
```

The test should show your wallet address and confirm connection. You can also check your balance in the Hyperliquid testnet app.

## Quick Start Guide

1. **Get your wallet address** (from your private key or Hyperliquid account)
2. **Visit**: [https://hyperliquidfaucet.com/](https://hyperliquidfaucet.com/)
3. **Connect wallet** (add testnet network if needed)
4. **Claim mock USDC** (1,000 every 4 hours)
5. **Trade for HYPE** on testnet if needed
6. **Test your bot** with `python test_hyperliquid.py`

## Troubleshooting

### "Network not found" or "Wrong network"
- Make sure you've added Hyperliquid testnet to your wallet
- Verify you're connected to the testnet network (not mainnet)
- Check that `HYPERLIQUID_TESTNET=true` is set in your `.env`

### "Insufficient funds" error
- Make sure you've claimed tokens from the faucet
- Wait for the transaction to confirm (usually instant)
- Try a different faucet if one isn't working

### "Wallet not eligible"
- Some faucets require you to hold ETH on mainnet
- Try the official Hyperliquid faucet which has fewer restrictions
- Make sure your wallet address is correct

### "Can't connect wallet"
- Make sure your wallet is unlocked
- Try refreshing the faucet page
- Check that you're using a compatible wallet (MetaMask, WalletConnect, etc.)

## Important Notes

⚠️ **Testnet tokens have NO real value** - they're only for testing
⚠️ **Testnet and mainnet are separate** - testnet tokens won't work on mainnet
⚠️ **Faucets have limits** - usually daily/hourly limits on how much you can claim
⚠️ **Always verify you're on testnet** - double-check before trading

## Need More Tokens?

If you need more testnet tokens:
- Wait for the cooldown period (usually 4-24 hours depending on faucet)
- Try multiple faucets
- Ask in Hyperliquid community channels (Discord, Telegram)

## Testing Your Setup

Once you have testnet tokens:

```bash
# 1. Make sure testnet is enabled
echo "HYPERLIQUID_TESTNET=true" >> .env

# 2. Test connection
python test_hyperliquid.py

# 3. You should see:
# ✅ Exchange client initialized successfully (TESTNET)
#    Wallet address: 0x...
# ✅ Account connected to Hyperliquid TESTNET
```

If you see these messages, you're ready to test your trading bot on testnet! 🚀

