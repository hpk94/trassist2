# How to Get a Hyperliquid Private Key

## Overview
Hyperliquid uses Ethereum-compatible wallets, so you can use any Ethereum wallet private key. The private key is a 64-character hexadecimal string (usually prefixed with `0x`).

## Option 1: Export from Hyperliquid Email Account (Easiest!)

If you created a Hyperliquid account using your email address, Hyperliquid automatically generates a non-custodial wallet for you. You can export the private key directly from the Hyperliquid interface.

### Steps to Export Private Key from Hyperliquid:

1. **Log in to Hyperliquid**
   - Visit [https://app.hyperliquid.xyz](https://app.hyperliquid.xyz)
   - Log in using your email address

2. **Access Settings**
   - Click on the **gear icon (⚙️)** or **Settings** menu
   - Usually located in the upper-right corner of the interface

3. **Export Your Private Key**
   - Look for **"Export Email Wallet"** or **"Export Private Key"** option
   - You may need to verify your identity (email confirmation, 2FA, etc.)
   - Follow the on-screen instructions
   - **Copy and securely save your private key**

4. **Security Warning**
   - ⚠️ **Never share this private key with anyone**
   - ⚠️ **Store it securely** (password manager, encrypted file, etc.)
   - ⚠️ **This key gives full access to your Hyperliquid wallet**

### Why This is the Easiest Option:
- ✅ No need to create a separate wallet
- ✅ Wallet is already connected to your Hyperliquid account
- ✅ No need to transfer funds between wallets
- ✅ Works immediately with your existing Hyperliquid balance

## Option 2: Export from Existing Wallet (MetaMask, Trust Wallet, etc.)

### MetaMask

1. **Open MetaMask** browser extension or mobile app
2. **Click on the account name** (top of MetaMask)
3. **Go to Account Details**
4. **Click "Export Private Key"**
5. **Enter your password** to confirm
6. **Copy the private key** (starts with `0x`)
7. **⚠️ SECURITY WARNING**: Never share this key or commit it to version control!

### Trust Wallet

1. **Open Trust Wallet** app
2. **Go to Settings** → **Wallets**
3. **Select your wallet**
4. **Tap "Show Private Key"**
5. **Enter your passcode/biometric**
6. **Copy the private key**

### Other Wallets
Most Ethereum wallets have a similar "Export Private Key" or "Show Private Key" option in their settings.

## Option 3: Create a New Wallet

### Using Python (Recommended for Testing)

Create a simple Python script:

```python
from eth_account import Account

# Create a new account
account = Account.create()

print(f"Address: {account.address}")
print(f"Private Key: {account.key.hex()}")
print(f"\n⚠️ IMPORTANT: Save this private key securely!")
print(f"⚠️ This is the ONLY time you'll see it!")
```

Run it:
```bash
python3 -c "from eth_account import Account; acc = Account.create(); print(f'Address: {acc.address}\nPrivate Key: {acc.key.hex()}')"
```

### Using Online Tools (NOT RECOMMENDED for Mainnet)
- ⚠️ **Security Risk**: Online tools can be compromised
- Only use for testnet wallets
- Examples: MyEtherWallet (offline mode), Vanity Address Generator

## Option 4: For Testnet Only

### Quick Testnet Wallet Generator

For testnet testing, you can use this Python script:

```python
#!/usr/bin/env python3
"""Generate a testnet wallet for Hyperliquid"""
from eth_account import Account

account = Account.create()

print("=" * 60)
print("TESTNET WALLET GENERATED")
print("=" * 60)
print(f"Address:     {account.address}")
print(f"Private Key: {account.key.hex()}")
print("=" * 60)
print("\n⚠️  This is for TESTNET only!")
print("⚠️  Never use testnet keys for mainnet!")
print("\nAdd to your .env file:")
print(f"HYPERLIQUID_PRIVATE_KEY={account.key.hex()}")
print(f"HYPERLIQUID_TESTNET=true")
print("\nGet testnet tokens from: https://hyperliquidfaucet.com/")
```

Save as `generate_testnet_wallet.py` and run:
```bash
python3 generate_testnet_wallet.py
```

## Format Requirements

Your private key should:
- Start with `0x` (optional, but recommended)
- Be 66 characters total (`0x` + 64 hex characters)
- Or 64 characters without the `0x` prefix

**Valid formats:**
```
0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
```

## Setting Up in Your .env File

Once you have your private key:

```bash
# For Mainnet
HYPERLIQUID_PRIVATE_KEY=0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef

# For Testnet
HYPERLIQUID_PRIVATE_KEY=0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
HYPERLIQUID_TESTNET=true
```

## Security Best Practices

### ⚠️ CRITICAL SECURITY WARNINGS

1. **Never Share Your Private Key**
   - Anyone with your private key has full control of your wallet
   - Never commit it to Git
   - Never share it in chat, email, or screenshots

2. **Use Separate Wallets**
   - Use a dedicated wallet for trading bots
   - Don't use your main wallet
   - Use different wallets for testnet and mainnet

3. **Start Small**
   - Only fund the wallet with what you need for trading
   - Don't store large amounts in a bot-controlled wallet

4. **Backup Securely**
   - Store private keys in a password manager
   - Use hardware wallets for large amounts
   - Consider using a multi-sig wallet for production

5. **Testnet vs Mainnet**
   - Always test on testnet first
   - Use separate private keys for testnet and mainnet
   - Testnet keys can be regenerated, mainnet keys cannot

## Verifying Your Private Key

Test that your private key works:

```bash
# Run the test script
python test_hyperliquid.py
```

You should see:
```
✅ Exchange client initialized successfully
   Wallet address: 0x...
✅ Account connected to Hyperliquid
```

## Getting Testnet Tokens

If you're using testnet:

1. Visit [Hyperliquid Faucet](https://hyperliquidfaucet.com/)
2. Enter your wallet address
3. Request testnet tokens
4. Wait for tokens to arrive (usually instant)

## Troubleshooting

### "Invalid Hyperliquid private key"
- Check that the key starts with `0x`
- Verify it's exactly 66 characters (with `0x`) or 64 characters (without)
- Make sure there are no extra spaces or newlines

### "HYPERLIQUID_PRIVATE_KEY not set"
- Check your `.env` file exists
- Verify the variable name is exactly `HYPERLIQUID_PRIVATE_KEY`
- Make sure there are no quotes around the value in `.env`

### "Failed to initialize Hyperliquid exchange"
- Verify your private key format is correct
- Check that you have the required packages: `pip install eth-account hyperliquid-python-sdk`
- For testnet, ensure `HYPERLIQUID_TESTNET=true` is set

## Which Option Should I Choose?

### ✅ **Recommended: Option 1 (Hyperliquid Email Account)**
- **Best for**: Most users, especially if you already have a Hyperliquid account
- **Pros**: Easiest, no wallet transfers needed, already connected
- **Cons**: None really, this is the simplest approach

### Option 2 (MetaMask/Trust Wallet)
- **Best for**: Users who want to use an existing wallet they already have
- **Pros**: Use wallet you're already familiar with
- **Cons**: Need to transfer funds to Hyperliquid

### Option 3 (Create New Wallet)
- **Best for**: Users who want a dedicated trading wallet
- **Pros**: Separate wallet for trading, better security isolation
- **Cons**: Need to fund the new wallet

### Option 4 (Testnet Only)
- **Best for**: Testing and development
- **Pros**: Free testnet tokens, no risk
- **Cons**: Only for testing, not real trading

## Quick Reference

```bash
# Generate a new testnet wallet
python3 -c "from eth_account import Account; acc = Account.create(); print(f'HYPERLIQUID_PRIVATE_KEY={acc.key.hex()}')"

# Test your configuration
python test_hyperliquid.py

# Check your wallet address
python3 -c "from eth_account import Account; import os; from dotenv import load_dotenv; load_dotenv(); acc = Account.from_key(os.getenv('HYPERLIQUID_PRIVATE_KEY')); print(f'Address: {acc.address}')"
```

