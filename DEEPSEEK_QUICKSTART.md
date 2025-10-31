# 🚀 DeepSeek Quick Start (30 seconds)

## TL;DR

Save 50% on costs by using DeepSeek for trade gate decisions!

```bash
# Add to .env
LITELLM_VISION_MODEL=gpt-4o
LITELLM_TEXT_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=sk-your-deepseek-key-here

# Run
python app.py
```

---

## What You Get

✅ **GPT-4o** analyzes charts (high quality, vision)  
✅ **DeepSeek** makes trade decisions (low cost, fast)  
✅ **50% cost savings** compared to using GPT-4o for everything  

---

## Config Comparison

### Before (GPT-4o only)
```bash
LITELLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```
💰 Cost: ~$36/month (100 trades)

### After (GPT-4o + DeepSeek)
```bash
LITELLM_VISION_MODEL=gpt-4o
LITELLM_TEXT_MODEL=deepseek/deepseek-chat
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
```
💰 Cost: ~$18/month (100 trades)

**Savings: $18/month! 💸**

---

## Get DeepSeek API Key

1. Visit: https://platform.deepseek.com
2. Sign up
3. Get API key (starts with `sk-`)
4. Add to `.env`

---

## How It Works

```
Chart Image → GPT-4o (vision) → Analysis
                                   ↓
Market Data → Validation → DeepSeek (text) → Trade Decision
                                                ↓
                                              Execute
```

- **Chart analysis**: GPT-4o (needs vision)
- **Trade gate**: DeepSeek (text only, cheaper)

---

## Test It

```bash
# Check which models are being used
python app.py

# You should see in the logs:
# "Chart: Sending request to LLM Vision API (model: gpt-4o)..."
# "Gate: Sending request to LLM for trade gate decision (model: deepseek/deepseek-chat)..."
```

---

## Full Guide

For complete details, see [`DEEPSEEK_GUIDE.md`](DEEPSEEK_GUIDE.md)

---

**Ready?** Add your DeepSeek API key and start saving! 🎉

