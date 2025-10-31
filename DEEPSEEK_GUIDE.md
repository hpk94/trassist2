# 🚀 Using DeepSeek with Your Trading Assistant

## Overview

DeepSeek is now supported! You can use DeepSeek models for **trade gate decisions** while using vision-capable models (like GPT-4o) for **chart analysis**. This gives you cost-effective performance where it matters.

---

## 💡 Why Use DeepSeek?

### Benefits
- ✅ **Very cost-effective** - Much cheaper than GPT-4o/Claude
- ✅ **Fast responses** - Quick decision-making
- ✅ **Good reasoning** - Strong performance on analytical tasks
- ✅ **Perfect for trade gates** - Excellent at making go/no-go decisions

### Limitations
- ❌ **No vision support** - Cannot analyze charts/images
- ⚠️ **Best for structured tasks** - Works great for trade gate decisions

---

## 🔧 Configuration Options

### Option 1: DeepSeek for Trade Gates Only (Recommended)

Use GPT-4o for chart analysis and DeepSeek for trade gates:

```bash
# .env file
LITELLM_VISION_MODEL=gpt-4o              # For chart analysis
LITELLM_TEXT_MODEL=deepseek/deepseek-chat # For trade gate decisions
OPENAI_API_KEY=sk-your-openai-key
DEEPSEEK_API_KEY=sk-your-deepseek-key
```

**This gives you:**
- Accurate chart analysis with GPT-4o vision
- Cost-effective trade decisions with DeepSeek
- Best balance of quality and cost

### Option 2: Mixed Providers

Use different providers for different tasks:

```bash
# .env file
LITELLM_VISION_MODEL=claude-3-5-sonnet-20241022  # Anthropic for charts
LITELLM_TEXT_MODEL=deepseek/deepseek-chat        # DeepSeek for gates
ANTHROPIC_API_KEY=sk-ant-your-key
DEEPSEEK_API_KEY=sk-your-deepseek-key
```

### Option 3: Single Model for Everything

If you prefer, you can still use one model for both:

```bash
# .env file
LITELLM_MODEL=gpt-4o  # Uses gpt-4o for both vision and text
OPENAI_API_KEY=sk-your-key
```

---

## 📋 Available DeepSeek Models

### DeepSeek Chat (Recommended)
```bash
LITELLM_TEXT_MODEL=deepseek/deepseek-chat
```
- General purpose model
- Best for trade gate decisions
- Good reasoning capabilities

### DeepSeek Coder
```bash
LITELLM_TEXT_MODEL=deepseek/deepseek-coder
```
- Optimized for code-related tasks
- Can be used if your prompts involve technical analysis

### DeepSeek V3 (if available)
```bash
LITELLM_TEXT_MODEL=deepseek/deepseek-v3
```
- Latest version
- Check DeepSeek docs for availability

---

## 🎯 Complete Setup Guide

### Step 1: Get DeepSeek API Key

1. Visit [DeepSeek Platform](https://platform.deepseek.com)
2. Sign up and get your API key
3. Copy your API key (starts with `sk-`)

### Step 2: Update Your `.env` File

```bash
# Vision model for chart analysis
LITELLM_VISION_MODEL=gpt-4o

# Text model for trade gate decisions
LITELLM_TEXT_MODEL=deepseek/deepseek-chat

# API Keys
OPENAI_API_KEY=sk-your-openai-key-here
DEEPSEEK_API_KEY=sk-your-deepseek-key-here

# Your existing MEXC config
MEXC_API_KEY=...
MEXC_API_SECRET=...
```

### Step 3: Test It

```bash
# Run your trading assistant
python app.py
```

You should see in the logs:
- Chart analysis using `gpt-4o`
- Trade gate decision using `deepseek/deepseek-chat`

---

## 💰 Cost Comparison

### Per 1M Tokens

| Task | Model | Cost | Monthly Est.* |
|------|-------|------|---------------|
| **Chart Analysis** | GPT-4o | ~$6 | ~$18 |
| **Trade Gate** | DeepSeek | ~$0.14 | ~$0.42 |
| **Total** | Mixed | ~$6.14 | ~$18.42 |

vs.

| Task | Model | Cost | Monthly Est.* |
|------|-------|------|---------------|
| **Both** | GPT-4o only | ~$6 | ~$36 |

**Savings: ~50%** by using DeepSeek for trade gates!

*Based on ~100 chart analyses/day, assuming 2k tokens for chart, 500 tokens for gate

---

## 🔄 How It Works

### With Dual Models (Recommended)

```
1. Trading Chart Image
   ↓
2. GPT-4o Vision analyzes chart
   ↓
3. Market data validation
   ↓
4. DeepSeek makes gate decision
   ↓
5. Trade execution (if approved)
```

### Code Flow

```python
# Chart analysis - uses LITELLM_VISION_MODEL (gpt-4o)
llm_output = analyze_trading_chart(test_image)

# Market validation (programmatic)
signal_valid, status, conditions, market_values = poll_until_decision(...)

# Trade gate - uses LITELLM_TEXT_MODEL (deepseek)
gate_result = llm_trade_gate_decision(
    llm_output, market_values, 
    checklist_passed, invalidation_triggered, conditions
)
```

---

## 🧪 Testing

### Test Vision Model

```bash
# Test chart analysis
python -c "
from app import analyze_trading_chart
result = analyze_trading_chart('BTCUSDT.P_2025-09-02_22-55-40_545b4.png')
print('Chart analysis model:', result.get('model_used', 'check logs'))
"
```

### Test Text Model

Create a test script:

```python
# test_gate.py
import os
from dotenv import load_dotenv
import litellm

load_dotenv()
text_model = os.getenv("LITELLM_TEXT_MODEL", "gpt-4o")

response = litellm.completion(
    model=text_model,
    messages=[{"role": "user", "content": "Say 'Trade gate model working' if you can read this."}]
)

print(f"Text model: {text_model}")
print(f"Response: {response.choices[0].message.content}")
```

Run it:
```bash
python test_gate.py
```

---

## ⚙️ Advanced Configurations

### Use DeepSeek for Development, GPT-4o for Production

Development `.env`:
```bash
LITELLM_VISION_MODEL=gpt-4o
LITELLM_TEXT_MODEL=deepseek/deepseek-chat  # Cheaper for testing
```

Production `.env`:
```bash
LITELLM_VISION_MODEL=gpt-4o
LITELLM_TEXT_MODEL=gpt-4o  # More conservative for live trading
```

### Use Gemini + DeepSeek (Ultra Budget)

```bash
LITELLM_VISION_MODEL=gemini/gemini-1.5-pro  # $3/1M tokens
LITELLM_TEXT_MODEL=deepseek/deepseek-chat   # $0.14/1M tokens
GEMINI_API_KEY=your-gemini-key
DEEPSEEK_API_KEY=your-deepseek-key
```

**Ultra-low cost!** ~$3.14/1M tokens total

---

## 🚨 Troubleshooting

### Error: "deepseek not supported"

Make sure you're using the correct model name:
```bash
# ✅ Correct
LITELLM_TEXT_MODEL=deepseek/deepseek-chat

# ❌ Wrong
LITELLM_TEXT_MODEL=deepseek-chat  # Missing prefix
```

### Error: "Authentication failed"

Check your API key:
```bash
# In .env file
DEEPSEEK_API_KEY=sk-your-actual-key-here

# NOT
DEEPSEEK_API_KEY=your-deepseek-key  # This is a placeholder
```

### DeepSeek used for chart analysis (wrong!)

Make sure you set the vision model:
```bash
# .env file must have:
LITELLM_VISION_MODEL=gpt-4o  # Vision-capable model

# If you only set LITELLM_TEXT_MODEL, vision will use default
```

### Both using same model

If you want different models for each task, set **both**:
```bash
LITELLM_VISION_MODEL=gpt-4o
LITELLM_TEXT_MODEL=deepseek/deepseek-chat
```

If you set only `LITELLM_MODEL`, both use that model:
```bash
LITELLM_MODEL=gpt-4o  # Both vision and text use gpt-4o
```

---

## 📊 Performance Comparison

### Chart Analysis + Trade Gate (100 trades)

| Configuration | Cost | Chart Quality | Gate Quality | Total |
|--------------|------|---------------|--------------|-------|
| GPT-4o + GPT-4o | $36 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GPT-4o + DeepSeek | $18.42 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Claude + DeepSeek | $27.42 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Gemini + DeepSeek | $9.42 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**Recommendation:** GPT-4o + DeepSeek for best balance

---

## 🎯 Summary

### Best Practices

1. ✅ **Use GPT-4o for chart analysis** (vision required)
2. ✅ **Use DeepSeek for trade gates** (cost-effective)
3. ✅ **Test both models separately** before live trading
4. ✅ **Monitor quality** of trade gate decisions
5. ✅ **Keep vision model high-quality** (chart analysis is critical)

### Quick Config

```bash
# Recommended .env configuration
LITELLM_VISION_MODEL=gpt-4o
LITELLM_TEXT_MODEL=deepseek/deepseek-chat
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
MEXC_API_KEY=...
MEXC_API_SECRET=...
```

### One-Line Test

```bash
echo "LITELLM_VISION_MODEL=gpt-4o
LITELLM_TEXT_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=sk-your-key" >> .env && python app.py
```

---

## 📚 Related Documentation

- **LiteLLM Models**: [`LITELLM_CONFIGURATION.md`](LITELLM_CONFIGURATION.md)
- **Quick Start**: [`QUICKSTART.md`](QUICKSTART.md)
- **Get Started**: [`GET_STARTED.md`](GET_STARTED.md)
- **Main README**: [`README.md`](README.md)

---

**Ready to save costs with DeepSeek?** Set it up and start trading! 🚀📈

