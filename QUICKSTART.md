# Quick Start Guide - LiteLLM Integration

## TL;DR - Get Running in 2 Minutes

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your model in .env
echo "LITELLM_MODEL=gpt-4o" >> .env
echo "OPENAI_API_KEY=sk-your-key-here" >> .env

# 3. Run
python3 app.py
```

## What Changed?

Your code now uses **LiteLLM** instead of direct OpenAI calls. This means:

✅ **Same functionality** - Everything works as before  
✅ **More flexibility** - Easily switch between OpenAI, Claude, Gemini, etc.  
✅ **No code changes needed** - Just change environment variables

## Switching Models

### Option 1: Edit `.env` file (Permanent)
```bash
LITELLM_MODEL=gpt-4o                      # OpenAI (default)
# LITELLM_MODEL=claude-3-5-sonnet-20241022  # Anthropic
# LITELLM_MODEL=gemini/gemini-1.5-pro       # Google
```

### Option 2: Command line (One-time)
```bash
LITELLM_MODEL=claude-3-5-sonnet-20241022 python3 app.py
```

## Required Environment Variables

```bash
# Model selection
LITELLM_MODEL=gpt-4o

# API Key (one of these, matching your model)
OPENAI_API_KEY=sk-...           # For OpenAI models
# or
ANTHROPIC_API_KEY=sk-ant-...    # For Claude models
# or
GEMINI_API_KEY=...              # For Gemini models

# MEXC (unchanged)
MEXC_API_KEY=...
MEXC_API_SECRET=...
```

## Vision-Capable Models (for Chart Analysis)

Only these models work with image analysis:

| Model | Provider | Recommended |
|-------|----------|-------------|
| `gpt-4o` | OpenAI | ✅ **Best** |
| `gpt-4-turbo` | OpenAI | ✅ Good |
| `claude-3-5-sonnet-20241022` | Anthropic | ✅ Good |
| `claude-3-opus-20240229` | Anthropic | ✅ Good |
| `gemini/gemini-1.5-pro` | Google | ✅ Cost-effective |

## Common Errors

### "litellm could not be resolved"
```bash
pip install litellm
```

### "AuthenticationError"
Check you have the right API key in `.env`:
- OpenAI models need `OPENAI_API_KEY`
- Claude models need `ANTHROPIC_API_KEY`
- Gemini models need `GEMINI_API_KEY`

### "Model not found"
Check the model name format:
- Gemini needs `gemini/` prefix: `gemini/gemini-1.5-pro`
- Azure needs `azure/` prefix: `azure/gpt-4o`
- Ollama needs `ollama/` prefix: `ollama/llama2`

## Cost Comparison

Per 1M tokens (approximate):

| Model | Input | Output | Total (typical) |
|-------|-------|--------|-----------------|
| GPT-4o | $2.50 | $10.00 | ~$6 |
| Claude 3.5 Sonnet | $3.00 | $15.00 | ~$9 |
| Gemini 1.5 Pro | $1.25 | $5.00 | ~$3 |

💡 **Tip**: Use cheaper models for testing, production models for live trading

## Testing Different Models

Compare results from different models:

```bash
# Test with GPT-4o
LITELLM_MODEL=gpt-4o python3 app.py

# Test with Claude
LITELLM_MODEL=claude-3-5-sonnet-20241022 python3 app.py

# Test with Gemini
LITELLM_MODEL=gemini/gemini-1.5-pro python3 app.py

# Compare outputs in llm_outputs/ directory
```

## Need More Info?

- **Full setup guide**: [`LITELLM_CONFIGURATION.md`](LITELLM_CONFIGURATION.md)
- **Migration details**: [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md)
- **Main README**: [`README.md`](README.md)

## Still Using Old OpenAI Code?

If you haven't updated yet:

```bash
# Backup current code
git commit -am "Backup before LiteLLM migration"

# Pull latest changes
git pull

# Install new dependencies
pip install -r requirements.txt

# Add LITELLM_MODEL to .env
echo "LITELLM_MODEL=gpt-4o" >> .env

# Run!
python3 app.py
```

## Rollback (if needed)

```bash
git checkout HEAD~1 app.py requirements.txt
pip install -r requirements.txt
```

---

**That's it!** 🎉 You're now using LiteLLM and can switch between AI models instantly.

