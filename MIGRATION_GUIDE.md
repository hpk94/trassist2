# Migration Guide: OpenAI → LiteLLM

This guide walks you through migrating from the direct OpenAI client to LiteLLM.

## What Changed?

### Code Changes
1. **Import statement**: `from openai import OpenAI` → `import litellm`
2. **Client initialization**: Removed `client = OpenAI()` calls
3. **API calls**: `client.chat.completions.create()` → `litellm.completion()`
4. **Model configuration**: Now uses `LITELLM_MODEL` environment variable

### Files Modified
- `app.py`: Updated to use LiteLLM
- `requirements.txt`: Replaced `openai` with `litellm`

## Migration Steps

### 1. Install LiteLLM

Activate your virtual environment and install the new dependency:

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Update Environment Variables

Create or update your `.env` file with:

```bash
# Add this new variable
LITELLM_MODEL=gpt-4o

# Keep your existing OpenAI key
OPENAI_API_KEY=sk-...

# All other existing variables remain the same
MEXC_API_KEY=...
MEXC_API_SECRET=...
```

### 3. Test the Migration

Run a test to verify everything works:

```bash
python app.py
```

The app should work exactly as before, but now you can easily switch models!

## Benefits of LiteLLM

### 1. **Easy Model Switching**
Change models without code changes:
```bash
# Use Claude instead
LITELLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...

# Use Gemini
LITELLM_MODEL=gemini/gemini-1.5-pro
GEMINI_API_KEY=...
```

### 2. **Cost Optimization**
Test with cheaper models for development:
```bash
# Development/testing
LITELLM_MODEL=gpt-3.5-turbo

# Production
LITELLM_MODEL=gpt-4o
```

### 3. **Automatic Fallbacks**
LiteLLM can automatically fall back to alternative models if your primary model fails or hits rate limits.

### 4. **Local Models Support**
Run models locally with Ollama:
```bash
LITELLM_MODEL=ollama/llama2
```

### 5. **Unified Interface**
Same API calls work across all providers - no vendor lock-in!

## Switching Between Models

### For Chart Analysis (Vision Required)

Best options:
```bash
# OpenAI (default, recommended)
LITELLM_MODEL=gpt-4o

# Anthropic Claude (good alternative)
LITELLM_MODEL=claude-3-5-sonnet-20241022

# Google Gemini (cost-effective)
LITELLM_MODEL=gemini/gemini-1.5-pro
```

### For Trade Gate Decisions (Text Only)

You can use any model:
```bash
# High quality
LITELLM_MODEL=gpt-4o

# Balanced
LITELLM_MODEL=gpt-4-turbo

# Cost-effective
LITELLM_MODEL=gpt-3.5-turbo
```

## Rollback Instructions

If you need to rollback to OpenAI:

```bash
# 1. Uninstall litellm
pip uninstall litellm

# 2. Install openai
pip install openai==1.106.1

# 3. Restore the original code
git checkout app.py requirements.txt
```

## Common Issues

### Issue: "AuthenticationError"
**Solution**: Make sure you've set the correct API key for your chosen provider:
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GEMINI_API_KEY`

### Issue: "Model not found"
**Solution**: Check the model name format. Some providers need prefixes:
- Gemini: `gemini/gemini-1.5-pro`
- Azure: `azure/gpt-4o`
- Ollama: `ollama/llama2`

### Issue: Vision/image not working
**Solution**: Ensure you're using a vision-capable model. See `LITELLM_CONFIGURATION.md` for the list.

## Support

- Full documentation: `LITELLM_CONFIGURATION.md`
- LiteLLM docs: https://docs.litellm.ai/
- Issues: Check your `.env` file first, then API keys

## Testing Different Models

Try comparing results from different models:

```bash
# Test 1: GPT-4o (baseline)
LITELLM_MODEL=gpt-4o python app.py

# Test 2: Claude 3.5 Sonnet
LITELLM_MODEL=claude-3-5-sonnet-20241022 python app.py

# Test 3: Gemini 1.5 Pro
LITELLM_MODEL=gemini/gemini-1.5-pro python app.py
```

Compare the JSON outputs in `llm_outputs/` to see which model performs best for your use case!

