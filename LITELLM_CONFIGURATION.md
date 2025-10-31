# LiteLLM Configuration Guide

This project now uses LiteLLM, which provides a unified interface to call different LLM models (OpenAI, Anthropic, Google, etc.) using the same API format.

## Quick Start

Set the `LITELLM_MODEL` environment variable in your `.env` file to choose which model to use:

```bash
# Default (OpenAI GPT-4o)
LITELLM_MODEL=gpt-4o

# Anthropic Claude
LITELLM_MODEL=claude-3-5-sonnet-20241022

# Google Gemini
LITELLM_MODEL=gemini/gemini-1.5-pro

# OpenAI GPT-4 Turbo
LITELLM_MODEL=gpt-4-turbo

# Azure OpenAI
LITELLM_MODEL=azure/gpt-4o
```

## Supported Providers

### OpenAI
```bash
LITELLM_MODEL=gpt-4o
LITELLM_MODEL=gpt-4-turbo
LITELLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=sk-...
```

### Anthropic Claude
```bash
LITELLM_MODEL=claude-3-5-sonnet-20241022
LITELLM_MODEL=claude-3-opus-20240229
LITELLM_MODEL=claude-3-sonnet-20240229
ANTHROPIC_API_KEY=sk-ant-...
```

### Google Gemini
```bash
LITELLM_MODEL=gemini/gemini-1.5-pro
LITELLM_MODEL=gemini/gemini-1.5-flash
GEMINI_API_KEY=...
```

### Azure OpenAI
```bash
LITELLM_MODEL=azure/gpt-4o
AZURE_API_KEY=...
AZURE_API_BASE=https://your-endpoint.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview
```

### Cohere
```bash
LITELLM_MODEL=command-r-plus
COHERE_API_KEY=...
```

### Ollama (Local)
```bash
LITELLM_MODEL=ollama/llama2
LITELLM_MODEL=ollama/mistral
OLLAMA_API_BASE=http://localhost:11434
```

### DeepSeek
```bash
LITELLM_MODEL=deepseek/deepseek-chat
LITELLM_MODEL=deepseek/deepseek-coder
DEEPSEEK_API_KEY=sk-...
```
**Note**: DeepSeek does NOT support vision. Use for text-only tasks like trade gate decisions.

## Dual-Model Configuration (Advanced)

You can use **different models for different tasks**:

```bash
# Vision model for chart analysis (must support vision)
LITELLM_VISION_MODEL=gpt-4o

# Text model for trade gate decisions (any model, including DeepSeek)
LITELLM_TEXT_MODEL=deepseek/deepseek-chat

# API keys for both providers
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
```

**Why use dual models?**
- ✅ Save costs by using cheaper models for text-only tasks
- ✅ Use specialized models for specific tasks
- ✅ Keep high-quality vision for chart analysis
- ✅ Use fast/cheap models for trade gates

**Example combinations:**
```bash
# Cost-optimized
LITELLM_VISION_MODEL=gemini/gemini-1.5-pro    # $3/1M
LITELLM_TEXT_MODEL=deepseek/deepseek-chat     # $0.14/1M

# Quality-optimized
LITELLM_VISION_MODEL=gpt-4o                   # $6/1M
LITELLM_TEXT_MODEL=claude-3-5-sonnet-20241022 # $9/1M

# Balanced (recommended)
LITELLM_VISION_MODEL=gpt-4o                   # $6/1M
LITELLM_TEXT_MODEL=deepseek/deepseek-chat     # $0.14/1M
```

See [`DEEPSEEK_GUIDE.md`](DEEPSEEK_GUIDE.md) for detailed DeepSeek setup.

## Environment Variables

Create a `.env` file in the project root with:

```bash
# Choose your model
LITELLM_MODEL=gpt-4o

# Add the appropriate API key for your provider
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
# or
GEMINI_API_KEY=...

# Existing MEXC configuration
MEXC_API_KEY=your_mexc_api_key
MEXC_API_SECRET=your_mexc_api_secret
MEXC_DEFAULT_VOL=0.001
MEXC_OPEN_TYPE=2
```

## Installation

Install the updated dependencies:

```bash
pip install -r requirements.txt
```

## Advanced Configuration

### Custom Timeout
```python
# Add to your .env
LITELLM_TIMEOUT=300
```

### Debugging
Enable LiteLLM debug mode:
```python
import litellm
litellm.set_verbose = True
```

### Fallback Models
LiteLLM supports automatic fallback to alternative models if the primary fails:
```python
response = litellm.completion(
    model=LITELLM_MODEL,
    messages=messages,
    fallbacks=["gpt-4o", "gpt-4-turbo", "claude-3-sonnet-20240229"]
)
```

## Model Capabilities

Note: Not all models support vision (image analysis). For the `analyze_trading_chart` function that processes images, use:

**Vision-capable models:**
- `gpt-4o` (OpenAI) ✅ **Recommended**
- `gpt-4-turbo` (OpenAI) ✅
- `claude-3-5-sonnet-20241022` (Anthropic) ✅
- `claude-3-opus-20240229` (Anthropic) ✅
- `gemini/gemini-1.5-pro` (Google) ✅

**Text-only models** (won't work for chart analysis):
- `gpt-3.5-turbo`
- `command-r-plus`
- Most Ollama models

## Pricing Comparison

Approximate costs per 1M tokens (as of 2024):

| Model | Input | Output |
|-------|-------|--------|
| GPT-4o | $2.50 | $10.00 |
| GPT-4 Turbo | $10.00 | $30.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3 Opus | $15.00 | $75.00 |
| Gemini 1.5 Pro | $1.25 | $5.00 |

## Troubleshooting

### Error: "litellm.exceptions.AuthenticationError"
- Check that you've set the correct API key environment variable for your provider
- Verify the API key is valid and has proper permissions

### Error: "Model not found"
- Check the model name syntax matches LiteLLM's format
- Some models require specific provider prefixes (e.g., `gemini/`, `azure/`, `ollama/`)

### Vision/Image analysis not working
- Ensure you're using a vision-capable model (see list above)
- Verify the image is properly base64 encoded

## Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Supported Models List](https://docs.litellm.ai/docs/providers)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)

