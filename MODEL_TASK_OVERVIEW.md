# Model Task Overview

This document provides a comprehensive overview of which AI model is used for which task in the trading assistant application.

## Model Configuration Variables

The application uses the following environment variables to configure models:

| Variable | Default | Purpose | Vision Support Required |
|----------|---------|---------|------------------------|
| `LITELLM_VISION_MODEL` | `gpt-4o` | Chart analysis (Stage 1: Image extraction) | ✅ Yes |
| `LITELLM_TEXT_MODEL` | `gpt-4o` | Trade gate decisions & Chart analysis (Stage 2: Decision) | ❌ No |
| `MULTI_MODEL_CHATGPT` | `gpt-4o` | Multi-model comparison (OpenAI) | ✅ Yes |
| `MULTI_MODEL_DEEPSEEK` | `gpt-4o` | Multi-model comparison (DeepSeek fallback) | ✅ Yes |
| `MULTI_MODEL_GEMINI` | `gemini/gemini-1.5-pro` | Multi-model comparison (Gemini) | ✅ Yes |

---

## Task Flow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN PIPELINE FLOW                           │
└─────────────────────────────────────────────────────────────────┘

1. Chart Analysis (Two-Stage Process)
   ├─ Stage 1: Image Extraction
   │  └─ Model: LITELLM_VISION_MODEL
   │  └─ Prompt: VISION_EXTRACTION_PROMPT
   │  └─ Input: Chart image (base64)
   │  └─ Output: Extracted chart data (JSON)
   │
   └─ Stage 2: Market Data Analysis
      └─ Model: LITELLM_TEXT_MODEL
      └─ Prompt: MARKET_DATA_ANALYSIS_PROMPT
      └─ Input: Extracted data + Real-time market data
      └─ Output: Trading signal analysis (JSON)

2. Trade Gate Decision
   └─ Model: LITELLM_TEXT_MODEL
   └─ Prompt: TRADE_GATE_PROMPT
   └─ Input: LLM analysis + Market values + Checklist results
   └─ Output: Trade approval/rejection decision (JSON)

┌─────────────────────────────────────────────────────────────────┐
│              MULTI-MODEL COMPARISON FLOW                        │
└─────────────────────────────────────────────────────────────────┘

3. Multi-Model Analysis (Parallel Execution)
   ├─ Model 1: MULTI_MODEL_CHATGPT (OpenAI)
   │  └─ Runs: analyze_trading_chart_with_model()
   │  └─ Both stages use the same model
   │
   ├─ Model 2: MULTI_MODEL_DEEPSEEK (DeepSeek fallback)
   │  └─ Runs: analyze_trading_chart_with_model()
   │  └─ Both stages use the same model
   │
   └─ Model 3: MULTI_MODEL_GEMINI (Gemini)
      └─ Runs: analyze_trading_chart_with_model()
      └─ Both stages use the same model
```

---

## Detailed Task Breakdown

### 1. Chart Analysis - Stage 1: Image Extraction

**Function**: `analyze_trading_chart()` (lines 520-710 in `web_app.py`)

**Model Used**: `LITELLM_VISION_MODEL`
- **Default**: `gpt-4o`
- **Must support vision**: ✅ Yes
- **Configuration**: `LITELLM_VISION_MODEL` env var

**Task Description**:
- Extracts observable data from trading chart images
- Identifies symbols, timeframes, timestamps
- Extracts visual indicators (RSI, MACD, Volume)
- Identifies Fibonacci levels and chart patterns
- Determines support/resistance levels
- Validates chart patterns

**Prompt Used**: `VISION_EXTRACTION_PROMPT` (from `prompt.py`)

**Input**:
- Chart image (base64 encoded)
- System prompt with extraction instructions

**Output**:
- JSON with extracted chart data:
  - Symbol, timeframe, timestamp
  - Visual indicators
  - Fibonacci levels
  - Chart patterns
  - Support/resistance levels
  - Visual trend direction

**Code Location**: `web_app.py` lines 543-584

---

### 2. Chart Analysis - Stage 2: Market Data Analysis

**Function**: `analyze_trading_chart()` (lines 603-710 in `web_app.py`)

**Model Used**: `LITELLM_TEXT_MODEL`
- **Default**: `gpt-4o`
- **Must support vision**: ❌ No (text-only is fine)
- **Configuration**: `LITELLM_TEXT_MODEL` env var

**Task Description**:
- Combines extracted chart data with real-time market data
- Validates trading signals against technical indicators
- Assesses setup strength using RSI, Stochastic, Volume, Fibonacci
- Identifies confluence of confirmations
- Defines opening signal with measurable conditions
- Creates invalidation rules
- Recommends improvements

**Prompt Used**: `MARKET_DATA_ANALYSIS_PROMPT` (from `prompt.py`)

**Input**:
- Extracted chart data (from Stage 1)
- Real-time market data (OHLCV, indicators)
- Symbol and timeframe information

**Output**:
- JSON with complete trading analysis:
  - Opening signal (direction, checklist, invalidation rules)
  - Risk management (stop loss, take profits)
  - Validity assessment
  - Summary actions

**Code Location**: `web_app.py` lines 647-702

**Note**: This stage can use a cheaper text-only model (like DeepSeek) since it doesn't need vision capabilities.

---

### 3. Trade Gate Decision

**Function**: `llm_trade_gate_decision()` (lines 1230-1330 in `web_app.py`)

**Model Used**: `LITELLM_TEXT_MODEL`
- **Default**: `gpt-4o`
- **Must support vision**: ❌ No (text-only is fine)
- **Configuration**: `LITELLM_TEXT_MODEL` env var

**Task Description**:
- Reviews programmatically validated signal
- Checks current market context
- Validates checklist pass/fail status
- Checks for invalidation triggers
- Makes final go/no-go decision
- Provides execution parameters (entry, stop loss, take profits)
- Calculates risk/reward ratio

**Prompt Used**: `TRADE_GATE_PROMPT` (from `prompt.py`)

**Input**:
- LLM analysis snapshot (from Stage 2)
- Current market values (price, RSI, time)
- Programmatic check results (checklist, invalidations)

**Output**:
- JSON with trade decision:
  - `should_open`: boolean
  - `direction`: "long" or "short"
  - `confidence`: 0.0-1.0
  - `reasons`: array of strings
  - `execution`: entry type, prices, risk/reward
  - `checks`: validation results

**Code Location**: `web_app.py` lines 1298-1328

**Note**: This is a pure text-based decision task, perfect for cost-effective models like DeepSeek.

---

### 4. Multi-Model Comparison - Individual Model Analysis

**Function**: `analyze_trading_chart_with_model()` (lines 712-878 in `web_app.py`)

**Models Used**: 
- `MULTI_MODEL_CHATGPT` (default: `gpt-4o`)
- `MULTI_MODEL_DEEPSEEK` (default: `gpt-4o` - DeepSeek doesn't support vision)
- `MULTI_MODEL_GEMINI` (default: `gemini/gemini-1.5-pro`)

**Task Description**:
- Each model runs the complete two-stage analysis independently
- Stage 1: Image extraction (uses the specified model)
- Stage 2: Market data analysis (uses the same model if it supports vision, otherwise falls back to `LITELLM_TEXT_MODEL`)

**Execution**:
- All models run in parallel using threading
- Each model processes the same chart image
- Results are collected and compared

**Code Location**: `web_app.py` lines 880-1004

**Logic for Stage 2 Model Selection** (line 819):
```python
# If model supports vision (gpt, gemini, claude, qwen), use it for both stages
# Otherwise, fall back to LITELLM_TEXT_MODEL for decision stage
decision_model = model_name if any(x in model_name.lower() 
    for x in ["gpt", "gemini", "claude", "qwen"]) else LITELLM_TEXT_MODEL
```

---

### 5. Multi-Model Comparison - Aggregation

**Function**: `analyze_trading_chart_multi_model()` (lines 880-1150 in `web_app.py`)

**Task Description**:
- Coordinates parallel execution of multiple models
- Collects results from all models
- Compares:
  - Trading directions (long/short)
  - Confidence scores
  - Response times
  - Agreement between models
- Selects a model for pipeline execution (consensus or first successful)
- Saves comparison data to JSON file

**Output**:
- Comparison summary with:
  - Results from each model
  - Agreement status
  - Consensus direction
  - Fastest model
  - Highest confidence model
  - All individual results

**Code Location**: `web_app.py` lines 880-1150

---

## Model Selection Logic

### For Chart Analysis (Two-Stage)

1. **Stage 1 (Image Extraction)**:
   - Always uses: `LITELLM_VISION_MODEL`
   - Must support vision/image analysis

2. **Stage 2 (Market Data Analysis)**:
   - Uses: `LITELLM_TEXT_MODEL`
   - Can be any model (vision or text-only)
   - Cost optimization opportunity: Use cheaper text-only model here

### For Multi-Model Comparison

1. **Stage 1 (Image Extraction)**:
   - Uses: The specific model being tested (e.g., `MULTI_MODEL_CHATGPT`)
   - Must support vision

2. **Stage 2 (Market Data Analysis)**:
   - Uses: Same model if it supports vision (gpt, gemini, claude, qwen)
   - Falls back to: `LITELLM_TEXT_MODEL` if model doesn't support vision

### For Trade Gate

- Always uses: `LITELLM_TEXT_MODEL`
- Text-only task, no vision required
- Best place to use cost-effective models like DeepSeek

---

## Cost Optimization Recommendations

### Current Setup (Default)
```
LITELLM_VISION_MODEL=gpt-4o          # ~$6/1M tokens
LITELLM_TEXT_MODEL=gpt-4o            # ~$6/1M tokens
```

### Cost-Optimized Setup
```
LITELLM_VISION_MODEL=gemini/gemini-1.5-pro    # ~$3/1M tokens (50% savings)
LITELLM_TEXT_MODEL=deepseek/deepseek-chat     # ~$0.14/1M tokens (98% savings)
```

### Quality-Optimized Setup
```
LITELLM_VISION_MODEL=gpt-4o                   # Best vision quality
LITELLM_TEXT_MODEL=claude-3-5-sonnet-20241022   # Best reasoning
```

---

## Summary Table

| Task | Model Variable | Vision Required? | Typical Model | Cost Impact |
|------|---------------|------------------|---------------|-------------|
| **Chart Analysis - Stage 1** | `LITELLM_VISION_MODEL` | ✅ Yes | `gpt-4o`, `gemini/gemini-1.5-pro` | High (vision models) |
| **Chart Analysis - Stage 2** | `LITELLM_TEXT_MODEL` | ❌ No | `gpt-4o`, `deepseek/deepseek-chat` | Medium (can be optimized) |
| **Trade Gate Decision** | `LITELLM_TEXT_MODEL` | ❌ No | `deepseek/deepseek-chat` | Low (text-only) |
| **Multi-Model - OpenAI** | `MULTI_MODEL_CHATGPT` | ✅ Yes | `gpt-4o` | High |
| **Multi-Model - DeepSeek** | `MULTI_MODEL_DEEPSEEK` | ✅ Yes | `gpt-4o` (fallback) | High |
| **Multi-Model - Gemini** | `MULTI_MODEL_GEMINI` | ✅ Yes | `gemini/gemini-1.5-pro` | Medium |

---

## Key Takeaways

1. **Vision models are required** for chart image analysis (Stage 1)
2. **Text models can be used** for decision-making tasks (Stage 2, Trade Gate)
3. **Cost savings opportunity**: Use cheaper text-only models for Stage 2 and Trade Gate
4. **Multi-model comparison** tests all models in parallel for comparison purposes
5. **Model selection is automatic** based on environment variables

---

## File References

- **Main code**: `web_app.py`
  - Lines 48-57: Model configuration
  - Lines 520-710: Chart analysis (two-stage)
  - Lines 712-878: Single model analysis
  - Lines 880-1150: Multi-model comparison
  - Lines 1230-1330: Trade gate decision

- **Prompts**: `prompt.py`
  - `VISION_EXTRACTION_PROMPT`: Stage 1 extraction
  - `MARKET_DATA_ANALYSIS_PROMPT`: Stage 2 analysis
  - `TRADE_GATE_PROMPT`: Trade gate decision

