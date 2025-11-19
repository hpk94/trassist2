# Telegram Notifications with Multi-Model Analysis

## Overview

When multiple models analyze a trading chart, the Telegram notification now includes a comprehensive comparison section showing all models' results, not just the selected one.

## How It Works

### 1. Multi-Model Analysis

When an image is uploaded:
- All configured models (ChatGPT5.1, DeepSeek) analyze the chart in parallel
- Each model's result is stored with its direction, confidence, and response time
- The system selects the model with the highest confidence for the trading pipeline

### 2. Telegram Notification Content

The Telegram notification includes:

#### Standard Analysis Section
- Symbol, timeframe, screenshot time
- Proposed direction (from selected model)
- Technical indicators
- Risk management details
- Entry conditions checklist
- Invalidation rules
- Chart patterns

#### Multi-Model Comparison Section (NEW)
- **Selected Model**: Which model was chosen (highest confidence) ⭐
- **All Models Results**: 
  - Each model's direction (🟢 LONG / 🔴 SHORT)
  - Confidence score
  - Response time
  - Visual indicator showing which model was selected
- **Failed Models**: Any models that encountered errors
- **Consensus**: Whether all models agree on direction

## Example Telegram Message

```
🔍 COMPREHENSIVE CHART ANALYSIS

━━━━━━━━━━━━━━━━━━━━
📊 TRADE SETUP OVERVIEW
━━━━━━━━━━━━━━━━━━━━

Symbol: BTCUSDT
Timeframe: 1m
Screenshot: 2025-01-15 10:30
Direction: 🟢 LONG
Signal Status: ⏳ PENDING

━━━━━━━━━━━━━━━━━━━━
🔬 MULTI-MODEL ANALYSIS
━━━━━━━━━━━━━━━━━━━━

Selected Model: ChatGPT5.1 (highest confidence)

All Models Results:
• 🟢 ChatGPT5.1 ⭐
  └ Direction: LONG, Confidence: 85%, Time: 2.5s
• 🟢 DeepSeek
  └ Direction: LONG, Confidence: 72%, Time: 1.8s

Consensus: ✅ LONG (models agree)

━━━━━━━━━━━━━━━━━━━━
📈 TECHNICAL INDICATORS
━━━━━━━━━━━━━━━━━━━━
...
```

## Understanding the Results

### Agreement Status

- **✅ Consensus**: All models agree on the same direction
  - Example: Both models say LONG
  - Higher confidence in the trade signal

- **⚠️ Disagreement**: Models have different opinions
  - Example: 1 model says LONG, 1 says SHORT
  - The selected model (highest confidence) is used for trading
  - You can review all models' reasoning in the comparison file

### Model Selection

The system automatically selects the model with the **highest confidence score** for:
- Signal validation
- Trade gate decision
- Order execution (if approved)

However, you can see all models' results in:
1. The Telegram notification (summary)
2. The comparison JSON file (`llm_outputs/multi_model_comparison_*.json`)

### Failed Models

If a model fails (e.g., doesn't support vision, API error, timeout):
- It's shown in the "Failed Models" section
- Error message is truncated to 50 characters
- Other models continue processing
- The system uses successful models' results

## Benefits

### 1. **Transparency**
- See what all models think, not just one
- Understand if there's consensus or disagreement
- Make informed decisions based on multiple opinions

### 2. **Performance Tracking**
- Compare response times across models
- Track which models are fastest
- Identify models that frequently fail

### 3. **Confidence Assessment**
- Higher confidence when models agree
- Lower confidence when models disagree
- Use consensus as an additional signal filter

### 4. **Model Evaluation**
- Over time, see which model performs best
- Track which model's trades are most profitable
- Adjust model selection strategy based on results

## Notification Timing

The multi-model comparison is included in the **initial analysis notification** that is sent:
- Immediately after all models complete analysis
- Before signal validation and polling
- Typically 30-120 seconds after image upload

## Viewing Full Comparison

For detailed comparison data:
1. Check the comparison JSON file in `llm_outputs/`
2. Use the `/list-comparisons` API endpoint
3. Use the `/view-comparison?filepath=...` API endpoint

The comparison file contains:
- Full analysis results from each model
- Complete confidence scores
- Detailed error information
- Response times
- Summary statistics

## Configuration

The models tested are configured via environment variables:

```bash
MULTI_MODEL_CHATGPT=gpt-4o
MULTI_MODEL_DEEPSEEK=deepseek/deepseek-chat
```

Make sure you have the appropriate API keys configured for all models.

