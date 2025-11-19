# Multi-Model Analysis Feature

## Overview

The trading assistant now supports **multi-model analysis** - when an image is uploaded, all configured models analyze it in parallel, and you can compare their performance to see which model performs best at deciding on opening trades.

## Models Tested

The system tests models in parallel:

1. **ChatGPT5.1** - Configured via `MULTI_MODEL_CHATGPT` (default: `gpt-4o`)
2. **DeepSeek** - Configured via `MULTI_MODEL_DEEPSEEK` (default: `deepseek/deepseek-chat`)

## Configuration

Add these environment variables to your `.env` file to customize the models:

```bash
# Multi-model comparison configuration
MULTI_MODEL_CHATGPT=gpt-4o              # ChatGPT5.1 model
MULTI_MODEL_DEEPSEEK=deepseek/deepseek-chat  # DeepSeek model
```

**Note:** Make sure you have the appropriate API keys configured:
- `OPENAI_API_KEY` for ChatGPT
- `DEEPSEEK_API_KEY` for DeepSeek

## How It Works

1. **Image Upload**: When you upload an image via the web interface or API, the system automatically runs all three models in parallel.

2. **Parallel Analysis**: Each model analyzes the same image simultaneously using threading, making the process efficient.

3. **Result Comparison**: The system compares:
   - **Trading direction** (long/short) from each model
   - **Confidence scores** from each model
   - **Response times** for each model
   - **Agreement** - whether all models agree on the direction
   - **Consensus direction** - the most common direction

4. **Best Model Selection**: The system automatically selects the model with the highest confidence score for the rest of the trading pipeline (signal validation, trade gate, etc.).

5. **Results Storage**: All comparison data is saved to `llm_outputs/multi_model_comparison_YYYYMMDD_HHMMSS.json`

## Output Files

### Comparison File Structure

Each comparison file contains:

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "image_path": "/path/to/image.png",
  "symbol": "BTCUSDT",
  "timeframe": "1m",
  "models_tested": ["ChatGPT5.1", "DeepSeek"],
  "results": {
    "ChatGPT5.1": {
      "model_name": "gpt-4o",
      "result": { /* full LLM analysis result */ },
      "elapsed_time": 2.5,
      "timestamp": "2025-01-15T10:30:02"
    },
    // ... other models
  },
  "errors": {
    // Any models that failed
  },
  "summary": {
    "directions": {
      "ChatGPT5.1": "long",
      "DeepSeek": "long"
    },
    "confidences": {
      "ChatGPT5.1": 0.85,
      "DeepSeek": 0.72
    },
    "response_times": {
      "ChatGPT5.1": 2.5,
      "DeepSeek": 1.8
    },
    "agreement": false,
    "consensus_direction": "long",
    "highest_confidence_model": "ChatGPT5.1",
    "fastest_model": "DeepSeek"
  }
}
```

## API Endpoints

### List All Comparisons

```bash
GET /list-comparisons
```

Returns a list of all multi-model comparison files with summary information.

### View Specific Comparison

```bash
GET /view-comparison?filepath=/path/to/comparison.json
```

Returns the full comparison data for a specific file.

## Usage Examples

### Via Web Interface

1. Upload an image through the web interface
2. The system automatically runs multi-model analysis
3. Check the progress logs to see each model's results
4. View comparison files in `llm_outputs/` directory

### Via API

```bash
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@/path/to/chart.png" \
  -F "auto_analyze=true" \
  -F "save_permanently=true"
```

The response will include the comparison file path in `output_files.multi_model_comparison`.

## Analyzing Performance

To determine which model performs best:

1. **Check Agreement**: Models that consistently agree with each other may be more reliable
2. **Review Confidence Scores**: Higher confidence doesn't always mean better, but it's a factor
3. **Track Response Times**: Faster models can be more cost-effective
4. **Compare Trading Outcomes**: Over time, track which model's trades perform best

## Notes

- **Vision Support**: Not all models support vision/image analysis. If a model fails, it will be logged in the `errors` section of the comparison file.
- **Error Handling**: If a model fails, the system continues with the other models and still produces a comparison.
- **Model Selection**: The system uses the highest confidence model for the trading pipeline, but you can review all results in the comparison file.

## Troubleshooting

### Model Fails with Vision Error

Some models (like `deepseek/deepseek-chat`) don't support vision. Try using a vision-capable variant:
- For DeepSeek: Consider if a vision model exists

### API Key Issues

Make sure all required API keys are set in your `.env` file:
```bash
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
```

### Timeout Issues

If models are timing out, you can increase the timeout in `analyze_trading_chart_with_model()` function (currently 120 seconds).

