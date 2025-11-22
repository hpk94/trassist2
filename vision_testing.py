"""
Vision Model Testing Utilities

This module provides utilities for testing different vision model approaches:
1. Multi-model vision comparison (vision-only models)
2. Two-stage architecture (vision extraction + text decision)
"""

import os
import json
import base64
import time
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import litellm
from prompt import OPENAI_VISION_PROMPT, TRADE_GATE_PROMPT

load_dotenv()

# Vision-only extraction prompt (no decisions, just data extraction)
VISION_EXTRACTION_PROMPT = """## ROLE
You are a **Chart Data Extractor**. Your ONLY job is to extract observable data from trading chart images.
You do NOT make trading decisions. You only extract what you can see.

## TASK
Extract the following information from the chart image:

1. **Basic Information:**
   - Symbol (e.g., BTCUSDT)
   - Timeframe (e.g., 1m, 5m, 15m)
   - Timestamp from chart (YYYY-MM-DD HH:MM format)

2. **Visual Indicators** (extract values you can see on the chart):
   - RSI14 value (if visible)
   - Stochastic %K and %D (if visible)
   - MACD line, signal line, histogram (if visible)
   - Volume (relative to average if visible)

3. **Fibonacci Levels:**
   - Anchor points (swing high, swing low)
   - Retracement/extension levels with numeric values

4. **Chart Patterns:**
   - Patterns visible (triangles, wedges, flags, etc.)
   - Pattern location (candle index)

5. **Support/Resistance:**
   - Key support levels
   - Key resistance levels

6. **Visual Trend:**
   - Overall trend direction (bullish/bearish/neutral) based on visual inspection

## OUTPUT REQUIREMENTS
Return valid JSON only with this schema:

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1m",
  "time_of_screenshot": "YYYY-MM-DD HH:MM",
  "visual_indicators": {
    "RSI14": {"value": 45.2, "visible": true},
    "STOCH14_3_3": {"k_percent": 35.2, "d_percent": 28.1, "visible": true},
    "MACD12_26_9": {"macd_line": 12.5, "signal_line": 8.2, "histogram": 4.3, "visible": true},
    "VOLUME": {"relative": "above_average", "visible": true}
  },
  "fibonacci": {
    "anchors": {"from": 45200.0, "to": 46800.0},
    "levels": {"0.236": 45450.0, "0.382": 45600.0, "0.5": 46000.0, "0.618": 46200.0, "0.786": 46550.0}
  },
  "patterns": [
    {"pattern": "triangle_breakout", "candle_index": 0, "confidence": 0.82}
  ],
  "support_resistance": {
    "support": [45000.0, 45200.0],
    "resistance": [46800.0, 47000.0]
  },
  "visual_trend": "bullish"
}
```

## IMPORTANT
- Only extract what you can SEE on the chart
- If an indicator is not visible, set "visible": false
- Do NOT make trading decisions
- Do NOT interpret market data beyond what's visible
- Be precise with numeric values
"""


def extract_chart_data_vision_only(
    image_path: str, 
    model_name: str = None
) -> Dict[str, Any]:
    """
    Extract structured data from chart image using vision model only.
    No market data, no trading decisions - pure extraction.
    
    Args:
        image_path: Path to chart image
        model_name: Vision model to use (defaults to LITELLM_VISION_MODEL)
    
    Returns:
        Dictionary with extracted chart data
    """
    if model_name is None:
        model_name = os.getenv("LITELLM_VISION_MODEL", "gpt-4o")
    
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    response = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": VISION_EXTRACTION_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": "Extract data from this trading chart."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
        ],
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM API returned None content")
    
    return json.loads(content)


def make_trading_decision_from_extraction(
    extracted_data: Dict[str, Any],
    market_data: Dict[str, Any],
    model_name: str = None
) -> Dict[str, Any]:
    """
    Make trading decision using extracted vision data + real market data.
    Uses a text model (can be cheaper than vision model).
    
    Args:
        extracted_data: Data extracted from vision model
        market_data: Real-time market data (OHLCV, indicators)
        model_name: Text model to use (defaults to LITELLM_TEXT_MODEL)
    
    Returns:
        Trading signal with checklist and invalidations
    """
    if model_name is None:
        model_name = os.getenv("LITELLM_TEXT_MODEL", "gpt-4o")
    
    # Prepare context combining extracted data and market data
    context = f"""
## EXTRACTED CHART DATA
{json.dumps(extracted_data, indent=2)}

## REAL-TIME MARKET DATA
{json.dumps(market_data, indent=2)}

## TASK
Using the extracted chart data and real-time market data, create a trading signal with:
- Direction (long/short)
- Checklist of conditions
- Invalidation rules
- Risk management levels

Follow the same output schema as OPENAI_VISION_PROMPT for opening_signal, but use the real market data values for accuracy.
"""
    
    # Use the original vision prompt structure but with text-only input
    decision_prompt = OPENAI_VISION_PROMPT + "\n\n" + context
    
    response = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": decision_prompt},
            {"role": "user", "content": "Create a trading signal based on the provided chart extraction and market data."}
        ],
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM API returned None content")
    
    return json.loads(content)


def compare_vision_models(
    image_path: str,
    model_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compare multiple vision models on the same chart image.
    Tests vision-only extraction (no decisions).
    
    Args:
        image_path: Path to chart image
        model_names: List of vision model names to test
    
    Returns:
        Comparison results with extraction accuracy metrics
    """
    if model_names is None:
        model_names = [
            os.getenv("LITELLM_VISION_MODEL", "gpt-4o"),
            "claude-3-5-sonnet-20241022",
            "gemini/gemini-1.5-pro"
        ]
    
    results = {}
    errors = {}
    
    for model_name in model_names:
        try:
            start_time = time.time()
            extracted = extract_chart_data_vision_only(image_path, model_name)
            elapsed_time = time.time() - start_time
            
            results[model_name] = {
                "extracted_data": extracted,
                "elapsed_time": elapsed_time,
                "timestamp": time.time()
            }
        except Exception as e:
            errors[model_name] = {
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    # Compare results
    comparison = {
        "image_path": image_path,
        "models_tested": model_names,
        "results": results,
        "errors": errors,
        "comparison": {}
    }
    
    # Extract comparison metrics
    if results:
        symbols = {}
        timeframes = {}
        trends = {}
        
        for model, data in results.items():
            extracted = data["extracted_data"]
            symbols[model] = extracted.get("symbol")
            timeframes[model] = extracted.get("timeframe")
            trends[model] = extracted.get("visual_trend")
        
        comparison["comparison"] = {
            "symbols": symbols,
            "timeframes": timeframes,
            "trends": trends,
            "agreement": {
                "symbol": len(set(symbols.values())) == 1,
                "timeframe": len(set(timeframes.values())) == 1,
                "trend": len(set(trends.values())) == 1
            }
        }
    
    return comparison


def two_stage_analysis(
    image_path: str,
    market_data: Dict[str, Any],
    vision_model: str = None,
    decision_model: str = None
) -> Dict[str, Any]:
    """
    Two-stage analysis: vision extraction + text decision.
    
    Args:
        image_path: Path to chart image
        market_data: Real-time market data
        vision_model: Model for vision extraction
        decision_model: Model for trading decision
    
    Returns:
        Complete analysis with both stages
    """
    if vision_model is None:
        vision_model = os.getenv("LITELLM_VISION_MODEL", "gpt-4o")
    if decision_model is None:
        decision_model = os.getenv("LITELLM_TEXT_MODEL", "gpt-4o")
    
    # Stage 1: Vision extraction
    print(f"Stage 1: Extracting chart data with {vision_model}...")
    start_time = time.time()
    extracted_data = extract_chart_data_vision_only(image_path, vision_model)
    extraction_time = time.time() - start_time
    
    # Stage 2: Decision making
    print(f"Stage 2: Making trading decision with {decision_model}...")
    start_time = time.time()
    decision = make_trading_decision_from_extraction(
        extracted_data, 
        market_data, 
        decision_model
    )
    decision_time = time.time() - start_time
    
    return {
        "extraction": {
            "model": vision_model,
            "data": extracted_data,
            "time": extraction_time
        },
        "decision": {
            "model": decision_model,
            "data": decision,
            "time": decision_time
        },
        "total_time": extraction_time + decision_time
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vision_testing.py <image_path> [mode]")
        print("Modes: compare, two-stage")
        sys.exit(1)
    
    image_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "compare"
    
    if mode == "compare":
        print("Comparing vision models...")
        results = compare_vision_models(image_path)
        print(json.dumps(results, indent=2))
    elif mode == "two-stage":
        print("Two-stage analysis requires market data. See function documentation.")
    else:
        print(f"Unknown mode: {mode}")


