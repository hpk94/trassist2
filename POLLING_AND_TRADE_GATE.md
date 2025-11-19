# Polling and Trade Gate Decision Flow

## Overview

The system uses a two-stage decision process:
1. **Programmatic Polling**: Continuously checks if entry conditions are met using live market data
2. **LLM Trade Gate**: After conditions are met, an LLM makes the final decision on whether to open the trade

## Stage 1: Polling Process

### How Polling Works

1. **Initial Setup**
   - After the multi-model analysis completes, the system extracts the entry conditions from the selected model's output
   - These conditions include things like:
     - RSI14 >= 30
     - Stochastic K > D
     - Volume > average
     - Price above/below certain levels
     - MACD histogram increasing
     - etc.

2. **Polling Loop**
   ```
   Every [timeframe] seconds (e.g., 60s for 1m timeframe):
   ├── Fetch fresh market data (OHLCV candles)
   ├── Calculate technical indicators (RSI, MACD, Stochastic, BB, ATR)
   ├── Check entry conditions (checklist)
   ├── Check invalidation conditions
   └── Determine status: valid / invalidated / pending
   ```

3. **Polling Interval**
   - The polling interval matches the chart timeframe:
     - 1m chart → checks every 60 seconds
     - 5m chart → checks every 5 minutes
     - 15m chart → checks every 15 minutes
     - etc.

4. **Status Determination**
   - **VALID**: All core conditions met OR (≥2 core conditions + strong pattern)
   - **INVALIDATED**: Any invalidation condition triggered
   - **PENDING**: Conditions not yet met, continue polling

5. **Polling Continues Until**
   - ✅ Signal becomes VALID → proceed to trade gate
   - ❌ Signal becomes INVALIDATED → stop, send notification
   - ⏱️ Max cycles reached → stop (optional)
   - ⏹️ User stops analysis → stop

### What Gets Checked During Polling

**Entry Conditions (Checklist)**
- Technical indicator thresholds (RSI, Stochastic, etc.)
- Price level comparisons
- Indicator crossovers
- MACD/Stochastic specific conditions

**Invalidation Conditions**
- Price breaches (e.g., close below swing low)
- Indicator extremes (e.g., RSI > 70 for a long)
- Pattern breaks
- Other conditions that would invalidate the setup

**Important**: During polling, **NO LLM is called**. This is pure programmatic checking of conditions against live market data.

## Stage 2: Trade Gate Decision

### When Trade Gate is Called

The trade gate is **only called** when:
- ✅ Polling determines signal is **VALID**
- ✅ All entry conditions are met
- ✅ No invalidation conditions triggered

### Trade Gate Process

1. **Context Preparation**
   The system prepares a context object containing:
   ```json
   {
     "llm_snapshot": {
       "symbol": "BTCUSDT",
       "timeframe": "1m",
       "opening_signal": { /* original analysis */ },
       "risk_management": { /* stop loss, take profits */ }
     },
     "market_values": {
       "current_price": 50000.00,
       "current_rsi": 45.2,
       "current_time": "2025-01-15 10:30:00"
     },
     "program_checks": {
       "checklist_passed": true,
       "invalidation_triggered": false,
       "triggered_conditions": []
     }
   }
   ```

2. **LLM Call**
   - **Model Used**: `LITELLM_TEXT_MODEL` (configured separately from vision models)
   - **Default**: Same as vision model (e.g., `gpt-4o`)
   - **Can be different**: You can use a cheaper text-only model like `deepseek/deepseek-chat`
   - **Prompt**: `TRADE_GATE_PROMPT` - asks the LLM to be a "strict Trade Gatekeeper"

3. **LLM Decision**
   The LLM reviews:
   - Original chart analysis (from selected model)
   - Current live market data
   - Programmatic validation results
   - Risk management parameters
   
   And decides:
   - ✅ **Approve**: `should_open: true` → trade proceeds
   - ❌ **Reject**: `should_open: false` → trade blocked

4. **LLM Output**
   ```json
   {
     "should_open": true,
     "direction": "long",
     "confidence": 0.85,
     "reasons": ["Strong RSI recovery", "Volume confirms"],
     "warnings": [],
     "execution": {
       "entry_type": "market",
       "entry_price": 50000.00,
       "stop_loss": 49800.00,
       "take_profits": [...],
       "risk_reward": 2.0
     },
     "checks": {
       "invalidation_triggered": false,
       "checklist_score": {"met": 3, "total": 3},
       "context_alignment": "strong"
     }
   }
   ```

## Important: Model Selection

### Multi-Model Analysis (Chart Analysis)
- Uses: ChatGPT5.1, DeepSeek (all vision models)
- Purpose: Analyze the chart image
- Result: Selected model's analysis is used for the rest of the pipeline

### Trade Gate Decision
- Uses: `LITELLM_TEXT_MODEL` (configured separately)
- Purpose: Final approval/rejection decision
- Input: Text-based context (no image)
- **This is NOT the same model that analyzed the chart**

### Why Different Models?

1. **Cost Efficiency**
   - Vision models are expensive
   - Text-only models (like DeepSeek) are much cheaper
   - Trade gate only needs text reasoning, not vision

2. **Speed**
   - Text models are faster
   - Trade gate needs quick decisions

3. **Flexibility**
   - You can use the best vision model for chart analysis
   - And a cost-effective text model for gate decisions

## Configuration

### Trade Gate Model

Set in `.env`:
```bash
# Use same model for both
LITELLM_TEXT_MODEL=gpt-4o

# OR use cheaper model for gate
LITELLM_TEXT_MODEL=deepseek/deepseek-chat
```

### Polling Settings

Polling interval is automatic based on timeframe, but you can limit cycles:
```python
poll_until_decision(symbol, timeframe, llm_output, max_cycles=10)
```

## Complete Flow Diagram

```
Image Upload
    ↓
Multi-Model Analysis (ChatGPT5.1, DeepSeek)
    ↓
Select Best Model (highest confidence)
    ↓
Extract Entry Conditions & Invalidation Rules
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POLLING LOOP (No LLM calls)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
Fetch Market Data → Calculate Indicators
    ↓
Check Entry Conditions ✓
    ↓
Check Invalidation Conditions ✓
    ↓
Status: VALID / INVALIDATED / PENDING
    ↓
If PENDING → Wait [timeframe] → Loop back
    ↓
If INVALIDATED → Stop, Send Notification ❌
    ↓
If VALID → Continue to Trade Gate
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRADE GATE (LLM Text Model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
Prepare Context (original analysis + live data)
    ↓
Call LLM (LITELLM_TEXT_MODEL)
    ↓
LLM Decision: Approve / Reject
    ↓
If Reject → Stop, Send Notification ❌
    ↓
If Approve → Open Trade ✅
```

## Key Points

1. **Polling is programmatic** - No LLM calls during polling, just checking conditions
2. **Trade gate uses a different model** - `LITELLM_TEXT_MODEL`, not the vision model
3. **Two-stage validation** - Programmatic checks first, then LLM final approval
4. **Efficient** - Only calls LLM once when conditions are met
5. **Flexible** - Can use different models for different stages

## Example Timeline

For a 1-minute chart:

```
00:00 - Image uploaded
00:30 - Multi-model analysis complete (3 models analyzed)
00:30 - Polling starts (checks every 60 seconds)
01:30 - Cycle 1: Conditions not met, continue
02:30 - Cycle 2: Conditions not met, continue
03:30 - Cycle 3: ✅ Conditions met! Status = VALID
03:30 - Trade gate called (LITELLM_TEXT_MODEL)
03:32 - Trade gate approves ✅
03:32 - Trade opened
```

## Notifications

- **During Polling**: Periodic updates every 5 cycles
- **If Invalidated**: Immediate notification with triggered conditions
- **If Approved**: Notification with trade details
- **If Rejected**: Notification with rejection reasons

