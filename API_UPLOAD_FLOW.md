# API Image Upload Flow - Telegram Notification Integration

## Overview

The `/api/upload-image` endpoint now follows the exact same analysis flow as the web UI when `auto_analyze=true` is set. All responses, including trade signals, invalidations, and errors, are sent to Telegram.

## API Endpoint

```
POST /api/upload-image
```

### Parameters

- **image** (required): The trading chart image file
- **auto_analyze** (optional): Set to `true` to automatically run the complete analysis pipeline
- **save_permanently** (optional): Set to `true` to save the image in the `uploads/` directory
- **filename** (optional): Custom filename for the saved image

### Example Usage

```bash
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@/path/to/chart.png" \
  -F "auto_analyze=true" \
  -F "save_permanently=true"
```

## Complete Analysis Flow

When `auto_analyze=true`, the endpoint executes the full trading analysis pipeline:

### 1. **Image Upload & Telegram Notification**
- Image is received and saved (temporary or permanent based on `save_permanently`)
- Image is immediately sent to Telegram with basic metadata

### 2. **LLM Chart Analysis**
- Chart is analyzed using the vision model (e.g., GPT-4o)
- Extracts: symbol, timeframe, technical indicators, patterns, opening signals, invalidation conditions
- **📱 Telegram notification sent immediately** with initial analysis:
  - Detected symbol and timeframe
  - Proposed trading direction (LONG/SHORT)
  - Key technical indicators (RSI, MACD)
  - Top 3 chart patterns with confidence scores
  - Expected next steps and wait time

### 3. **Market Data Fetching**
- Fetches real-time market data from MEXC
- Calculates technical indicators: RSI14, MACD12_26_9, STOCH14_3_3, BB20_2, ATR14

### 4. **Signal Validation Polling**
- Continuously polls market data at timeframe intervals
- Validates the trading signal against:
  - **Core checklist conditions** (indicator thresholds, crossovers, etc.)
  - **Invalidation conditions** (price breaches, indicator violations)
  
### 5. **Notifications During Polling**

#### **If Signal is Invalidated:**
- **Telegram notification** is sent with:
  - Symbol
  - Current price
  - Triggered invalidation conditions
  - Current RSI
  - Timestamp
- Also sent via Pushover and Email (if configured)
- Analysis stops here

#### **If Signal Remains Pending:**
- Continues polling until signal becomes valid or invalidated
- Waits for one timeframe interval between checks

### 6. **Trade Gate Decision (if signal valid)**
- LLM evaluates whether to open the trade
- Considers:
  - LLM snapshot (symbol, timeframe, opening signal, risk management)
  - Current market values (price, RSI, time)
  - Programmatic checks (checklist status, invalidation status)

### 7. **Final Trade Notification (if approved)**

#### **If Trade is Approved:**
- **Telegram notification** is sent with:
  - Symbol
  - Direction (LONG/SHORT)
  - Entry price
  - Confidence score
  - Current RSI
  - Stop loss
  - Take profit levels
  - Risk/Reward ratio
  - Timestamp
- Also sent via Pushover and Email (if configured)
- Optionally places order on MEXC (if `MEXC_ENABLE_ORDERS=true`)

#### **If Trade is Rejected:**
- Analysis completes without opening trade
- Gate decision details are logged but no trade notification is sent

### 8. **Error Handling**

If any error occurs during analysis:
- **Telegram notification** is sent with:
  - Image filename
  - Error message
  - Timestamp
- Analysis stops gracefully

## Telegram Notification Examples

### 1. Image Upload Confirmation (Immediate)
```
📊 New Trading Chart Uploaded

Filename: trading_chart_20251031_123456.png
Size: 245.67 KB
Time: 20251031_123456
```

### 2. Initial Analysis Results (Within 30-60 seconds)
```
🔍 Initial Chart Analysis Complete

📊 Symbol: BTCUSDT
⏰ Timeframe: 15m
📸 Screenshot Time: 2025-10-31 12:30

📈 Proposed Direction: LONG

📉 Key Indicators:
• RSI14: 45.23
• MACD Histogram: 0.0012

🎯 Top Patterns Detected:
1. Bullish Flag (85%)
2. Higher Lows (75%)
3. Volume Surge (70%)

⏳ Next Steps:
• Fetching real-time market data...
• Validating signal with live indicators...
• Running trade gate analysis...

You'll receive another notification when validation completes 
(this may take several minutes depending on the timeframe).
```

### 3. Invalidated Trade (During Polling)
```
❌ TRADE INVALIDATED

📊 Symbol: BTCUSDT
💰 Price: $67,234.56
⚠️ Triggered Conditions: price_below_bollinger_middle, rsi_oversold
⏰ Time: 2025-10-31 12:34:56
```

### 4. Valid Trade (Approved)
```
🚀 VALID TRADE SIGNAL

📊 Symbol: BTCUSDT
📈 Direction: LONG
💰 Price: $67,234.56
📊 RSI: 45.23
🎯 Confidence: 85.0%

✅ Trade approved by AI gate
⏰ Time: 2025-10-31 12:34:56

Check your trading platform!
```

### 5. Analysis Error
```
❌ Analysis Error

Image: trading_chart_20251031_123456.png
Error: Failed to fetch market data from MEXC
Time: 20251031_123456

Please check the logs for more details.
```

## Configuration

### Required Environment Variables

```bash
# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Optional: Pushover Configuration
PUSHOVER_TOKEN=your_pushover_token
PUSHOVER_USER=your_pushover_user_key

# Optional: Email Configuration
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_TO=recipient@email.com

# MEXC API (for market data and optional trading)
MEXC_API_KEY=your_mexc_api_key
MEXC_API_SECRET=your_mexc_api_secret

# Optional: Enable live order placement (default: false)
MEXC_ENABLE_ORDERS=false

# LLM Models
LITELLM_VISION_MODEL=gpt-4o
LITELLM_TEXT_MODEL=gpt-4o
```

## Testing the API

### 1. Test Image Upload Only
```bash
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@test_chart.png"
```

### 2. Test Image Upload with Auto-Analysis
```bash
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@test_chart.png" \
  -F "auto_analyze=true"
```

### 3. Test from Apple Shortcuts
See `APPLE_SHORTCUTS_SETUP.md` for detailed instructions on setting up shortcuts to:
- Take screenshots
- Upload to API
- Automatically trigger analysis
- Receive Telegram notifications

## Response Format

### Successful Upload
```json
{
  "success": true,
  "message": "Image uploaded and analysis started - notifications will be sent to Telegram upon completion",
  "metadata": {
    "filename": "trading_chart_20251031_123456.png",
    "size_bytes": 245670,
    "size_kb": 239.91,
    "timestamp": "20251031_123456",
    "file_type": "png",
    "saved_permanently": false
  },
  "telegram_sent": true,
  "job_id": "analysis_20251031_123456"
}
```

### Error Response
```json
{
  "success": false,
  "error": "No image file provided in request"
}
```

## Key Differences from UI Flow

The API endpoint now has **identical behavior** to the UI flow with these minor differences:

1. **Progress Updates**: UI gets real-time progress via SSE (Server-Sent Events), API runs in background thread
2. **Image Display**: UI displays the uploaded image in browser, API returns metadata only
3. **Job Tracking**: API returns a `job_id` for potential future status tracking
4. **Notifications**: Both send identical Telegram notifications for all events

## Advantages of API Endpoint

1. **Apple Shortcuts Integration**: Seamlessly upload charts from iPhone/iPad
2. **Automation**: Can be triggered programmatically or via shortcuts
3. **Headless Operation**: No browser required
4. **Mobile-First**: Perfect for mobile trading workflows
5. **Consistent Notifications**: Same Telegram notifications as UI ensures you never miss a signal

## Future Enhancements

Potential improvements:
1. Add job status endpoint: `GET /api/job-status/{job_id}`
2. Add webhook support for real-time updates
3. Add batch upload support for multiple charts
4. Add historical analysis lookup by job_id
5. Add Telegram bot commands to query analysis results

## Support

For issues or questions:
1. Check `web_app.py` for implementation details
2. Check `services/notification_service.py` for notification logic
3. Check logs for error details
4. Ensure all environment variables are properly configured

