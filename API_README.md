# Trading Chart Analysis Workflow API

A FastAPI-based REST API with continuous workflow for uploading, analyzing, and managing trading chart images using AI-powered analysis.

## Features

- **Continuous Workflow**: Upload → Analyze → Retrieve → Reanalyze → Manage
- **Persistent Analysis**: Analysis IDs for tracking and managing charts over time
- **AI Analysis**: Automatic chart analysis using OpenAI's vision models
- **Trading Signals**: Generate trading signals with confidence scores
- **Reanalysis**: Re-analyze charts with updated AI models
- **Analysis Management**: List, retrieve, and delete analyses
- **File Persistence**: Keep charts and analysis results for future reference

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create a .env file with your API keys
OPENAI_API_KEY=your_openai_api_key
MEXC_API_KEY=your_mexc_api_key
MEXC_API_SECRET=your_mexc_api_secret
```

## Running the API

### Workflow API (Recommended)
```bash
python api_workflow.py
```

### Simple API (One-off analysis)
```bash
python api_simple.py
```

### Production Server
```bash
uvicorn api_workflow:app --host 0.0.0.0 --port 8002
```

The Workflow API will be available at `http://localhost:8002`

## API Endpoints

### Health Check
- **GET** `/health` - Check API health status
- **GET** `/` - Root endpoint with basic info

### Workflow Management
- **POST** `/upload` - Upload chart and start analysis workflow
- **GET** `/analysis/{id}` - Retrieve analysis results by ID
- **POST** `/analysis/{id}/reanalyze` - Re-analyze existing chart
- **GET** `/analyses` - List all analyses with metadata
- **DELETE** `/analysis/{id}` - Delete analysis and associated files

### File Management
- **GET** `/uploads` - List all uploaded image files

## Usage Examples

### Using curl

1. **Health Check**:
```bash
curl http://localhost:8002/health
```

2. **Upload Chart**:
```bash
curl -X POST "http://localhost:8002/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_chart_image.png"
```

3. **Get Analysis Results**:
```bash
curl -X GET "http://localhost:8002/analysis/{analysis_id}"
```

4. **Reanalyze Chart**:
```bash
curl -X POST "http://localhost:8002/analysis/{analysis_id}/reanalyze"
```

5. **List All Analyses**:
```bash
curl -X GET "http://localhost:8002/analyses"
```

6. **Delete Analysis**:
```bash
curl -X DELETE "http://localhost:8002/analysis/{analysis_id}"
```

### Using Python

```python
import requests

# 1. Upload chart
with open("chart.png", "rb") as f:
    response = requests.post(
        "http://localhost:8002/upload",
        files={"file": f}
    )

result = response.json()
analysis_id = result['analysis_id']

# 2. Get analysis results
response = requests.get(f"http://localhost:8002/analysis/{analysis_id}")
analysis = response.json()

# 3. Reanalyze if needed
response = requests.post(f"http://localhost:8002/analysis/{analysis_id}/reanalyze")
new_analysis = response.json()

# 4. List all analyses
response = requests.get("http://localhost:8002/analyses")
all_analyses = response.json()
```

### Using the Example Script

```bash
python example_workflow_usage.py
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8002/docs`
- **ReDoc**: `http://localhost:8002/redoc`

## Request Parameters

### Upload/Analyze Endpoints

- **file** (required): Image file (PNG, JPG, JPEG)
- **max_cycles** (optional): Maximum polling cycles for market data analysis

## Response Format

```json
{
  "success": true,
  "message": "Analysis completed successfully",
  "analysis_id": "20250103_143022",
  "results": {
    "llm_output": {
      "symbol": "BTCUSDT",
      "timeframe": "1h",
      "opening_signal": {...},
      "risk_management": {...}
    },
    "signal_valid": true,
    "signal_status": "valid",
    "triggered_conditions": [],
    "market_values": {...},
    "gate_result": {...}
  }
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- **200**: Success
- **400**: Bad request (invalid file type, missing parameters)
- **500**: Internal server error (analysis failure, file system error)

## File Management

- Uploaded files are stored in the `uploads/` directory
- Files are automatically cleaned up after analysis
- Use `/uploads` endpoint to list all uploaded files
- Temporary files from `/analyze` endpoint are automatically removed

## Integration with Trading Services

The API integrates with the following services:
- **LLM Service**: OpenAI vision model for chart analysis
- **Market Data Service**: MEXC API for real-time market data
- **Technical Analysis Service**: RSI and other indicator calculations
- **Signal Validation Service**: Trading signal validation logic

## Security Notes

- Ensure your API keys are properly secured in environment variables
- The API accepts any image file type - validate file contents in production
- Consider adding authentication/authorization for production use
- File uploads are limited by server memory and disk space

## Troubleshooting

1. **Import Errors**: Make sure all dependencies are installed
2. **API Key Errors**: Verify your OpenAI and MEXC API keys are set correctly
3. **File Upload Errors**: Check file permissions and disk space
4. **Analysis Errors**: Ensure the image contains a valid trading chart

## Development

To extend the API:
1. Add new endpoints in `api.py`
2. Update response models as needed
3. Add new services in the `services/` directory
4. Update tests and documentation
