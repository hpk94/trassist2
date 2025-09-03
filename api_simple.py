"""Simplified FastAPI application for trading chart analysis with image upload endpoint."""

import os
import shutil
import json
import base64
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Trading Chart Analysis API",
    description="API for uploading and analyzing trading chart images",
    version="1.0.0"
)

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Response models
class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    success: bool
    message: str
    analysis_id: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    version: str

def analyze_trading_chart(image_path: str) -> dict:
    """Analyze a trading chart image using OpenAI Vision."""
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        client = OpenAI()
        
        # Simple prompt for chart analysis
        prompt = """
        Analyze this trading chart image and provide a JSON response with the following structure:
        {
            "symbol": "trading pair symbol (e.g., BTCUSDT)",
            "timeframe": "timeframe (e.g., 1h, 4h, 1d)",
            "time_of_screenshot": "timestamp in YYYY-MM-DD HH:MM format",
            "opening_signal": {
                "direction": "buy or sell",
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation"
            },
            "technical_indicators": {
                "RSI": "current RSI value if visible",
                "support": "support level if visible",
                "resistance": "resistance level if visible"
            },
            "risk_management": {
                "stop_loss": "suggested stop loss level",
                "take_profit": "suggested take profit level"
            }
        }
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        raise Exception(f"Chart analysis failed: {str(e)}")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/upload", response_model=AnalysisResponse)
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a trading chart image and analyze it.
    
    Args:
        file: Image file to upload (PNG, JPG, JPEG)
    
    Returns:
        Analysis results or error message
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (PNG, JPG, JPEG)"
            )
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix if file.filename else '.png'
        filename = f"chart_{timestamp}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run analysis
        try:
            results = analyze_trading_chart(file_path)
            
            # Clean up uploaded file after analysis
            background_tasks.add_task(cleanup_file, file_path)
            
            return AnalysisResponse(
                success=True,
                message="Analysis completed successfully",
                analysis_id=timestamp,
                results=results
            )
            
        except Exception as analysis_error:
            # Clean up file on analysis error
            background_tasks.add_task(cleanup_file, file_path)
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(analysis_error)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_existing_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Analyze an uploaded image without saving it permanently.
    
    Args:
        file: Image file to analyze
    
    Returns:
        Analysis results
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (PNG, JPG, JPEG)"
            )
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix if file.filename else '.png'
        temp_filename = f"temp_chart_{timestamp}{file_extension}"
        temp_file_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run analysis
        try:
            results = analyze_trading_chart(temp_file_path)
            
            # Clean up temporary file
            background_tasks.add_task(cleanup_file, temp_file_path)
            
            return AnalysisResponse(
                success=True,
                message="Analysis completed successfully",
                analysis_id=timestamp,
                results=results
            )
            
        except Exception as analysis_error:
            # Clean up file on analysis error
            background_tasks.add_task(cleanup_file, temp_file_path)
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(analysis_error)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

def cleanup_file(file_path: str):
    """Clean up uploaded file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not clean up file {file_path}: {e}")

@app.get("/uploads")
async def list_uploads():
    """List all uploaded files."""
    try:
        if not os.path.exists(UPLOAD_DIR):
            return {"uploads": []}
        
        files = []
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return {"uploads": files}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list uploads: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
