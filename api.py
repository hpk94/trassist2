"""FastAPI application for trading chart analysis with image upload endpoint."""

import os
import shutil
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.trading_orchestrator import TradingOrchestrator

# Initialize FastAPI app
app = FastAPI(
    title="Trading Chart Analysis API",
    description="API for uploading and analyzing trading chart images",
    version="1.0.0"
)

# Initialize trading orchestrator
orchestrator = TradingOrchestrator()

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
    file: UploadFile = File(...),
    max_cycles: Optional[int] = None
):
    """
    Upload a trading chart image and analyze it.
    
    Args:
        file: Image file to upload (PNG, JPG, JPEG)
        max_cycles: Maximum polling cycles for analysis (optional)
    
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
            results = orchestrator.run_complete_analysis(file_path, max_cycles)
            
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
    file: UploadFile = File(...),
    max_cycles: Optional[int] = None
):
    """
    Analyze an uploaded image without saving it permanently.
    
    Args:
        file: Image file to analyze
        max_cycles: Maximum polling cycles for analysis (optional)
    
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
            results = orchestrator.run_complete_analysis(temp_file_path, max_cycles)
            
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
