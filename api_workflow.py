"""FastAPI application with continuous workflow for trading chart analysis."""

import os
import shutil
import json
import base64
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Path as PathParam
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Trading Chart Analysis Workflow API",
    description="Continuous workflow API for uploading, analyzing, and managing trading chart images",
    version="2.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create directories
UPLOAD_DIR = "uploads"
ANALYSIS_DIR = "analyses"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Response models
class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    success: bool
    message: str
    analysis_id: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AnalysisListResponse(BaseModel):
    """Response model for listing analyses."""
    success: bool
    analyses: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    version: str

def analyze_trading_chart_with_validation(image_path: str) -> dict:
    """Analyze a trading chart image and validate the signal."""
    try:
        from services.trading_orchestrator import TradingOrchestrator
        
        # Initialize orchestrator
        orchestrator = TradingOrchestrator()
        
        # Run complete analysis including signal validation
        results = orchestrator.run_complete_analysis(image_path, max_cycles=1)
        
        return results
    
    except Exception as e:
        raise Exception(f"Chart analysis failed: {str(e)}")

def save_analysis(analysis_id: str, results: dict, image_path: str) -> str:
    """Save analysis results to file."""
    analysis_data = {
        "analysis_id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "image_path": image_path,
        "results": results
    }
    
    analysis_file = os.path.join(ANALYSIS_DIR, f"{analysis_id}.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    return analysis_file

def load_analysis(analysis_id: str) -> dict:
    """Load analysis results from file."""
    analysis_file = os.path.join(ANALYSIS_DIR, f"{analysis_id}.json")
    if not os.path.exists(analysis_file):
        raise FileNotFoundError(f"Analysis {analysis_id} not found")
    
    with open(analysis_file, 'r') as f:
        return json.load(f)

@app.get("/")
async def root():
    """Serve the main interface."""
    return FileResponse('static/index.html')

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )



@app.post("/upload", response_model=AnalysisResponse)
async def upload_chart(
    file: UploadFile = File(...)
):
    """
    Upload a trading chart image and start analysis workflow.
    
    Args:
        file: Image file to upload (PNG, JPG, JPEG)
    
    Returns:
        Analysis ID and initial results
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (PNG, JPG, JPEG)"
            )
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix if file.filename else '.png'
        filename = f"chart_{analysis_id}_{timestamp}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run analysis
        try:
            results = analyze_trading_chart_with_validation(file_path)
            
            # Save analysis results
            save_analysis(analysis_id, results, file_path)
            
            return AnalysisResponse(
                success=True,
                message="Chart uploaded and analyzed successfully",
                analysis_id=analysis_id,
                results=results
            )
            
        except Exception as analysis_error:
            # Clean up file on analysis error
            if os.path.exists(file_path):
                os.remove(file_path)
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

@app.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str = PathParam(..., description="Analysis ID to retrieve")
):
    """
    Get analysis results by ID.
    
    Args:
        analysis_id: The analysis ID returned from upload
    
    Returns:
        Analysis results
    """
    try:
        analysis_data = load_analysis(analysis_id)
        
        return AnalysisResponse(
            success=True,
            message="Analysis retrieved successfully",
            analysis_id=analysis_id,
            results=analysis_data["results"]
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve analysis: {str(e)}"
        )

@app.post("/analysis/{analysis_id}/reanalyze", response_model=AnalysisResponse)
async def reanalyze_chart(
    analysis_id: str = PathParam(..., description="Analysis ID to reanalyze")
):
    """
    Re-analyze an existing chart with updated analysis.
    
    Args:
        analysis_id: The analysis ID to reanalyze
    
    Returns:
        Updated analysis results
    """
    try:
        # Load existing analysis
        analysis_data = load_analysis(analysis_id)
        image_path = analysis_data["image_path"]
        
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404,
                detail="Original image file not found"
            )
        
        # Run new analysis
        new_results = analyze_trading_chart_with_validation(image_path)
        
        # Update analysis results
        analysis_data["results"] = new_results
        analysis_data["reanalyzed_at"] = datetime.now().isoformat()
        
        analysis_file = os.path.join(ANALYSIS_DIR, f"{analysis_id}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        return AnalysisResponse(
            success=True,
            message="Chart reanalyzed successfully",
            analysis_id=analysis_id,
            results=new_results
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reanalysis failed: {str(e)}"
        )

@app.get("/analyses", response_model=AnalysisListResponse)
async def list_analyses():
    """
    List all analyses with metadata.
    
    Returns:
        List of all analyses with basic info
    """
    try:
        analyses = []
        
        for filename in os.listdir(ANALYSIS_DIR):
            if filename.endswith('.json'):
                analysis_id = filename[:-5]  # Remove .json extension
                try:
                    analysis_data = load_analysis(analysis_id)
                    # Extract signal status and market data timestamp
                    results = analysis_data["results"]
                    signal_status = "unknown"
                    market_data_timestamp = None
                    
                    if isinstance(results, dict):
                        # Handle new format with signal validation
                        if "signal_status" in results:
                            signal_status = results["signal_status"]
                        if "market_values" in results and results["market_values"]:
                            current_time = results["market_values"].get("current_time")
                            if current_time:
                                # Convert to DD:MM HH:MM:SS format
                                try:
                                    from datetime import datetime
                                    if isinstance(current_time, str):
                                        dt = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
                                    else:
                                        dt = current_time
                                    market_data_timestamp = dt.strftime("%d:%m %H:%M:%S")
                                except:
                                    market_data_timestamp = str(current_time)
                        
                        # Extract basic info
                        symbol = results.get("llm_output", {}).get("symbol", "Unknown") if "llm_output" in results else results.get("symbol", "Unknown")
                        timeframe = results.get("llm_output", {}).get("timeframe", "Unknown") if "llm_output" in results else results.get("timeframe", "Unknown")
                        opening_signal = results.get("llm_output", {}).get("opening_signal", {}) if "llm_output" in results else results.get("opening_signal", {})
                        signal_direction = opening_signal.get("direction", "Unknown")
                        confidence = opening_signal.get("confidence", 0)
                    else:
                        # Handle old format
                        symbol = results.get("symbol", "Unknown")
                        timeframe = results.get("timeframe", "Unknown")
                        signal_direction = results.get("opening_signal", {}).get("direction", "Unknown")
                        confidence = results.get("opening_signal", {}).get("confidence", 0)
                    
                    analyses.append({
                        "analysis_id": analysis_id,
                        "timestamp": analysis_data["timestamp"],
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "signal_direction": signal_direction,
                        "confidence": confidence,
                        "signal_status": signal_status,
                        "market_data_timestamp": market_data_timestamp,
                        "reanalyzed_at": analysis_data.get("reanalyzed_at")
                    })
                except Exception as e:
                    print(f"Warning: Could not load analysis {analysis_id}: {e}")
        
        # Sort by timestamp (newest first)
        analyses.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return AnalysisListResponse(
            success=True,
            analyses=analyses
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list analyses: {str(e)}"
        )

@app.delete("/analysis/{analysis_id}")
async def delete_analysis(
    analysis_id: str = PathParam(..., description="Analysis ID to delete")
):
    """
    Delete an analysis and its associated files.
    
    Args:
        analysis_id: The analysis ID to delete
    
    Returns:
        Success message
    """
    try:
        # Load analysis to get file paths
        analysis_data = load_analysis(analysis_id)
        image_path = analysis_data["image_path"]
        
        # Delete analysis file
        analysis_file = os.path.join(ANALYSIS_DIR, f"{analysis_id}.json")
        if os.path.exists(analysis_file):
            os.remove(analysis_file)
        
        # Delete image file
        if os.path.exists(image_path):
            os.remove(image_path)
        
        return {"success": True, "message": f"Analysis {analysis_id} deleted successfully"}
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete analysis: {str(e)}"
        )

@app.get("/uploads")
async def list_uploaded_files():
    """List all uploaded image files."""
    try:
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
    uvicorn.run(app, host="0.0.0.0", port=8002)
