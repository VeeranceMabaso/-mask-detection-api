from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import os
import sys

# Add the inference module to path
sys.path.append('../inference')
from predict import MaskDetector

# Initialize FastAPI app
app = FastAPI(
    title="Face Mask Detection API",
    description="API for detecting face masks in images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global detector
    try:
        detector = MaskDetector("../models/mask_detector.h5")
        print("Mask Detection API started successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Face Mask Detection API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "POST /detect - Upload image to detect mask",
            "health": "GET /health - API health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector is not None
    }

@app.post("/detect")
async def detect_mask(file: UploadFile = File(...)):
    """
    Detect mask presence in uploaded image
    
    - **file**: Image file (JPEG, PNG, etc.)
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Make prediction
        result = detector.predict(image)
        
        return JSONResponse(content={
            "success": True,
            "prediction": result['class'],
            "confidence": round(result['confidence'], 4),
            "has_mask": result['class'] == 'with_mask',
            "message": f"Mask detected: {result['class'] == 'with_mask'}"
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/detect-batch")
async def detect_mask_batch(files: list[UploadFile] = File(...)):
    """
    Detect mask presence in multiple images
    """
    results = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                "file": file.filename,
                "success": False,
                "error": "File must be an image"
            })
            continue
        
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            result = detector.predict(image)
            results.append({
                "file": file.filename,
                "success": True,
                "prediction": result['class'],
                "confidence": round(result['confidence'], 4),
                "has_mask": result['class'] == 'with_mask'
            })
            
        except Exception as e:
            results.append({
                "file": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "results": results
    })

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload for development
    )