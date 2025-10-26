from pydantic import BaseModel
from typing import List, Optional

class DetectionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    has_mask: bool
    message: str

class BatchDetectionResponse(BaseModel):
    success: bool
    results: List[dict]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool