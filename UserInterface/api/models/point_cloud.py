from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import numpy as np

class PointCloudProcessRequest(BaseModel):
    regions: Optional[Dict[str, Any]] = None
    modes: Optional[Dict[str, bool]] = None

class DenoiseRequest(BaseModel):
    nb_neighbors: int = 100
    std_ratio: float = 0.5
    settings: Dict[str, bool] = {"show": True}

class GroupPointsRequest(BaseModel):
    axis: str
    index: int

    class Config:
        json_schema_extra = {
            "example": {
                "axis": "x",
                "index": 0
            }
        }

class DefectLinesRequest(BaseModel):
    defect_indices: List[int]

class ModelPredictionRequest(BaseModel):
    points: List[float]

class ProcessingResponse(BaseModel):
    status: str
    message: str
    received: Optional[Dict[str, Any]] = None

class DefectLinesResponse(BaseModel):
    status: str
    message: str
    removed_count: Optional[int] = None

class PredictionResponse(BaseModel):
    status: str
    label: int
    probability: float

class UploadResponse(BaseModel):
    message: str
    file_size_mb: float

    class Config:
        json_schema_extra = {
            "example": {
                "message": "文件上传成功，大小: 2.5 MB",
                "file_size_mb": 2.5
            }
        }

def numpy_to_list(arr: np.ndarray) -> List[float]:
    """Convert numpy array to list for JSON serialization"""
    return arr.tolist() if arr is not None else None
