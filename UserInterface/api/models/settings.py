from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class CylinderSettings(BaseModel):
    cylinder_method: str = "NormalRANSAC"
    normal_neighbors: int = 30
    min_radius: float = 6.0
    max_radius: float = 11.0
    ransac_threshold: float = 0.01
    max_iterations: int = 1000
    normal_distance_weight: float = 0.8

class ArcSettings(BaseModel):
    arcNormalNeighbors: int = 10
    fitIterations: int = 50
    samplePercentage: int = 50

class LineSettings(BaseModel):
    point_size: int = 3
    defect_lines: List[int] = []

class SaveSettingsRequest(BaseModel):
    name: str
    cylinderSettings: CylinderSettings
    arcSettings: ArcSettings

class SettingsResponse(BaseModel):
    id: int
    name: str
    cylinderSettings: Dict[str, Any]
    arcSettings: Dict[str, Any]
    createdAt: datetime

class PointCloudGenerationRequest(BaseModel):
    noise_std: float = 0.01
    arc_angle: float = 360.0
    axis_direction: List[float] = [0, 0, 1]
    axis_density: int = 500
    arc_density: int = 100

class LineStatsResponse(BaseModel):
    status: str
    lineStats: List[Dict[str, Any]]
    overallStats: Dict[str, Any]

class CloudStoreResponse(BaseModel):
    id: int
    filename: str
    views: Dict[str, str]
    createdAt: datetime
