# ============================================================================
# File: app/schemas.py
# ============================================================================
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
class SensorReading(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius", ge=-40, le=85)
    rel_humidity: float = Field(..., description="Relative humidity (%)", ge=0, le=100)
    pressure: float = Field(..., description="Pressure in hPa", ge=300, le=1100)
    gas_resistance: float = Field(..., description="Gas resistance in Ohms", gt=0)
    timestamp: Optional[str] = Field(None, description="ISO timestamp")
class IAQResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    iaq: Optional[float] = Field(None, description="IAQ index (0-500)")
    category: Optional[str] = Field(None, description="Air quality category")
    status: str = Field(..., description="Prediction status")
    model_type: Optional[str] = None
    raw_inputs: Optional[Dict[str, float]] = None
    buffer_size: Optional[int] = None
    required: Optional[int] = None
    message: Optional[str] = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: str
    window_size: int
    loaded: bool
    config: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    models_available: Dict[str, bool]
    active_model: str


class ModelSelection(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: str = Field(..., description="Model type: 'mlp' or 'kan'")
