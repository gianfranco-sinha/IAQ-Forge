# ============================================================================
# File: app/schemas.py
# ============================================================================
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict, model_validator


# ---------------------------------------------------------------------------
# Module-level field mapping (configured at startup, no import-time coupling)
# ---------------------------------------------------------------------------
_field_mapping: Dict[str, str] = {}


def configure_field_mapping(mapping: Dict[str, str]) -> None:
    """Set the active field mapping (called from app startup)."""
    global _field_mapping
    _field_mapping = dict(mapping)


def get_field_mapping() -> Dict[str, str]:
    """Return the currently active field mapping."""
    return _field_mapping


class SensorReading(BaseModel):
    """Sensor reading input.

    Accepts either a generic ``readings`` dict or the legacy BME680 keyword
    fields (temperature, rel_humidity, pressure, voc_resistance).  Both
    formats are supported for backward compatibility.
    """

    readings: Optional[Dict[str, float]] = Field(
        None, description="Sensor readings keyed by feature name"
    )
    prior_variables: Optional[Dict[str, float]] = Field(
        None, description='External prior variables (e.g. {"presence": 1.0})'
    )
    iaq_actual: Optional[float] = Field(
        None, description="Actual IAQ score from sensor (e.g. BSEC IAQ)"
    )
    timestamp: str = Field(description="ISO 8601 timestamp (e.g. 2026-03-06T14:30:00Z)")
    sensor_id: Optional[str] = Field(
        None, description="Unique sensor hardware ID (serial number, MAC, etc.)"
    )
    firmware_version: Optional[str] = Field(
        None, description="Firmware version of the sending sensor"
    )
    sequence_number: Optional[int] = Field(
        None,
        description="Monotonically increasing sequence number for ordering and replay detection",
    )
    # Legacy BME680 fields — populated into ``readings`` if present
    temperature: Optional[float] = Field(None, exclude=True)
    rel_humidity: Optional[float] = Field(None, exclude=True)
    pressure: Optional[float] = Field(None, exclude=True)
    voc_resistance: Optional[float] = Field(None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _build_readings(cls, values):
        """Apply field_mapping then merge legacy fields into readings dict."""
        if isinstance(values, dict):
            field_mapping = _field_mapping
            if field_mapping:
                readings = values.get("readings")
                if readings and isinstance(readings, dict):
                    values["readings"] = {
                        field_mapping.get(k, k): v for k, v in readings.items()
                    }
                else:
                    for ext, internal in field_mapping.items():
                        if ext in values and values[ext] is not None:
                            values[internal] = values.pop(ext)

            # Merge legacy fields into readings when readings is absent
            if values.get("readings") is None:
                legacy_keys = [
                    "temperature",
                    "rel_humidity",
                    "pressure",
                    "voc_resistance",
                ]
                legacy = {
                    k: values[k]
                    for k in legacy_keys
                    if k in values and values[k] is not None
                }
                if legacy:
                    values["readings"] = legacy
        return values

    def get_readings(self) -> Dict[str, float]:
        """Return the sensor readings dict (always populated after validation)."""
        return self.readings or {}


# ---------------------------------------------------------------------------
# Bayesian inference response structure
# ---------------------------------------------------------------------------


class PriorVariableEffect(BaseModel):
    """Effect of a single prior variable on the predictive distribution."""

    variable: str
    value: float
    state: str = Field(description="Resolved state (e.g. 'true' or 'false')")
    target_shift: float
    prior_std: float
    description: Optional[str] = None


class BayesianUpdate(BaseModel):
    """Result of applying Gaussian conjugate update from prior variables."""

    pre_mean: float
    pre_std: float
    post_mean: float
    post_std: float
    variables_applied: List[PriorVariableEffect]


class Observation(BaseModel):
    """Direct sensor measurements — the evidence conditioning our inference."""

    sensor_type: str
    readings: Dict[str, float]
    engineered_features: Optional[Dict[str, float]] = None
    timestamp: Optional[str] = None


class UncertaintyEstimate(BaseModel):
    """Quantified uncertainty around the predicted value."""

    std: float
    ci_lower: float = Field(description="Lower bound of 95% credible interval")
    ci_upper: float = Field(description="Upper bound of 95% credible interval")
    method: str = Field(
        description="mc_dropout | weight_sampling | history_std | deterministic"
    )


class Predicted(BaseModel):
    """The model's predicted value for the latent IAQ variable given the evidence."""

    mean: float
    category: str
    uncertainty: Optional[UncertaintyEstimate] = None
    iaq_standard: str = "bsec"


class Prior(BaseModel):
    """Belief about IAQ before this observation — from recent history or training distribution."""

    mean: float
    std: float
    source: str = Field(description="history_window | training_distribution")
    n_observations: int


class InferenceMetadata(BaseModel):
    """How the inference was performed."""

    model_config = ConfigDict(protected_namespaces=())
    model_type: str
    window_size: int
    buffer_size: int
    uncertainty_method: Optional[str] = None
    mc_samples: Optional[int] = None


class IAQResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # Backward-compatible top-level fields
    iaq: Optional[float] = Field(None, description="Predicted IAQ index")
    category: Optional[str] = Field(None, description="Air quality category")
    status: str = Field(..., description="Prediction status")
    model_type: Optional[str] = None
    raw_inputs: Optional[Dict[str, float]] = None
    buffer_size: Optional[int] = None
    required: Optional[int] = None
    message: Optional[str] = None

    # Structured inference fields
    observation: Optional[Observation] = None
    predicted: Optional[Predicted] = None
    prior: Optional[Prior] = None
    inference: Optional[InferenceMetadata] = None
    bayesian_update: Optional[BayesianUpdate] = None


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
    model_type: str = Field(..., description="Model type to activate")


# ---------------------------------------------------------------------------
# Sensor registration (field mapping API)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Structured response envelope (LLM Readiness Phase 1)
# ---------------------------------------------------------------------------


class DomainErrorCode(str, Enum):
    """Domain-specific error codes for programmatic error handling."""

    NO_DATA = "NO_DATA"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    INFLUX_UNREACHABLE = "INFLUX_UNREACHABLE"
    TRAINING_DIVERGED = "TRAINING_DIVERGED"
    NEGATIVE_R2 = "NEGATIVE_R2"
    STALE_CONFIG = "STALE_CONFIG"
    CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"
    CONFIGURATION = "CONFIGURATION"


class StructuredResponse(BaseModel):
    """Unified response envelope for REST endpoints and MCP tools.

    Every endpoint wraps its result in this envelope so that callers
    (human or LLM agent) get a consistent schema with actionable
    error information.
    """

    status: str = Field(
        ...,
        description="success | warning | partial | error | fatal",
    )
    result: Optional[Any] = Field(None, description="Payload on success")
    warnings: List[str] = Field(default_factory=list)
    error_code: Optional[str] = Field(
        None, description="DomainErrorCode value on failure"
    )
    detail: Optional[str] = Field(None, description="Human-readable error detail")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Extra structured context"
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Concrete recovery suggestions the caller can act on",
    )


# ---------------------------------------------------------------------------
# Sensor registration (field mapping API)
# ---------------------------------------------------------------------------


class SensorRegisterRequest(BaseModel):
    """Request body for POST /sensors/register."""

    source_type: str = Field(
        "csv_headers",
        description="Source type: 'csv_headers' or 'example_payload'",
    )
    fields: List[str] = Field(..., description="List of source field names to map")
    sample_values: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Sample values per field for range validation",
    )
    backend: str = Field("fuzzy", description="Mapping backend: 'fuzzy' or 'ollama'")
    sensor_id: Optional[str] = Field(None, description="Unique sensor hardware ID")
    firmware_version: Optional[str] = Field(None, description="Sensor firmware version")


class FieldMatchResponse(BaseModel):
    """A single field mapping in the registration response."""

    source_field: str
    target_feature: str
    target_quantity: str
    confidence: float
    method: str


class SensorRegisterResponse(BaseModel):
    """Response from POST /sensors/register."""

    mapping_id: str
    status: str = "proposed"
    mapping: List[FieldMatchResponse]
    unresolved: List[str]


class SensorConfirmRequest(BaseModel):
    """Request body for POST /sensors/register/{mapping_id}/confirm."""

    overrides: Optional[Dict[str, str]] = Field(
        None,
        description="Manual overrides: {source_field: target_feature}",
    )


class SensorConfirmResponse(BaseModel):
    """Response from POST /sensors/register/{mapping_id}/confirm."""

    status: str = "confirmed"
    field_mapping: Dict[str, str]
    sensor_id: Optional[str] = None
    firmware_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Per-sensor drift report
# ---------------------------------------------------------------------------


class FeatureDriftStats(BaseModel):
    """Per-feature drift statistics."""

    mean: float
    std: float
    min: float
    max: float
    cv: float = Field(description="Coefficient of variation")
    estimated_drift: Optional[float] = Field(
        None, description="Estimated total drift = slope * sensor_age_days"
    )
    estimated_drift_pct: Optional[float] = Field(
        None, description="Estimated drift as % of mean"
    )
    drift_status: str = Field(description="OK | DRIFT | unknown")


class SensorDriftReport(BaseModel):
    """Drift report for a specific sensor device.

    Uses the BME680 3-year drift profile as the canonical drift model
    for all BME680 sensors. This assumption will be validated as more
    sensor units are deployed.
    """

    sensor_id: str
    sensor_start: str = Field(description="Sensor install date (from drift summary)")
    first_reading: str = Field(description="ISO timestamp of first observed reading")
    last_reading: str = Field(description="ISO timestamp of most recent reading")
    sensor_age_days: float
    readings_count: int = Field(description="Total readings received (lifetime)")
    readings_in_window: int = Field(description="Readings in current rolling window")
    features: Dict[str, FeatureDriftStats]
    warnings: List[str]
    health: str = Field(description="good | warning | drift")
