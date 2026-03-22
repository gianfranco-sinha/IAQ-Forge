# ============================================================================
# File: app/main.py
# ============================================================================
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from app.models import IAQPredictor, _KAN_AVAILABLE
from app.inference import InferenceEngine
from app.schemas import (
    SensorReading,
    IAQResponse,
    ModelInfo,
    HealthResponse,
    ModelSelection,
    SensorRegisterRequest,
    SensorRegisterResponse,
    FieldMatchResponse,
    SensorConfirmRequest,
    SensorConfirmResponse,
    SensorDriftReport,
    StructuredResponse,
    configure_field_mapping,
)
from app.config import settings
from app.database import influx_manager
from app.exceptions import IAQError
import app.builtin_profiles  # noqa: F401  — registers sensor/standard profiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictors and inference engines
predictors = {}
inference_engines = {}
active_model = settings.DEFAULT_MODEL

# Transient storage for proposed field mappings (not persisted)
_pending_mappings: Dict[str, "MappingResult"] = {}


MODEL_PATHS = {
    "mlp": settings.MLP_MODEL_PATH,
    "kan": settings.KAN_MODEL_PATH,
    "lstm": settings.LSTM_MODEL_PATH,
    "cnn": settings.CNN_MODEL_PATH,
    "bnn": settings.BNN_MODEL_PATH,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global predictors, inference_engines, active_model

    # Configure field mapping from model_config.yaml for SensorReading validation
    cfg = settings.load_model_config()
    field_mapping = cfg.get("sensor", {}).get("field_mapping", {})
    if field_mapping:
        configure_field_mapping(field_mapping)

    if settings.ENVIRONMENT == "production":
        logger.warning(
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║              *** PRODUCTION ENVIRONMENT ***              ║\n"
            "╠══════════════════════════════════════════════════════════╣\n"
            "║  Real sensor data and live InfluxDB writes are active.   ║\n"
            "║  API key auth : %-8s                                ║\n"
            "║  InfluxDB     : %-8s                                ║\n"
            "║  Root path    : %-20s                   ║\n"
            "║                                                          ║\n"
            "║  Do NOT run training jobs against this instance          ║\n"
            "║  without taking a backup first.                          ║\n"
            "╚══════════════════════════════════════════════════════════╝",
            "enabled" if settings.API_KEY else "DISABLED",
            "enabled" if settings.INFLUX_ENABLED else "disabled",
            os.getenv("ROOT_PATH", ""),
        )

    logger.info("Loading IAQ prediction models...")

    for model_type, model_path in MODEL_PATHS.items():
        try:
            model_cfg = settings.get_model_config(model_type)
            window_size = model_cfg.get("window_size", 10)

            # KAN: use remote sidecar when local KAN is unavailable
            if model_type == "kan" and not _KAN_AVAILABLE and settings.KAN_REMOTE_URL:
                from app.remote_predictor import RemoteKANPredictor

                predictor = RemoteKANPredictor(
                    base_url=settings.KAN_REMOTE_URL,
                    timeout=settings.KAN_REMOTE_TIMEOUT,
                    window_size=window_size,
                )
                if not predictor.load_model(model_path):
                    logger.warning(
                        "KAN sidecar at %s not available — skipping KAN",
                        settings.KAN_REMOTE_URL,
                    )
                    continue
                predictors[model_type] = predictor
                inference_engines[model_type] = InferenceEngine(predictor)
                logger.info("KAN model loaded via remote sidecar at %s", settings.KAN_REMOTE_URL)
                continue

            if model_type == "kan" and not _KAN_AVAILABLE:
                logger.warning(
                    "KAN not available (Python >3.9?) and KAN_REMOTE_URL not set — skipping KAN"
                )
                continue

            predictor = IAQPredictor(
                model_type=model_type, window_size=window_size
            )
            if not predictor.load_model(model_path):
                logger.warning(
                    "No trained %s model found at %s", model_type.upper(), model_path
                )
                continue
            predictors[model_type] = predictor
            inference_engines[model_type] = InferenceEngine(predictor)
            logger.info("%s model loaded successfully", model_type.upper())
        except Exception as e:
            logger.warning("Failed to load %s model: %s", model_type.upper(), e)

    if not predictors:
        logger.error(
            "No trained models found. The service will start in degraded mode — "
            "all prediction endpoints will return 503.\n"
            "  Train models with:  python -m iaq4j train --model mlp --epochs 200\n"
            "  Or create dummies:  python training/create_dummy_models.py"
        )
    else:
        if active_model not in predictors:
            fallback = next(iter(predictors))
            logger.warning(
                "Default model '%s' not available. Falling back to '%s'.",
                active_model,
                fallback,
            )
            active_model = fallback
        logger.info(
            "Active model: %s  |  Available: %s", active_model, list(predictors.keys())
        )

    # Check InfluxDB connection
    if settings.INFLUX_ENABLED:
        db_status = influx_manager.health_check()
        if db_status["status"] == "healthy":
            logger.info("InfluxDB connection established")
        else:
            logger.warning(
                f"InfluxDB unavailable: {db_status.get('error', 'Unknown error')}"
            )

    yield

    # Cleanup
    influx_manager.close()
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="ML-based indoor air quality prediction — sensor and standard agnostic",
    lifespan=lifespan,
    root_path=os.getenv("ROOT_PATH", ""),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://enviro-sensors.uk",
        "http://enviro-sensors.uk",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


# ---------------------------------------------------------------------------
# Global exception handlers → StructuredResponse
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    status = "error" if exc.status_code < 500 else "fatal"
    return JSONResponse(
        status_code=exc.status_code,
        content=StructuredResponse(
            status=status,
            detail=exc.detail,
        ).model_dump(exclude_none=True),
    )


@app.exception_handler(IAQError)
async def iaq_error_handler(request: Request, exc: IAQError):
    return JSONResponse(
        status_code=422,
        content=StructuredResponse(
            status="error",
            error_code=exc.code.value,
            detail=str(exc),
            next_steps=[exc.suggestion] if exc.suggestion else [],
        ).model_dump(exclude_none=True),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=StructuredResponse(
            status="fatal",
            detail=f"{type(exc).__name__}: {exc}",
            next_steps=["Check server logs for full traceback"],
        ).model_dump(exclude_none=True),
    )


async def require_api_key(x_api_key: str = Header(None)):
    """Reject requests without a valid API key (when API_KEY is set)."""
    if not settings.API_KEY:
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


auth = [Depends(require_api_key)]


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictors else "degraded",
        models_available={m: m in predictors for m in MODEL_PATHS},
        active_model=active_model,
    )


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check including database status."""
    db_health = influx_manager.health_check()

    return {
        "service": {
            "status": "healthy" if predictors else "degraded",
            "models_loaded": list(predictors.keys()),
            "active_model": active_model,
        },
        "database": db_health,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/models", response_model=dict)
async def list_models():
    """List available models and their info."""
    models_info = {}

    for name, predictor in predictors.items():
        models_info[name] = {
            "loaded": True,
            "window_size": predictor.window_size,
            "config": predictor.config,
        }

    return {"active": active_model, "available": models_info}


@app.post("/model/select", dependencies=auth)
async def select_model(selection: ModelSelection):
    """Switch active model."""
    global active_model

    if selection.model_type not in predictors:
        raise HTTPException(
            status_code=404, detail=f"Model '{selection.model_type}' not loaded"
        )

    active_model = selection.model_type
    logger.info(f"Switched to {active_model} model")

    return {
        "active_model": active_model,
        "message": f"Switched to {active_model} model",
    }


@app.post("/predict", response_model=IAQResponse, dependencies=auth)
async def predict_iaq(reading: SensorReading):
    """Predict IAQ from sensor reading with enhanced statistics."""
    if active_model not in predictors:
        if not predictors:
            raise HTTPException(
                status_code=503,
                detail="No trained models available. Train with: python -m iaq4j train --model mlp",
            )
        raise HTTPException(
            status_code=503,
            detail=f"Active model '{active_model}' not available. Loaded models: {list(predictors.keys())}",
        )

    try:
        engine = inference_engines[active_model]
        sensor_readings = reading.get_readings()

        result = engine.predict_single(
            sensor_readings,
            prior_variables=reading.prior_variables,
            sensor_id=reading.sensor_id,
            sequence_number=reading.sequence_number,
            timestamp=reading.timestamp,
        )

        # Log prediction to InfluxDB if enabled and prediction was successful
        if result.get("status") == "ready" and result.get("iaq") is not None:
            identity = settings.get_sensor_identity()
            influx_manager.write_prediction(
                timestamp=reading.timestamp,
                readings=sensor_readings,
                iaq_predicted=result["iaq"],
                model_type=active_model,
                iaq_actual=reading.iaq_actual,
                sensor_id=reading.sensor_id or identity.get("sensor_id"),
                firmware_version=reading.firmware_version
                or identity.get("firmware_version"),
            )

        return IAQResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/uncertainty", dependencies=auth)
async def predict_with_uncertainty(reading: SensorReading):
    """Predict IAQ with uncertainty estimation (all model types).

    Models with dropout layers (MLP, LSTM, CNN) use MC dropout.
    Models without dropout (KAN) use history-based uncertainty.
    """
    if active_model not in inference_engines:
        raise HTTPException(
            status_code=503, detail=f"Active model '{active_model}' not available"
        )

    try:
        engine = inference_engines[active_model]

        result = engine.predict_with_uncertainty(
            reading.get_readings(),
            n_samples=20,
            prior_variables=reading.prior_variables,
        )

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/compare", dependencies=auth)
async def predict_compare(reading: SensorReading):
    """Get predictions from all available models."""
    if not predictors:
        raise HTTPException(status_code=503, detail="No models available")

    results = {}

    for name, predictor in predictors.items():
        try:
            result = predictor.predict(reading.get_readings())
            results[name] = result
        except Exception as e:
            logger.error(f"Error with {name} model: {e}")
            results[name] = {"error": str(e)}

    return {"models": results, "reading": reading.model_dump()}


@app.post("/reset/all", dependencies=auth)
async def reset_all_buffers():
    """Reset buffers for all models."""
    for predictor in predictors.values():
        predictor.buffer = []

    for engine in inference_engines.values():
        engine.reset_history()

    return {"status": "all buffers reset", "models": list(predictors.keys())}


@app.post("/reset/{model_type}", dependencies=auth)
async def reset_buffer(model_type: str):
    """Reset the sliding window buffer for a specific model."""
    if model_type not in predictors:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found")

    predictors[model_type].buffer = []

    return {
        "model": model_type,
        "status": "buffer reset",
        "window_size": predictors[model_type].window_size,
    }


@app.get("/statistics")
async def get_statistics():
    """Get prediction statistics from active model."""
    if active_model not in inference_engines:
        raise HTTPException(status_code=503, detail="No active model")

    engine = inference_engines[active_model]
    stats = engine.get_statistics()

    return {"model": active_model, "statistics": stats}


@app.get("/statistics/{model_type}")
async def get_model_statistics(model_type: str):
    """Get prediction statistics for specific model."""
    if model_type not in inference_engines:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found")

    engine = inference_engines[model_type]
    stats = engine.get_statistics()

    return {"model": model_type, "statistics": stats}


@app.get("/health/sensor")
async def check_sensor_health():
    """Analyze potential sensor drift or calibration issues."""
    if active_model not in inference_engines:
        raise HTTPException(status_code=503, detail="No active model")

    engine = inference_engines[active_model]
    analysis = engine.analyze_sensor_drift()

    if analysis is None:
        return {
            "status": "insufficient_data",
            "message": "Need at least 50 predictions to analyze sensor health",
        }

    return {"model": active_model, "analysis": analysis}


# =========================================================================
# Sensor registration (field mapping API)
# =========================================================================


@app.post("/sensors/register", response_model=SensorRegisterResponse, dependencies=auth)
async def register_sensor(req: SensorRegisterRequest):
    """Run the field mapper on submitted field names and return a proposed mapping."""
    from app.field_mapper import FieldMapper, MappingResult
    from app.profiles import get_sensor_profile

    profile = get_sensor_profile()
    mapper = FieldMapper(profile)

    result = mapper.map_fields(
        req.fields,
        sample_values=req.sample_values,
        backend=req.backend,
    )

    mapping_id = str(uuid.uuid4())
    _pending_mappings[mapping_id] = {
        "result": result,
        "sensor_id": req.sensor_id,
        "firmware_version": req.firmware_version,
    }

    return SensorRegisterResponse(
        mapping_id=mapping_id,
        status="proposed",
        mapping=[
            FieldMatchResponse(
                source_field=m.source_field,
                target_feature=m.target_feature,
                target_quantity=m.target_quantity,
                confidence=m.confidence,
                method=m.method,
            )
            for m in result.matches
        ],
        unresolved=result.unresolved,
    )


@app.post(
    "/sensors/register/{mapping_id}/confirm",
    response_model=SensorConfirmResponse,
    dependencies=auth,
)
async def confirm_sensor_mapping(mapping_id: str, req: SensorConfirmRequest = None):
    """Persist a proposed mapping to model_config.yaml."""
    import yaml

    if mapping_id not in _pending_mappings:
        raise HTTPException(
            status_code=404, detail=f"Mapping '{mapping_id}' not found or expired"
        )

    pending = _pending_mappings.pop(mapping_id)
    result = pending["result"]
    sensor_id = pending.get("sensor_id")
    firmware_version = pending.get("firmware_version")

    # Build field_mapping dict
    field_mapping = {m.source_field: m.target_feature for m in result.matches}

    # Apply overrides
    if req and req.overrides:
        field_mapping.update(req.overrides)

    # Save to model_config.yaml
    config_path = Path(__file__).resolve().parent.parent / "model_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("sensor", {})["field_mapping"] = field_mapping

    # Persist sensor identity if provided
    if sensor_id or firmware_version:
        cfg.setdefault("sensor", {}).setdefault("identity", {})
        if sensor_id:
            cfg["sensor"]["identity"]["sensor_id"] = sensor_id
        if firmware_version:
            cfg["sensor"]["identity"]["firmware_version"] = firmware_version

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    settings.invalidate_config_cache()
    configure_field_mapping(field_mapping)
    logger.info("Field mapping saved: %s", field_mapping)

    return SensorConfirmResponse(
        status="confirmed",
        field_mapping=field_mapping,
        sensor_id=sensor_id,
        firmware_version=firmware_version,
    )


@app.get("/sensors", dependencies=auth)
async def get_sensor_mapping():
    """Return the active field mapping from config."""
    cfg = settings.load_model_config()
    field_mapping = cfg.get("sensor", {}).get("field_mapping", {})
    return {
        "sensor_type": cfg.get("sensor", {}).get("type", "unknown"),
        "field_mapping": field_mapping,
        "identity": cfg.get("sensor", {}).get("identity", {}),
    }


@app.delete("/sensors/mapping", dependencies=auth)
async def delete_sensor_mapping():
    """Remove the active field mapping from model_config.yaml."""
    import yaml

    config_path = Path(__file__).resolve().parent.parent / "model_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sensor = cfg.get("sensor", {})
    if "field_mapping" not in sensor:
        return {"status": "no_mapping", "message": "No field mapping configured"}

    del sensor["field_mapping"]

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    settings.invalidate_config_cache()
    configure_field_mapping({})
    logger.info("Field mapping removed from config")
    return {"status": "removed"}


@app.get("/sensors/{sensor_id}/sequence", dependencies=auth)
async def get_sensor_sequence(sensor_id: str):
    """Get the current sequence state for a sensor.

    Returns the last seen sequence number for the specified sensor,
    enabling clients to check for dropped readings or sequence gaps.
    """
    engine = inference_engines.get(active_model)
    if engine is None:
        raise HTTPException(status_code=503, detail="No active model loaded")

    return engine.get_sequence_state(sensor_id)


@app.get("/sensors/{sensor_id}/drift", dependencies=auth)
async def get_sensor_drift(sensor_id: str):
    """Get drift report for a specific sensor.

    Uses BME680 3-year drift coefficients as the canonical drift profile.
    Requires at least 10 readings from this sensor_id to generate a report.
    """
    engine = inference_engines.get(active_model)
    if engine is None:
        raise HTTPException(status_code=503, detail="No active model loaded")

    report = engine.get_sensor_drift_report(sensor_id)
    if report is None:
        return {
            "status": "insufficient_data",
            "sensor_id": sensor_id,
            "message": "Need at least 10 readings to generate drift report",
        }

    return report


@app.post("/sensors/{sensor_id}/drift/save", dependencies=auth)
async def save_sensor_drift(sensor_id: str):
    """Persist drift state for a sensor to disk."""
    engine = inference_engines.get(active_model)
    if engine is None:
        raise HTTPException(status_code=503, detail="No active model loaded")

    path = engine.save_sensor_drift(sensor_id)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No drift state for sensor '{sensor_id}'",
        )

    return {"status": "saved", "sensor_id": sensor_id, "path": str(path)}


@app.get("/sensors/drift/list", dependencies=auth)
async def list_sensor_drift_reports():
    """List all sensor_ids that have drift data."""
    engine = inference_engines.get(active_model)
    if engine is None:
        raise HTTPException(status_code=503, detail="No active model loaded")

    sensor_ids = engine.list_sensor_drift_reports()
    return {"sensor_ids": sensor_ids, "count": len(sensor_ids)}
