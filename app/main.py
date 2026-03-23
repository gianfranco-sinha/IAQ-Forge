# ============================================================================
# File: app/main.py
# ============================================================================
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os
from datetime import datetime

from app.schemas import (
    SensorReading,
    IAQResponse,
    HealthResponse,
    ModelSelection,
    SensorRegisterRequest,
    SensorRegisterResponse,
    FieldMatchResponse,
    SensorConfirmRequest,
    SensorConfirmResponse,
    StructuredResponse,
    configure_field_mapping,
)
from app.config import settings
from app.database import influx_manager
from app.exceptions import IAQError
from app.prediction_service import PredictionService, MODEL_PATHS
from app.sensor_registration_service import SensorRegistrationService
import app.builtin_profiles  # noqa: F401  — registers sensor/standard profiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service instances — created in lifespan, used by route handlers
prediction_svc: PredictionService = None  # type: ignore[assignment]
registration_svc: SensorRegistrationService = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global prediction_svc, registration_svc

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

    prediction_svc = PredictionService()
    prediction_svc.load_models()

    registration_svc = SensorRegistrationService()

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


# ---------------------------------------------------------------------------
# Health & model management
# ---------------------------------------------------------------------------


@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if prediction_svc.has_models else "degraded",
        models_available=prediction_svc.models_available_map(),
        active_model=prediction_svc.active_model,
    )


@app.get("/health/detailed")
async def detailed_health_check():
    return {
        "service": {
            "status": "healthy" if prediction_svc.has_models else "degraded",
            "models_loaded": prediction_svc.available_models,
            "active_model": prediction_svc.active_model,
        },
        "database": influx_manager.health_check(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/models", response_model=dict)
async def list_models():
    return prediction_svc.list_models()


@app.post("/model/select", dependencies=auth)
async def select_model(selection: ModelSelection):
    try:
        prediction_svc.select_model(selection.model_type)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Model '{selection.model_type}' not loaded"
        )
    return {
        "active_model": prediction_svc.active_model,
        "message": f"Switched to {prediction_svc.active_model} model",
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=IAQResponse, dependencies=auth)
async def predict_iaq(reading: SensorReading):
    if not prediction_svc.has_models:
        raise HTTPException(
            status_code=503,
            detail="No trained models available. Train with: python -m iaq4j train --model mlp",
        )

    try:
        result = prediction_svc.predict(
            reading,
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
                readings=reading.get_readings(),
                iaq_predicted=result["iaq"],
                model_type=prediction_svc.active_model,
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
    if not prediction_svc.has_models:
        raise HTTPException(
            status_code=503, detail=f"Active model '{prediction_svc.active_model}' not available"
        )
    try:
        return prediction_svc.predict_with_uncertainty(
            reading, prior_variables=reading.prior_variables,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/compare", dependencies=auth)
async def predict_compare(reading: SensorReading):
    if not prediction_svc.has_models:
        raise HTTPException(status_code=503, detail="No models available")
    return {"models": prediction_svc.predict_compare(reading), "reading": reading.model_dump()}


# ---------------------------------------------------------------------------
# Buffer reset
# ---------------------------------------------------------------------------


@app.post("/reset/all", dependencies=auth)
async def reset_all_buffers():
    return prediction_svc.reset_all()


@app.post("/reset/{model_type}", dependencies=auth)
async def reset_buffer(model_type: str):
    try:
        return prediction_svc.reset_model(model_type)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found")


# ---------------------------------------------------------------------------
# Statistics & sensor health
# ---------------------------------------------------------------------------


@app.get("/statistics")
async def get_statistics():
    try:
        stats = prediction_svc.get_statistics()
    except KeyError:
        raise HTTPException(status_code=503, detail="No active model")
    return {"model": prediction_svc.active_model, "statistics": stats}


@app.get("/statistics/{model_type}")
async def get_model_statistics(model_type: str):
    try:
        stats = prediction_svc.get_statistics(model_type)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found")
    return {"model": model_type, "statistics": stats}


@app.get("/health/sensor")
async def check_sensor_health():
    try:
        analysis = prediction_svc.analyze_sensor_drift()
    except KeyError:
        raise HTTPException(status_code=503, detail="No active model")

    if analysis is None:
        return {
            "status": "insufficient_data",
            "message": "Need at least 50 predictions to analyze sensor health",
        }
    return {"model": prediction_svc.active_model, "analysis": analysis}


# =========================================================================
# Sensor registration (field mapping API)
# =========================================================================


@app.post("/sensors/register", response_model=SensorRegisterResponse, dependencies=auth)
async def register_sensor(req: SensorRegisterRequest):
    mapping_id, result = registration_svc.propose_mapping(
        req.fields,
        sample_values=req.sample_values,
        backend=req.backend,
        sensor_id=req.sensor_id,
        firmware_version=req.firmware_version,
    )
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
    try:
        confirmed = registration_svc.confirm_mapping(
            mapping_id, overrides=req.overrides if req else None,
        )
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Mapping '{mapping_id}' not found or expired"
        )
    return SensorConfirmResponse(
        status="confirmed",
        field_mapping=confirmed["field_mapping"],
        sensor_id=confirmed["sensor_id"],
        firmware_version=confirmed["firmware_version"],
    )


@app.get("/sensors", dependencies=auth)
async def get_sensor_mapping():
    return registration_svc.get_mapping()


@app.delete("/sensors/mapping", dependencies=auth)
async def delete_sensor_mapping():
    return registration_svc.delete_mapping()


@app.get("/sensors/{sensor_id}/sequence", dependencies=auth)
async def get_sensor_sequence(sensor_id: str):
    engine = prediction_svc.get_engine(prediction_svc.active_model)
    if engine is None:
        raise HTTPException(status_code=503, detail="No active model loaded")
    return engine.get_sequence_state(sensor_id)


@app.get("/sensors/{sensor_id}/drift", dependencies=auth)
async def get_sensor_drift(sensor_id: str):
    engine = prediction_svc.get_engine(prediction_svc.active_model)
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
    engine = prediction_svc.get_engine(prediction_svc.active_model)
    if engine is None:
        raise HTTPException(status_code=503, detail="No active model loaded")
    path = engine.save_sensor_drift(sensor_id)
    if path is None:
        raise HTTPException(
            status_code=404, detail=f"No drift state for sensor '{sensor_id}'"
        )
    return {"status": "saved", "sensor_id": sensor_id, "path": str(path)}


@app.get("/sensors/drift/list", dependencies=auth)
async def list_sensor_drift_reports():
    engine = prediction_svc.get_engine(prediction_svc.active_model)
    if engine is None:
        raise HTTPException(status_code=503, detail="No active model loaded")
    sensor_ids = engine.list_sensor_drift_reports()
    return {"sensor_ids": sensor_ids, "count": len(sensor_ids)}
