# ============================================================================
# File: app/main.py
# ============================================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from app.models import IAQPredictor
from app.inference import InferenceEngine
from app.schemas import (
    SensorReading, IAQResponse, ModelInfo, HealthResponse, ModelSelection
)
from app.config import settings
from app.database import influx_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictors and inference engines
predictors = {}
inference_engines = {}
active_model = settings.DEFAULT_MODEL


MODEL_PATHS = {
    "mlp": settings.MLP_MODEL_PATH,
    "kan": settings.KAN_MODEL_PATH,
    "lstm": settings.LSTM_MODEL_PATH,
    "cnn": settings.CNN_MODEL_PATH,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global predictors, inference_engines, active_model

    logger.info("Loading IAQ prediction models...")

    for model_type, model_path in MODEL_PATHS.items():
        try:
            predictor = IAQPredictor(model_type=model_type, window_size=settings.WINDOW_SIZE)
            if not predictor.load_model(model_path):
                logger.warning("No trained %s model found at %s", model_type.upper(), model_path)
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
            "  Train models with:  python -m iaqforge train --model mlp --epochs 200\n"
            "  Or create dummies:  python training/create_dummy_models.py"
        )
    else:
        if active_model not in predictors:
            fallback = next(iter(predictors))
            logger.warning(
                "Default model '%s' not available. Falling back to '%s'.",
                active_model, fallback,
            )
            active_model = fallback
        logger.info("Active model: %s  |  Available: %s", active_model, list(predictors.keys()))

    # Check InfluxDB connection
    if settings.INFLUX_ENABLED:
        db_status = influx_manager.health_check()
        if db_status['status'] == 'healthy':
            logger.info("InfluxDB connection established")
        else:
            logger.warning(f"InfluxDB unavailable: {db_status.get('error', 'Unknown error')}")

    yield

    # Cleanup
    influx_manager.close()
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="ML-based BSEC IAQ index reproduction from raw BME680 sensor data",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictors else "degraded",
        models_available={
            'mlp': 'mlp' in predictors,
            'kan': 'kan' in predictors
        },
        active_model=active_model
    )


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check including database status."""
    db_health = influx_manager.health_check()

    return {
        'service': {
            'status': 'healthy' if predictors else 'degraded',
            'models_loaded': list(predictors.keys()),
            'active_model': active_model
        },
        'database': db_health,
        'timestamp': datetime.utcnow().isoformat()
    }


@app.get("/models", response_model=dict)
async def list_models():
    """List available models and their info."""
    models_info = {}

    for name, predictor in predictors.items():
        models_info[name] = {
            'loaded': True,
            'window_size': predictor.window_size,
            'config': predictor.config
        }

    return {
        'active': active_model,
        'available': models_info
    }


@app.post("/model/select")
async def select_model(selection: ModelSelection):
    """Switch active model."""
    global active_model

    if selection.model_type not in predictors:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{selection.model_type}' not loaded"
        )

    active_model = selection.model_type
    logger.info(f"Switched to {active_model} model")

    return {
        'active_model': active_model,
        'message': f'Switched to {active_model} model'
    }


@app.post("/predict", response_model=IAQResponse)
async def predict_iaq(reading: SensorReading):
    """Predict IAQ from sensor reading with enhanced statistics."""
    if active_model not in predictors:
        if not predictors:
            raise HTTPException(
                status_code=503,
                detail="No trained models available. Train with: python -m iaqforge train --model mlp",
            )
        raise HTTPException(
            status_code=503,
            detail=f"Active model '{active_model}' not available. Loaded models: {list(predictors.keys())}",
        )

    try:
        engine = inference_engines[active_model]

        result = engine.predict_single(
            reading.temperature,
            reading.rel_humidity,
            reading.pressure,
            reading.gas_resistance
        )

        # Log prediction to InfluxDB if enabled and prediction was successful
        if result.get('status') == 'ready' and result.get('iaq') is not None:
            timestamp = reading.timestamp if reading.timestamp else datetime.utcnow().isoformat()
            influx_manager.write_prediction(
                timestamp=timestamp,
                temperature=reading.temperature,
                rel_humidity=reading.rel_humidity,
                pressure=reading.pressure,
                gas_resistance=reading.gas_resistance,
                iaq_predicted=result['iaq'],
                model_type=active_model
            )

        return IAQResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/uncertainty")
async def predict_with_uncertainty(reading: SensorReading):
    """Predict IAQ with uncertainty estimation (MLP only)."""
    if active_model not in inference_engines:
        raise HTTPException(
            status_code=503,
            detail=f"Active model '{active_model}' not available"
        )

    if active_model != 'mlp':
        raise HTTPException(
            status_code=400,
            detail="Uncertainty estimation only available for MLP model"
        )

    try:
        engine = inference_engines[active_model]

        result = engine.predict_with_uncertainty(
            reading.temperature,
            reading.rel_humidity,
            reading.pressure,
            reading.gas_resistance,
            n_samples=10
        )

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/compare")
async def predict_compare(reading: SensorReading):
    """Get predictions from all available models."""
    if not predictors:
        raise HTTPException(status_code=503, detail="No models available")

    results = {}

    for name, predictor in predictors.items():
        try:
            result = predictor.predict(
                reading.temperature,
                reading.rel_humidity,
                reading.pressure,
                reading.gas_resistance
            )
            results[name] = result
        except Exception as e:
            logger.error(f"Error with {name} model: {e}")
            results[name] = {'error': str(e)}

    return {
        'models': results,
        'reading': reading.model_dump()
    }


@app.post("/reset/{model_type}")
async def reset_buffer(model_type: str):
    """Reset the sliding window buffer for a specific model."""
    if model_type not in predictors:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found")

    predictors[model_type].buffer = []

    return {
        'model': model_type,
        'status': 'buffer reset',
        'window_size': predictors[model_type].window_size
    }


@app.post("/reset/all")
async def reset_all_buffers():
    """Reset buffers for all models."""
    for predictor in predictors.values():
        predictor.buffer = []

    for engine in inference_engines.values():
        engine.reset_history()

    return {
        'status': 'all buffers reset',
        'models': list(predictors.keys())
    }


@app.get("/statistics")
async def get_statistics():
    """Get prediction statistics from active model."""
    if active_model not in inference_engines:
        raise HTTPException(status_code=503, detail="No active model")

    engine = inference_engines[active_model]
    stats = engine.get_statistics()

    return {
        'model': active_model,
        'statistics': stats
    }


@app.get("/statistics/{model_type}")
async def get_model_statistics(model_type: str):
    """Get prediction statistics for specific model."""
    if model_type not in inference_engines:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found")

    engine = inference_engines[model_type]
    stats = engine.get_statistics()

    return {
        'model': model_type,
        'statistics': stats
    }


@app.get("/health/sensor")
async def check_sensor_health():
    """Analyze potential sensor drift or calibration issues."""
    if active_model not in inference_engines:
        raise HTTPException(status_code=503, detail="No active model")

    engine = inference_engines[active_model]
    analysis = engine.analyze_sensor_drift()

    if analysis is None:
        return {
            'status': 'insufficient_data',
            'message': 'Need at least 50 predictions to analyze sensor health'
        }

    return {
        'model': active_model,
        'analysis': analysis
    }