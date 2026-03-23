"""PredictionService — owns model lifecycle, prediction dispatch, and mutable state.

Extracted from app/main.py (R3) to make state testable and thread-safe-ready.
"""

import logging
from typing import Any, Dict, List, Optional

from app.config import settings
from app.inference import InferenceEngine
from app.models import IAQPredictor, _KAN_AVAILABLE

logger = logging.getLogger(__name__)

MODEL_PATHS = {
    "mlp": settings.MLP_MODEL_PATH,
    "kan": settings.KAN_MODEL_PATH,
    "lstm": settings.LSTM_MODEL_PATH,
    "cnn": settings.CNN_MODEL_PATH,
    "bnn": settings.BNN_MODEL_PATH,
}


class PredictionService:
    """Manages loaded models, inference engines, and the active model selection."""

    def __init__(self) -> None:
        self._predictors: Dict[str, Any] = {}
        self._engines: Dict[str, InferenceEngine] = {}
        self._active_model: str = settings.DEFAULT_MODEL

    # -- Lifecycle -------------------------------------------------------------

    def load_models(self) -> None:
        """Load all available models from disk.  Called once during app startup."""
        for model_type, model_path in MODEL_PATHS.items():
            try:
                self._load_single(model_type, model_path)
            except Exception as e:
                logger.warning("Failed to load %s model: %s", model_type.upper(), e)

        if not self._predictors:
            logger.error(
                "No trained models found. The service will start in degraded mode — "
                "all prediction endpoints will return 503.\n"
                "  Train models with:  python -m iaq4j train --model mlp --epochs 200\n"
                "  Or create dummies:  python training/create_dummy_models.py"
            )
        else:
            if self._active_model not in self._predictors:
                self._active_model = next(iter(self._predictors))
                logger.warning(
                    "Default model '%s' not available. Falling back to '%s'.",
                    settings.DEFAULT_MODEL,
                    self._active_model,
                )
            logger.info(
                "Active model: %s  |  Available: %s",
                self._active_model,
                list(self._predictors.keys()),
            )

    def _load_single(self, model_type: str, model_path: str) -> None:
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
                return
            self._predictors[model_type] = predictor
            self._engines[model_type] = InferenceEngine(predictor)
            logger.info("KAN model loaded via remote sidecar at %s", settings.KAN_REMOTE_URL)
            return

        if model_type == "kan" and not _KAN_AVAILABLE:
            logger.warning(
                "KAN not available (Python >3.9?) and KAN_REMOTE_URL not set — skipping KAN"
            )
            return

        predictor = IAQPredictor(model_type=model_type, window_size=window_size)
        if not predictor.load_model(model_path):
            logger.warning("No trained %s model found at %s", model_type.upper(), model_path)
            return
        self._predictors[model_type] = predictor
        self._engines[model_type] = InferenceEngine(predictor)
        logger.info("%s model loaded successfully", model_type.upper())

    # -- Queries ---------------------------------------------------------------

    @property
    def active_model(self) -> str:
        return self._active_model

    @property
    def available_models(self) -> List[str]:
        return list(self._predictors.keys())

    @property
    def has_models(self) -> bool:
        return bool(self._predictors)

    def get_predictor(self, model_type: str) -> Optional[Any]:
        return self._predictors.get(model_type)

    def get_engine(self, model_type: str) -> Optional[InferenceEngine]:
        return self._engines.get(model_type)

    def list_models(self) -> dict:
        models_info = {}
        for name, predictor in self._predictors.items():
            models_info[name] = {
                "loaded": True,
                "window_size": predictor.window_size,
                "config": predictor.config,
            }
        return {"active": self._active_model, "available": models_info}

    def models_available_map(self) -> Dict[str, bool]:
        return {m: m in self._predictors for m in MODEL_PATHS}

    # -- Commands --------------------------------------------------------------

    def select_model(self, model_type: str) -> str:
        """Switch the active model.  Raises KeyError if not loaded."""
        if model_type not in self._predictors:
            raise KeyError(f"Model '{model_type}' not loaded")
        self._active_model = model_type
        logger.info("Switched to %s model", model_type)
        return model_type

    def predict(self, reading, prior_variables=None, sensor_id=None,
                sequence_number=None, timestamp=None) -> dict:
        """Run prediction on the active model's engine.  Returns result dict."""
        engine = self._engines.get(self._active_model)
        if engine is None:
            raise RuntimeError(f"Active model '{self._active_model}' has no engine")
        sensor_readings = reading.get_readings()
        return engine.predict_single(
            sensor_readings,
            prior_variables=prior_variables,
            sensor_id=sensor_id,
            sequence_number=sequence_number,
            timestamp=timestamp,
        )

    def predict_with_uncertainty(self, reading, n_samples: int = 20,
                                 prior_variables=None) -> dict:
        engine = self._engines.get(self._active_model)
        if engine is None:
            raise RuntimeError(f"Active model '{self._active_model}' has no engine")
        return engine.predict_with_uncertainty(
            reading.get_readings(),
            n_samples=n_samples,
            prior_variables=prior_variables,
        )

    def predict_compare(self, reading) -> dict:
        results = {}
        for name, predictor in self._predictors.items():
            try:
                results[name] = predictor.predict(reading.get_readings())
            except Exception as e:
                logger.error("Error with %s model: %s", name, e)
                results[name] = {"error": str(e)}
        return results

    def reset_model(self, model_type: str) -> dict:
        """Reset buffer for a specific model."""
        if model_type not in self._predictors:
            raise KeyError(f"Model '{model_type}' not found")
        self._predictors[model_type].buffer = []
        return {
            "model": model_type,
            "status": "buffer reset",
            "window_size": self._predictors[model_type].window_size,
        }

    def reset_all(self) -> dict:
        for predictor in self._predictors.values():
            predictor.buffer = []
        for engine in self._engines.values():
            engine.reset_history()
        return {"status": "all buffers reset", "models": list(self._predictors.keys())}

    def get_statistics(self, model_type: Optional[str] = None) -> dict:
        mt = model_type or self._active_model
        engine = self._engines.get(mt)
        if engine is None:
            raise KeyError(f"Model '{mt}' not found")
        return engine.get_statistics()

    def analyze_sensor_drift(self):
        engine = self._engines.get(self._active_model)
        if engine is None:
            raise KeyError(f"Active model '{self._active_model}' has no engine")
        return engine.analyze_sensor_drift()
