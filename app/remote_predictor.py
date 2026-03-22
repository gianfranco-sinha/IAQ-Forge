"""Remote KAN predictor — proxies only the forward pass to a sidecar container.

All feature engineering, buffering, scaling, clamping, and categorization happen
locally.  Only the raw tensor forward pass is sent over HTTP to the KAN sidecar.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import httpx
import joblib
import numpy as np
import torch

logger = logging.getLogger(__name__)


class RemoteKANPredictor:
    """Drop-in replacement for IAQPredictor when KAN runs in a remote sidecar.

    Conforms to the same interface that InferenceEngine expects:
    - ``predict(readings, n_mc_samples, timestamp)``
    - ``load_model(model_path) -> bool``
    - ``buffer``, ``window_size``, ``model_type``, ``config``, ``target_scaler``
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 0.5,
        window_size: int = 10,
        model_path: str = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model_type: str = "kan"
        self.window_size: int = window_size
        self.model_path: str = model_path
        self.model = None  # Not used locally, but InferenceEngine may check
        self.feature_scaler = None
        self.target_scaler = None
        self.config: Optional[dict] = None
        self.device: str = "cpu"
        self._baselines: dict = {}
        self.buffer: list = []
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

        from app.profiles import get_iaq_standard, get_sensor_profile
        self.sensor_profile = get_sensor_profile()
        self.iaq_standard = get_iaq_standard()

    def load_model(self, model_path: str) -> bool:
        """Load scalers and config locally, then health-check the sidecar."""
        model_dir = Path(model_path)

        try:
            # Load config
            config_path = model_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path) as f:
                self.config = json.load(f)

            # Load scalers
            feature_scaler_path = model_dir / "feature_scaler.pkl"
            target_scaler_path = model_dir / "target_scaler.pkl"
            if feature_scaler_path.exists():
                self.feature_scaler = joblib.load(feature_scaler_path)
            if target_scaler_path.exists():
                self.target_scaler = joblib.load(target_scaler_path)

            # Load baselines from config
            self._baselines = self.config.get("baselines", {})

            # Read window_size from config if saved
            if "window_size" in self.config:
                self.window_size = self.config["window_size"]

            # Health-check the sidecar
            resp = self._client.get("/health")
            resp.raise_for_status()
            health = resp.json()
            if not health.get("model_loaded"):
                logger.warning("KAN sidecar is healthy but model not loaded — calling /load")
                load_resp = self._client.post("/load")
                load_resp.raise_for_status()

            logger.info("RemoteKANPredictor connected to sidecar at %s", self.base_url)
            return True

        except httpx.HTTPError as e:
            logger.error("KAN sidecar unreachable at %s: %s", self.base_url, e)
            return False
        except Exception as e:
            logger.error("Failed to load remote KAN model: %s", e)
            return False

    def _forward_remote(self, tensor: np.ndarray) -> float:
        """Send a tensor to the sidecar /forward endpoint and return the prediction."""
        payload = {
            "tensor": tensor.tolist(),
            "shape": list(tensor.shape),
        }
        resp = self._client.post("/forward", json=payload)
        resp.raise_for_status()
        result = resp.json()
        return result["prediction"][0][0]

    def predict(
        self, readings: dict = None, n_mc_samples: int = 1, timestamp=None, **kwargs
    ) -> dict:
        """Predict IAQ from sensor readings via the remote KAN sidecar.

        Same flow as IAQPredictor.predict() but the model forward pass is remote.
        KAN is deterministic — n_mc_samples is ignored.
        """
        if readings is None:
            readings = kwargs

        try:
            # Step 1: profile-driven feature engineering
            features = self.sensor_profile.engineer_features_single(
                readings, self._baselines, timestamp=timestamp
            )

            # Step 2: buffer
            self.buffer.append(features)
            if len(self.buffer) > self.window_size:
                self.buffer.pop(0)

            if len(self.buffer) < self.window_size:
                return {
                    "iaq": None,
                    "status": "buffering",
                    "buffer_size": len(self.buffer),
                    "required": self.window_size,
                    "message": f"Collecting data... {len(self.buffer)}/{self.window_size}",
                }

            # Step 3: flatten window -> scale
            window_flat = np.array(self.buffer).flatten().reshape(1, -1)

            if self.feature_scaler is not None:
                window_flat = self.feature_scaler.transform(window_flat)

            # Step 4: remote forward pass
            raw_output = self._forward_remote(window_flat)

            # Step 5: inverse scale
            if self.target_scaler is not None:
                iaq_value = float(
                    self.target_scaler.inverse_transform(
                        np.array([[raw_output]])
                    )[0, 0]
                )
            else:
                iaq_value = float(raw_output)

            # Step 6: standard-driven clamp and categorize
            iaq_value = self.iaq_standard.clamp(iaq_value)
            category = self.iaq_standard.categorize(iaq_value)

            # Build engineered features dict
            eng_names = self.sensor_profile.engineered_feature_names
            eng_values = features[-len(eng_names):] if eng_names else []
            eng_dict = dict(zip(eng_names, [float(v) for v in eng_values]))

            predicted = {
                "mean": iaq_value,
                "category": category,
                "iaq_standard": self.iaq_standard.name,
            }

            result = {
                "iaq": iaq_value,
                "category": category,
                "status": "ready",
                "model_type": self.model_type,
                "raw_inputs": readings,
                "buffer_size": len(self.buffer),
                "required": self.window_size,
                "observation": {
                    "sensor_type": self.sensor_profile.name,
                    "readings": readings,
                    "engineered_features": eng_dict or None,
                },
                "predicted": predicted,
                "inference": {
                    "model_type": self.model_type,
                    "window_size": self.window_size,
                    "buffer_size": len(self.buffer),
                    "uncertainty_method": "deterministic",
                    "mc_samples": None,
                },
            }

            return result

        except httpx.HTTPError as e:
            return {
                "iaq": None,
                "status": "error",
                "message": f"KAN sidecar error: {e}",
            }
        except Exception as e:
            return {
                "iaq": None,
                "status": "error",
                "message": f"Prediction failed: {str(e)}",
            }

    def reset_buffer(self) -> None:
        """Reset the sliding window buffer."""
        self.buffer = []

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
