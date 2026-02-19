# ============================================================================
# File: app/inference.py
# ============================================================================
"""
Inference logic for IAQ prediction.
Handles batch processing, streaming, and real-time predictions.
"""

import logging
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class InferenceEngine:
    """High-level inference engine for IAQ prediction."""

    def __init__(self, predictor):
        """
        Args:
            predictor: IAQPredictor instance
        """
        self.predictor = predictor
        self.prediction_history = []
        self.max_history = 1000

    def _compute_prior(self) -> Optional[dict]:
        """Compute a prior belief about IAQ from recent history or training distribution."""
        if len(self.prediction_history) >= 5:
            window = min(20, len(self.prediction_history))
            recent_iaqs = [h['iaq'] for h in self.prediction_history[-window:]]
            return {
                "mean": float(np.mean(recent_iaqs)),
                "std": float(np.std(recent_iaqs)),
                "source": "history_window",
                "n_observations": window,
            }

        if self.predictor.target_scaler is not None:
            scaler = self.predictor.target_scaler
            if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
                lo = float(scaler.data_min_[0])
                hi = float(scaler.data_max_[0])
                return {
                    "mean": (lo + hi) / 2.0,
                    "std": (hi - lo) / 4.0,
                    "source": "training_distribution",
                    "n_observations": 0,
                }

        return None

    def predict_single(self, readings: dict = None,
                       include_uncertainty: bool = True,
                       n_mc_samples: int = 10, **kwargs) -> Dict:
        """Single prediction with prior, uncertainty, and structured output.

        Args:
            readings: sensor readings dict.
            include_uncertainty: whether to run MC dropout for uncertainty.
            n_mc_samples: number of MC dropout forward passes.
        """
        if readings is None:
            readings = kwargs

        # Compute prior BEFORE the new prediction
        prior = self._compute_prior()

        mc = n_mc_samples if include_uncertainty else 1
        result = self.predictor.predict(readings, n_mc_samples=mc)

        # Attach prior
        if prior is not None:
            result["prior"] = prior

        # Add to history if prediction was successful
        if result.get('status') == 'ready' and result.get('iaq') is not None:
            history_entry = dict(readings)
            history_entry['iaq'] = result['iaq']
            self.prediction_history.append(history_entry)

            if len(self.prediction_history) > self.max_history:
                self.prediction_history.pop(0)

            # Supplement with history-based uncertainty for models without dropout
            predicted = result.get("predicted", {})
            unc = predicted.get("uncertainty", {})
            if (unc.get("method") == "deterministic"
                    and len(self.prediction_history) >= 10):
                recent_iaqs = [h['iaq'] for h in self.prediction_history[-20:]]
                result["predicted"]["uncertainty"] = {
                    "std": float(np.std(recent_iaqs)),
                    "ci_lower": float(np.percentile(recent_iaqs, 2.5)),
                    "ci_upper": float(np.percentile(recent_iaqs, 97.5)),
                    "method": "history_std",
                }
                result["inference"]["uncertainty_method"] = "history_std"

            # Add statistics if we have history
            if len(self.prediction_history) >= 10:
                recent_iaqs = [h['iaq'] for h in self.prediction_history[-10:]]
                result['statistics'] = {
                    'recent_avg': np.mean(recent_iaqs),
                    'recent_std': np.std(recent_iaqs),
                    'recent_min': np.min(recent_iaqs),
                    'recent_max': np.max(recent_iaqs),
                    'trend': 'improving' if recent_iaqs[-1] < recent_iaqs[0] else 'worsening'
                }

        return result

    def predict_batch(self, readings: List[Dict]) -> List[Dict]:
        """Batch prediction for multiple readings."""
        return [self.predict_single(reading) for reading in readings]

    def predict_with_uncertainty(self, readings: dict = None,
                                 n_samples: int = 10, **kwargs) -> Dict:
        """Prediction with uncertainty estimation.

        Delegates to predict_single with MC dropout enabled.
        Works for all model types — models without dropout get history-based
        uncertainty instead.
        """
        return self.predict_single(
            readings=readings or kwargs,
            include_uncertainty=True,
            n_mc_samples=n_samples,
        )

    def get_statistics(self) -> Dict:
        """Get statistics from prediction history."""
        if not self.prediction_history:
            return {
                'count': 0,
                'message': 'No predictions yet'
            }

        iaqs = [h['iaq'] for h in self.prediction_history]

        from app.profiles import get_iaq_standard
        standard = get_iaq_standard()

        return {
            'count': len(self.prediction_history),
            'iaq_mean': float(np.mean(iaqs)),
            'iaq_std': float(np.std(iaqs)),
            'iaq_min': float(np.min(iaqs)),
            'iaq_max': float(np.max(iaqs)),
            'percentiles': {
                'p25': float(np.percentile(iaqs, 25)),
                'p50': float(np.percentile(iaqs, 50)),
                'p75': float(np.percentile(iaqs, 75)),
                'p95': float(np.percentile(iaqs, 95))
            },
            'distribution': standard.category_distribution(iaqs),
        }

    def reset_history(self):
        """Clear prediction history."""
        self.prediction_history = []
        logger.info("Prediction history cleared")

    def analyze_sensor_drift(self) -> Optional[Dict]:
        """Analyze if sensor might be drifting based on prediction history."""
        if len(self.prediction_history) < 50:
            return None

        from app.profiles import get_sensor_profile
        profile = get_sensor_profile()

        recent = self.prediction_history[-50:]

        # Check stability of each raw feature
        feature_stability = {}
        warnings = []

        for feat in profile.raw_features:
            values = [h.get(feat) for h in recent if feat in h]
            if not values:
                continue
            feat_mean = np.mean(values)
            feat_std = np.std(values)
            cv = float(feat_std / feat_mean) if feat_mean != 0 else 0.0
            feature_stability[feat] = {
                'mean': float(feat_mean),
                'std': float(feat_std),
                'cv': cv,
            }
            if cv > 0.3:
                warnings.append(f"High {feat} variability (CV={cv:.2f})")
            elif cv < 0.01 and feat_mean != 0:
                warnings.append(f"{feat} appears stuck (CV={cv:.4f})")

        # Check IAQ trend
        iaqs = [h['iaq'] for h in recent]
        iaq_trend = np.polyfit(range(len(iaqs)), iaqs, 1)[0]

        if abs(iaq_trend) > 1.0:
            warnings.append("Rapid IAQ trend detected")

        return {
            'feature_stability': feature_stability,
            'iaq_trend': float(iaq_trend),
            'warnings': warnings,
            'health': 'good' if not warnings else 'warning'
        }


class StreamingInference:
    """Handle streaming/continuous inference with rate limiting."""

    def __init__(self, predictor, max_rate_hz: float = 1.0):
        """
        Args:
            predictor: IAQPredictor instance
            max_rate_hz: Maximum prediction rate in Hz
        """
        self.predictor = predictor
        self.min_interval = 1.0 / max_rate_hz
        self.last_prediction_time = 0
        self.dropped_readings = 0

    def predict(self, readings: dict = None, force: bool = False,
                **kwargs) -> Optional[Dict]:
        """Predict with rate limiting.

        Accepts a readings dict or keyword args for backward compatibility.
        """
        if readings is None:
            readings = kwargs

        import time
        current_time = time.time()

        if not force and (current_time - self.last_prediction_time) < self.min_interval:
            self.dropped_readings += 1
            return None

        result = self.predictor.predict(readings)
        self.last_prediction_time = current_time

        return result

    def get_stats(self) -> Dict:
        """Get streaming statistics."""
        return {
            'dropped_readings': self.dropped_readings,
            'max_rate_hz': 1.0 / self.min_interval,
            'buffer_status': {
                'current_size': len(self.predictor.buffer),
                'required_size': self.predictor.window_size,
                'ready': len(self.predictor.buffer) >= self.predictor.window_size
            }
        }
