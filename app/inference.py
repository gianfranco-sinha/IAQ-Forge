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

    def predict_single(self, temperature: float, rel_humidity: float,
                       pressure: float, gas_resistance: float) -> Dict:
        """
        Single prediction with enhanced metadata.

        Returns:
            Dictionary with prediction and metadata
        """
        result = self.predictor.predict(temperature, rel_humidity, pressure, gas_resistance)

        # Add to history if prediction was successful
        if result.get('status') == 'ready' and result.get('iaq') is not None:
            self.prediction_history.append({
                'iaq': result['iaq'],
                'temperature': temperature,
                'rel_humidity': rel_humidity,
                'pressure': pressure,
                'gas_resistance': gas_resistance
            })

            # Trim history if too large
            if len(self.prediction_history) > self.max_history:
                self.prediction_history.pop(0)

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
        """
        Batch prediction for multiple readings.

        Args:
            readings: List of dicts with keys: temperature, humidity, pressure, resistance

        Returns:
            List of prediction results
        """
        results = []

        for reading in readings:
            result = self.predict_single(
                reading['temperature'],
                reading['rel_humidity'],
                reading['pressure'],
                reading['gas_resistance']
            )
            results.append(result)

        return results

    def predict_with_uncertainty(self, temperature: float, rel_humidity: float,
                                 pressure: float, gas_resistance: float,
                                 n_samples: int = 10) -> Dict:
        """
        Prediction with uncertainty estimation using Monte Carlo dropout.
        Only works if model has dropout layers.

        Args:
            temperature, humidity, pressure, resistance: Sensor readings
            n_samples: Number of forward passes for uncertainty estimation

        Returns:
            Prediction with uncertainty bounds
        """
        # Enable dropout during inference for uncertainty estimation
        if self.predictor.model_type == 'mlp':
            self.predictor.model.train()  # Enables dropout

        predictions = []

        for _ in range(n_samples):
            result = self.predictor.predict(temperature, rel_humidity, pressure, gas_resistance)
            if result.get('iaq') is not None:
                predictions.append(result['iaq'])

        # Back to eval mode
        self.predictor.model.eval()

        if not predictions:
            return {
                'iaq': None,
                'status': 'error',
                'message': 'Could not generate predictions'
            }

        predictions = np.array(predictions)

        return {
            'iaq': float(np.mean(predictions)),
            'uncertainty': {
                'std': float(np.std(predictions)),
                'confidence_95_lower': float(np.percentile(predictions, 2.5)),
                'confidence_95_upper': float(np.percentile(predictions, 97.5)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            },
            'status': 'ready',
            'model_type': self.predictor.model_type,
            'n_samples': n_samples
        }

    def get_statistics(self) -> Dict:
        """Get statistics from prediction history."""
        if not self.prediction_history:
            return {
                'count': 0,
                'message': 'No predictions yet'
            }

        iaqs = [h['iaq'] for h in self.prediction_history]

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
            'distribution': {
                'excellent': sum(1 for iaq in iaqs if iaq <= 50),
                'good': sum(1 for iaq in iaqs if 50 < iaq <= 100),
                'moderate': sum(1 for iaq in iaqs if 100 < iaq <= 200),
                'poor': sum(1 for iaq in iaqs if 200 < iaq <= 300),
                'very_poor': sum(1 for iaq in iaqs if iaq > 300)
            }
        }

    def reset_history(self):
        """Clear prediction history."""
        self.prediction_history = []
        logger.info("Prediction history cleared")

    def analyze_sensor_drift(self) -> Optional[Dict]:
        """
        Analyze if sensor might be drifting based on prediction history.
        Useful for detecting sensor calibration issues.
        """
        if len(self.prediction_history) < 50:
            return None

        recent = self.prediction_history[-50:]

        # Check gas resistance stability
        gas_resistances = [h['gas_resistance'] for h in recent]
        res_mean = np.mean(gas_resistances)
        res_std = np.std(gas_resistances)

        # Check for unusual patterns
        iaqs = [h['iaq'] for h in recent]
        iaq_trend = np.polyfit(range(len(iaqs)), iaqs, 1)[0]

        warnings = []

        # Check for drift
        if abs(iaq_trend) > 1.0:  # IAQ changing by >1 point per reading
            warnings.append("Rapid IAQ trend detected")

        # Check for resistance instability
        if res_std / res_mean > 0.3:  # >30% variation
            warnings.append("High gas gas_resistance variability")

        # Check for stuck readings
        if res_std / res_mean < 0.01:  # <1% variation
            warnings.append("Gas gas_resistance appears stuck")

        return {
            'gas_resistance_stability': {
                'mean': float(res_mean),
                'std': float(res_std),
                'cv': float(res_std / res_mean)  # Coefficient of variation
            },
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

    def predict(self, temperature: float, rel_humidity: float,
                pressure: float, gas_resistance: float,
                force: bool = False) -> Optional[Dict]:
        """
        Predict with rate limiting.

        Args:
            force: If True, bypass rate limiting

        Returns:
            Prediction result or None if rate limited
        """
        import time
        current_time = time.time()

        if not force and (current_time - self.last_prediction_time) < self.min_interval:
            self.dropped_readings += 1
            return None

        result = self.predictor.predict(temperature, rel_humidity, pressure, gas_resistance)
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
