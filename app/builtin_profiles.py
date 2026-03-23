# ============================================================================
# File: app/builtin_profiles.py
# Built-in sensor profiles and IAQ standards.
# ============================================================================
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.profiles import (
    SensorProfile,
    register_sensor,
)
from app.sensor_physics import BME680_LP
from app.standards import register_yaml_standards


class BME680Profile(SensorProfile):
    """Bosch BME680/BME688 environmental sensor."""

    @property
    def name(self) -> str:
        return "bme680"

    @property
    def raw_features(self) -> List[str]:
        return ["temperature", "rel_humidity", "pressure", "voc_resistance"]

    @property
    def feature_quantities(self) -> Dict[str, str]:
        return {
            "temperature": "temperature",
            "rel_humidity": "relative_humidity",
            "pressure": "barometric_pressure",
            "voc_resistance": "voc_resistance",
        }

    @property
    def valid_ranges(self) -> Dict[str, Tuple[float, float]]:
        ranges = super().valid_ranges  # computed from registry
        ranges["iaq_accuracy"] = (2, 3)  # quality column, not a quantity
        return ranges

    @property
    def quality_column(self) -> Optional[str]:
        return "iaq_accuracy"

    @property
    def quality_min(self) -> Optional[float]:
        return 2

    @property
    def expected_interval_seconds(self) -> Optional[float]:
        return BME680_LP.interval_seconds

    @property
    def warmup_seconds(self) -> float:
        return BME680_LP.heater.warmup_seconds

    @property
    def heater_temperature_c(self) -> Optional[float]:
        return BME680_LP.heater.temperature_c

    @property
    def engineered_feature_names(self) -> List[str]:
        return [
            "abs_humidity",
            "baseline_24h", "gas_ratio_24h", "log_ratio_24h",
            "baseline_7d", "gas_ratio_7d", "log_ratio_7d",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        ]

    def _compute_envelope(
        self,
        voc_series: pd.Series,
        use_time_rolling: bool,
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute 24h and 7d rolling-max EWM baselines from VOC resistance."""
        if use_time_rolling:
            roll_24h = voc_series.rolling("24h", min_periods=1).max()
            roll_7d = voc_series.rolling("7D", min_periods=1).max()
        else:
            interval = self.expected_interval_seconds or 3.0
            win_24h = int(24 * 3600 / interval)
            win_7d = int(7 * 24 * 3600 / interval)
            roll_24h = voc_series.rolling(win_24h, min_periods=1).max()
            roll_7d = voc_series.rolling(win_7d, min_periods=1).max()

        baseline_24h = roll_24h.ewm(alpha=0.001).mean()
        baseline_7d = roll_7d.ewm(alpha=0.001).mean()
        return baseline_24h, baseline_7d

    def compute_baselines(
        self, raw: np.ndarray, timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        voc_idx = self.raw_features.index("voc_resistance")
        voc_col = raw[:, voc_idx]

        if timestamps is not None:
            voc_series = pd.Series(voc_col, index=pd.DatetimeIndex(timestamps))
            use_time = True
        else:
            voc_series = pd.Series(voc_col)
            use_time = False

        baseline_24h, baseline_7d = self._compute_envelope(voc_series, use_time)

        return {
            "voc_resistance": float(np.median(voc_col)),
            "baseline_24h": float(baseline_24h.iloc[-1]),
            "baseline_7d": float(baseline_7d.iloc[-1]),
        }

    def engineer_features(
        self,
        raw: np.ndarray,
        baselines: Optional[Dict[str, float]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        from app.quantities import calculate_absolute_humidity

        voc_idx = self.raw_features.index("voc_resistance")
        temp_idx = self.raw_features.index("temperature")
        hum_idx = self.raw_features.index("rel_humidity")

        abs_humidity = calculate_absolute_humidity(raw[:, temp_idx], raw[:, hum_idx])

        # Envelope baselines (per-sample rolling)
        voc_col = raw[:, voc_idx]
        if timestamps is not None:
            voc_series = pd.Series(voc_col, index=pd.DatetimeIndex(timestamps))
            use_time = True
        else:
            voc_series = pd.Series(voc_col)
            use_time = False

        baseline_24h, baseline_7d = self._compute_envelope(voc_series, use_time)
        baseline_24h_vals = baseline_24h.values
        baseline_7d_vals = baseline_7d.values

        gas_ratio_24h = np.clip(voc_col / baseline_24h_vals, 1e-6, None)
        log_ratio_24h = np.log(gas_ratio_24h)
        gas_ratio_7d = np.clip(voc_col / baseline_7d_vals, 1e-6, None)
        log_ratio_7d = np.log(gas_ratio_7d)

        # Temporal cyclical features
        if timestamps is not None:
            ts_index = pd.DatetimeIndex(timestamps)
            hours = ts_index.hour.values.astype(float)
            dows = ts_index.dayofweek.values.astype(float)
        else:
            hours = np.zeros(len(raw))
            dows = np.zeros(len(raw))

        hour_sin, hour_cos = self._cyclical_encode(hours, 24.0)
        dow_sin, dow_cos = self._cyclical_encode(dows, 7.0)

        return np.column_stack([
            raw,
            abs_humidity.reshape(-1, 1),
            baseline_24h_vals.reshape(-1, 1),
            gas_ratio_24h.reshape(-1, 1),
            log_ratio_24h.reshape(-1, 1),
            baseline_7d_vals.reshape(-1, 1),
            gas_ratio_7d.reshape(-1, 1),
            log_ratio_7d.reshape(-1, 1),
            hour_sin.reshape(-1, 1),
            hour_cos.reshape(-1, 1),
            dow_sin.reshape(-1, 1),
            dow_cos.reshape(-1, 1),
        ])

    def engineer_features_single(
        self,
        reading: Dict[str, float],
        baselines: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        from app.quantities import calculate_absolute_humidity

        raw_vals = [reading[f] for f in self.raw_features]

        voc_r = reading["voc_resistance"]

        # Fallback chain: baseline_24h → voc_resistance → raw voc_r
        bl_24h = (
            baselines.get("baseline_24h", baselines.get("voc_resistance", voc_r))
            if baselines else voc_r
        )
        bl_7d = (
            baselines.get("baseline_7d", baselines.get("voc_resistance", voc_r))
            if baselines else voc_r
        )

        gas_ratio_24h = max(voc_r / bl_24h, 1e-6)
        log_ratio_24h = np.log(gas_ratio_24h)
        gas_ratio_7d = max(voc_r / bl_7d, 1e-6)
        log_ratio_7d = np.log(gas_ratio_7d)

        abs_hum = calculate_absolute_humidity(
            np.array([reading["temperature"]]),
            np.array([reading["rel_humidity"]]),
        )[0]

        hour = float(timestamp.hour) if timestamp is not None else 0.0
        dow = float(timestamp.weekday()) if timestamp is not None else 0.0

        hour_sin, hour_cos = self._cyclical_encode(np.array([hour]), 24.0)
        dow_sin, dow_cos = self._cyclical_encode(np.array([dow]), 7.0)

        return np.array(raw_vals + [
            abs_hum,
            bl_24h, gas_ratio_24h, log_ratio_24h,
            bl_7d, gas_ratio_7d, log_ratio_7d,
            hour_sin[0], hour_cos[0],
            dow_sin[0], dow_cos[0],
        ])


class BME680NoPressureProfile(BME680Profile):
    """BME680 with pressure removed — used by ablation experiments."""

    @property
    def name(self) -> str:
        return "bme680_no_pressure"

    @property
    def raw_features(self) -> List[str]:
        return ["temperature", "rel_humidity", "voc_resistance"]

    @property
    def feature_quantities(self) -> Dict[str, str]:
        return {
            "temperature": "temperature",
            "rel_humidity": "relative_humidity",
            "voc_resistance": "voc_resistance",
        }

    @property
    def valid_ranges(self) -> Dict[str, Tuple[float, float]]:
        ranges = super().valid_ranges
        ranges.pop("pressure", None)
        return ranges


class SPS30Profile(SensorProfile):
    """Sensirion SPS30 particulate matter sensor."""

    @property
    def name(self) -> str:
        return "sps30"

    @property
    def raw_features(self) -> List[str]:
        return ["pm1_0", "pm2_5", "pm4_0", "pm10"]

    @property
    def feature_quantities(self) -> Dict[str, str]:
        return {
            "pm1_0": "pm1_0",
            "pm2_5": "pm2_5",
            "pm4_0": "pm4_0",
            "pm10": "pm10",
        }

    @property
    def engineered_feature_names(self) -> List[str]:
        return ["pm25_pm10_ratio", "pm1_pm25_ratio"]

    def engineer_features(
        self,
        raw: np.ndarray,
        baselines: Optional[Dict[str, float]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        pm1_idx = self.raw_features.index("pm1_0")
        pm25_idx = self.raw_features.index("pm2_5")
        pm10_idx = self.raw_features.index("pm10")

        pm25_pm10_ratio = raw[:, pm25_idx] / np.maximum(raw[:, pm10_idx], 0.1)
        pm1_pm25_ratio = raw[:, pm1_idx] / np.maximum(raw[:, pm25_idx], 0.1)

        return np.column_stack(
            [raw, pm25_pm10_ratio.reshape(-1, 1), pm1_pm25_ratio.reshape(-1, 1)]
        )

    def engineer_features_single(
        self,
        reading: Dict[str, float],
        baselines: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        raw_vals = [reading[f] for f in self.raw_features]

        pm25_pm10_ratio = reading["pm2_5"] / max(reading["pm10"], 0.1)
        pm1_pm25_ratio = reading["pm1_0"] / max(reading["pm2_5"], 0.1)

        return np.array(raw_vals + [pm25_pm10_ratio, pm1_pm25_ratio])


# ---------------------------------------------------------------------------
# Register built-in profiles and YAML-driven standards
# ---------------------------------------------------------------------------
register_sensor("bme680", BME680Profile)
register_sensor("sps30", SPS30Profile)
register_yaml_standards()
