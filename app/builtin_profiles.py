# ============================================================================
# File: app/builtin_profiles.py
# Built-in sensor profiles and IAQ standards.
# ============================================================================
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.profiles import (
    IAQStandard,
    SensorProfile,
    register_sensor,
    register_standard,
)


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
    def engineered_feature_names(self) -> List[str]:
        return ["voc_ratio", "abs_humidity"]

    def compute_baselines(self, raw: np.ndarray) -> Dict[str, float]:
        voc_idx = self.raw_features.index("voc_resistance")
        return {"voc_resistance": float(np.median(raw[:, voc_idx]))}

    def engineer_features(
        self, raw: np.ndarray, baselines: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        from training.utils import calculate_absolute_humidity

        voc_idx = self.raw_features.index("voc_resistance")
        temp_idx = self.raw_features.index("temperature")
        hum_idx = self.raw_features.index("rel_humidity")

        baseline_voc = (
            baselines.get("voc_resistance", float(np.median(raw[:, voc_idx])))
            if baselines
            else float(np.median(raw[:, voc_idx]))
        )
        voc_ratio = raw[:, voc_idx] / baseline_voc
        abs_humidity = calculate_absolute_humidity(
            raw[:, temp_idx], raw[:, hum_idx]
        )

        return np.column_stack(
            [raw, voc_ratio.reshape(-1, 1), abs_humidity.reshape(-1, 1)]
        )

    def engineer_features_single(
        self, reading: Dict[str, float], baselines: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        from training.utils import calculate_absolute_humidity

        raw_vals = [reading[f] for f in self.raw_features]

        voc_r = reading["voc_resistance"]
        baseline_voc = (
            baselines.get("voc_resistance", voc_r) if baselines else voc_r
        )
        voc_ratio = voc_r / baseline_voc

        abs_hum = calculate_absolute_humidity(
            np.array([reading["temperature"]]),
            np.array([reading["rel_humidity"]]),
        )[0]

        return np.array(raw_vals + [voc_ratio, abs_hum])


class BSECStandard(IAQStandard):
    """Bosch BSEC IAQ index — 0-500 scale with 5 categories."""

    @property
    def name(self) -> str:
        return "bsec"

    @property
    def target_column(self) -> str:
        return "iaq"

    @property
    def scale_range(self) -> Tuple[float, float]:
        return (0.0, 500.0)

    @property
    def categories(self) -> List[Tuple[float, str]]:
        return [
            (50, "Excellent"),
            (100, "Good"),
            (200, "Moderate"),
            (300, "Poor"),
            (float("inf"), "Very Poor"),
        ]


# ---------------------------------------------------------------------------
# Register built-in profiles
# ---------------------------------------------------------------------------
register_sensor("bme680", BME680Profile)
register_standard("bsec", BSECStandard)
