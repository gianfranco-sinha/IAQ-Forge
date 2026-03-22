# ============================================================================
# File: app/profiles.py
# Core abstractions for sensor-agnostic, IAQ-standard-agnostic architecture.
# ============================================================================
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


class SensorProfile(ABC):
    """Defines what a sensor provides and how to engineer features from it."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable sensor name, e.g. 'bme680'."""
        ...

    @property
    @abstractmethod
    def raw_features(self) -> List[str]:
        """Ordered list of raw column names the sensor provides."""
        ...

    @property
    def feature_quantities(self) -> Dict[str, str]:
        """Map feature names to quantity names from quantities.yaml.

        Override in subclass to enable registry-driven valid_ranges and
        field_descriptions.  Keys are feature names (matching raw_features),
        values are quantity names from quantities.yaml.
        """
        return {}

    @property
    def feature_units(self) -> Dict[str, str]:
        """Map feature names to the units the sensor reports.

        Only needed when the sensor reports in a non-canonical unit.  If a
        feature is absent from this dict, the canonical unit is assumed.
        """
        return {}

    @property
    def valid_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Valid (min, max) ranges for each raw feature and quality columns.

        By default, computed from feature_quantities + the quantity registry.
        Override in subclass to extend or replace.
        """
        from app.quantities import get_quantity

        ranges: Dict[str, Tuple[float, float]] = {}
        for feat, qty_name in self.feature_quantities.items():
            q = get_quantity(qty_name)
            if q.valid_range:
                ranges[feat] = q.valid_range
        return ranges

    @property
    def field_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Per-feature metadata: unit, physical meaning.

        By default, computed from feature_quantities + the quantity registry.
        Override in subclass to extend or replace.
        """
        from app.quantities import get_quantity

        descs: Dict[str, Dict[str, str]] = {}
        for feat, qty_name in self.feature_quantities.items():
            q = get_quantity(qty_name)
            descs[feat] = {
                "unit": q.canonical_unit,
                "description": q.description,
            }
        return descs

    @property
    def quality_column(self) -> Optional[str]:
        """Column name for data-quality filtering (e.g. 'iaq_accuracy').
        Return None if the sensor has no quality indicator."""
        return None

    @property
    def quality_min(self) -> Optional[float]:
        """Minimum acceptable value for quality_column."""
        return None

    @property
    def expected_interval_seconds(self) -> Optional[float]:
        """Expected sampling interval in seconds. None = skip check."""
        return None

    @property
    def warmup_seconds(self) -> float:
        """Duration in seconds to discard after each power-on / segment start.

        MOX sensors (e.g. BME680) need time for the heater plate to stabilise.
        During warm-up, resistance readings are unreliable and should be
        excluded from training.  Override in subclass.  0 = no filtering.
        """
        return 0.0

    @property
    def heater_temperature_c(self) -> Optional[float]:
        """Heater plate temperature in °C used by the sensor firmware.

        Relevant for MOX sensors where gas resistance depends on heater
        temperature.  None = sensor has no heater / not applicable.
        """
        return None

    @property
    @abstractmethod
    def engineered_feature_names(self) -> List[str]:
        """Names of derived features appended after raw columns."""
        ...

    @property
    def total_features(self) -> int:
        """Total feature count = raw + engineered."""
        return len(self.raw_features) + len(self.engineered_feature_names)

    @property
    def all_feature_names(self) -> List[str]:
        """All feature names in order."""
        return self.raw_features + self.engineered_feature_names

    @staticmethod
    def _cyclical_encode(
        values: np.ndarray, period: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode values as (sin, cos) pair for cyclical continuity.

        e.g. hour-of-day with period=24, day-of-week with period=7.
        """
        angle = 2 * np.pi * values / period
        return np.sin(angle), np.cos(angle)

    def compute_baselines(
        self, raw: np.ndarray, timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute baseline values from training data (e.g. median gas resistance).
        Override in subclass if needed. Returns empty dict by default."""
        return {}

    @abstractmethod
    def engineer_features(
        self,
        raw: np.ndarray,
        baselines: Optional[Dict[str, float]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch feature engineering for training.

        Args:
            raw: shape (n_samples, len(raw_features)), columns in raw_features order.
            baselines: dict from compute_baselines().
            timestamps: DatetimeIndex values array (dtype datetime64) for temporal
                feature engineering. None when source has no timestamp index.

        Returns:
            shape (n_samples, total_features) — raw columns plus engineered columns.
        """
        ...

    @abstractmethod
    def engineer_features_single(
        self,
        reading: Dict[str, float],
        baselines: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        """Single-reading feature engineering for real-time inference.

        Args:
            reading: dict mapping raw feature names to values.
            baselines: dict from compute_baselines().
            timestamp: reading datetime for temporal feature engineering.
                None falls back to zero-valued cyclical features.

        Returns:
            1D array of length total_features.
        """
        ...


class IAQStandard(ABC):
    """Defines the target IAQ scale and how to categorize predictions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable standard name, e.g. 'bsec'."""
        ...

    @property
    @abstractmethod
    def target_column(self) -> str:
        """Column name in training data for the target value (e.g. 'iaq')."""
        ...

    @property
    @abstractmethod
    def scale_range(self) -> Tuple[float, float]:
        """(min, max) of the output scale for clamping predictions."""
        ...

    @property
    @abstractmethod
    def categories(self) -> List[Tuple[float, str]]:
        """Ordered (upper_bound, category_name) pairs.
        Last entry should use float('inf') or scale_range max."""
        ...

    def clamp(self, value: float) -> float:
        """Clamp a raw prediction to the valid scale range."""
        lo, hi = self.scale_range
        return max(lo, min(hi, value))

    def categorize(self, value: float) -> str:
        """Map a numeric value to a category name."""
        for threshold, name in self.categories:
            if value <= threshold:
                return name
        return self.categories[-1][1]

    def category_distribution(self, values) -> Dict[str, int]:
        """Count how many values fall in each category."""
        dist = {name: 0 for _, name in self.categories}
        for v in values:
            dist[self.categorize(v)] += 1
        return dist


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_SENSOR_REGISTRY: Dict[str, type] = {}
_STANDARD_REGISTRY: Dict[str, type] = {}


def register_sensor(name: str, cls: type) -> None:
    _SENSOR_REGISTRY[name] = cls


def register_standard(name: str, cls: type) -> None:
    _STANDARD_REGISTRY[name] = cls


def get_sensor_profile(sensor_type: Optional[str] = None) -> SensorProfile:
    """Return a SensorProfile by name.

    Args:
        sensor_type: Explicit sensor name (e.g. 'bme680').  When *None*,
            reads ``sensor.type`` from ``model_config.yaml`` (default 'bme680').
    """
    if sensor_type is None:
        from app.config import settings
        cfg = settings.load_model_config()
        sensor_type = cfg.get("sensor", {}).get("type", "bme680")

    cls = _SENSOR_REGISTRY.get(sensor_type)
    if cls is None:
        from app.exceptions import ConfigurationError
        raise ConfigurationError(
            f"Unknown sensor profile: '{sensor_type}'. "
            f"Registered: {list(_SENSOR_REGISTRY)}"
        )
    return cls()


def get_iaq_standard(standard_type: Optional[str] = None) -> IAQStandard:
    """Return an IAQStandard by name.

    Args:
        standard_type: Explicit standard name (e.g. 'bsec').  When *None*,
            reads ``iaq_standard.type`` from ``model_config.yaml`` (default 'bsec').
    """
    if standard_type is None:
        from app.config import settings
        cfg = settings.load_model_config()
        standard_type = cfg.get("iaq_standard", {}).get("type", "bsec")

    cls = _STANDARD_REGISTRY.get(standard_type)
    if cls is None:
        from app.exceptions import ConfigurationError
        raise ConfigurationError(
            f"Unknown IAQ standard: '{standard_type}'. "
            f"Registered: {list(_STANDARD_REGISTRY)}"
        )
    return cls()
