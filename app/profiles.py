# ============================================================================
# File: app/profiles.py
# Core abstractions for sensor-agnostic, IAQ-standard-agnostic architecture.
# ============================================================================
from abc import ABC, abstractmethod
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
    @abstractmethod
    def valid_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Valid (min, max) ranges for each raw feature and quality columns."""
        ...

    @property
    def field_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Per-feature metadata: unit, physical meaning, and example value.

        Override in subclass. Keys must match ``raw_features``.
        Example::

            {"temperature": {"unit": "°C", "description": "Ambient temperature", "example": "25.0"}}
        """
        return {}

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

    def compute_baselines(self, raw: np.ndarray) -> Dict[str, float]:
        """Compute baseline values from training data (e.g. median gas resistance).
        Override in subclass if needed. Returns empty dict by default."""
        return {}

    @abstractmethod
    def engineer_features(
        self, raw: np.ndarray, baselines: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Batch feature engineering for training.

        Args:
            raw: shape (n_samples, len(raw_features)), columns in raw_features order.
            baselines: dict from compute_baselines().

        Returns:
            shape (n_samples, total_features) — raw columns plus engineered columns.
        """
        ...

    @abstractmethod
    def engineer_features_single(
        self, reading: Dict[str, float], baselines: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Single-reading feature engineering for real-time inference.

        Args:
            reading: dict mapping raw feature names to values.
            baselines: dict from compute_baselines().

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


def get_sensor_profile() -> SensorProfile:
    """Return the SensorProfile configured in model_config.yaml (sensor.type).
    Falls back to 'bme680' if not set."""
    from app.config import settings

    cfg = settings.load_model_config()
    sensor_name = cfg.get("sensor", {}).get("type", "bme680")
    cls = _SENSOR_REGISTRY.get(sensor_name)
    if cls is None:
        raise ValueError(
            f"Unknown sensor profile: '{sensor_name}'. "
            f"Registered: {list(_SENSOR_REGISTRY)}"
        )
    return cls()


def get_iaq_standard() -> IAQStandard:
    """Return the IAQStandard configured in model_config.yaml (iaq_standard.type).
    Falls back to 'bsec' if not set."""
    from app.config import settings

    cfg = settings.load_model_config()
    std_name = cfg.get("iaq_standard", {}).get("type", "bsec")
    cls = _STANDARD_REGISTRY.get(std_name)
    if cls is None:
        raise ValueError(
            f"Unknown IAQ standard: '{std_name}'. "
            f"Registered: {list(_STANDARD_REGISTRY)}"
        )
    return cls()
