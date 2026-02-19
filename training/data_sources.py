"""Pluggable data sources for the training pipeline."""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from app.config import settings

logger = logging.getLogger("training.data_sources")


class DataSource(ABC):
    """Abstract base class for training data sources."""

    @abstractmethod
    def validate(self) -> None:
        """Validate that the data source is reachable (SOURCE_ACCESS stage)."""
        ...

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Fetch raw data (INGESTION stage).

        Must return a DataFrame whose columns match the active SensorProfile's
        raw_features plus the IAQStandard's target_column (and optionally the
        profile's quality_column).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label for logging."""
        ...

    @property
    def metadata(self) -> dict:
        """Source-specific metadata for data provenance tracking."""
        return {}


class InfluxDBSource(DataSource):
    """Fetches training data from InfluxDB."""

    def __init__(self, measurement="bme688", hours_back=168 * 8, min_iaq_accuracy=2):
        self.measurement = measurement
        self.hours_back = hours_back
        self.min_iaq_accuracy = min_iaq_accuracy
        self._client = None

    @property
    def name(self) -> str:
        return f"InfluxDB({settings.INFLUX_HOST}:{settings.INFLUX_PORT}/{settings.INFLUX_DATABASE})"

    def validate(self) -> None:
        """Connect to InfluxDB and verify it's reachable."""
        from influxdb import DataFrameClient

        logger.info(
            "Connecting to InfluxDB at %s:%s, database=%s",
            settings.INFLUX_HOST,
            settings.INFLUX_PORT,
            settings.INFLUX_DATABASE,
        )

        try:
            self._client = DataFrameClient(
                host=settings.INFLUX_HOST,
                port=settings.INFLUX_PORT,
                username=settings.INFLUX_USERNAME,
                password=settings.INFLUX_PASSWORD,
                database=settings.INFLUX_DATABASE,
            )
            self._client.ping()
        except Exception as e:
            logger.error("Failed to connect to InfluxDB: %s", e)
            raise

    def fetch(self) -> pd.DataFrame:
        """Fetch sensor data from InfluxDB, filter by quality column."""
        if self._client is None:
            raise RuntimeError("validate() must be called before fetch()")

        from app.profiles import get_iaq_standard, get_sensor_profile

        profile = get_sensor_profile()
        standard = get_iaq_standard()

        columns = list(profile.raw_features) + [standard.target_column]
        if profile.quality_column:
            columns.append(profile.quality_column)
        select_clause = ", ".join(columns)

        query = f"""
        SELECT {select_clause}
        FROM {self.measurement}
        WHERE time > now() - {self.hours_back}h
        """

        result = self._client.query(query)

        if self.measurement not in result:
            raise ValueError(f"No data found in measurement '{self.measurement}'")

        df = result[self.measurement]
        raw_count = len(df)

        if profile.quality_column and profile.quality_min is not None:
            df = df[df[profile.quality_column] >= profile.quality_min]
        df = df.dropna()

        logger.info(
            "Fetched %d raw points, %d after filtering, "
            "date range: %s to %s",
            raw_count,
            len(df),
            df.index.min(),
            df.index.max(),
        )

        return df

    @property
    def metadata(self) -> dict:
        return {
            "source_type": "influxdb",
            "measurement": self.measurement,
            "hours_back": self.hours_back,
            "min_iaq_accuracy": self.min_iaq_accuracy,
            "host": settings.INFLUX_HOST,
            "database": settings.INFLUX_DATABASE,
        }

    def close(self) -> None:
        """Close the InfluxDB client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None


class SyntheticSource(DataSource):
    """Generates synthetic sensor data for development and testing."""

    def __init__(self, num_samples=1000, seed=42):
        self.num_samples = num_samples
        self.seed = seed

    @property
    def name(self) -> str:
        return f"SyntheticSource({self.num_samples} samples)"

    @property
    def metadata(self) -> dict:
        return {
            "source_type": "synthetic",
            "num_samples": self.num_samples,
            "seed": self.seed,
        }

    def validate(self) -> None:
        """No-op — synthetic data is always available."""
        logger.info("Validating %s", self.name)

    def fetch(self) -> pd.DataFrame:
        """Generate synthetic sensor data matching the active sensor profile."""
        from app.profiles import get_iaq_standard, get_sensor_profile

        profile = get_sensor_profile()
        standard = get_iaq_standard()
        rng = np.random.default_rng(self.seed)

        data = {}
        for feat in profile.raw_features:
            lo, hi = profile.valid_ranges.get(feat, (0, 1))
            # Use a comfortable sub-range to avoid edge effects
            margin = (hi - lo) * 0.1
            data[feat] = rng.uniform(lo + margin, hi - margin, self.num_samples)

        # Target: uniform across standard's scale range with noise
        scale_lo, scale_hi = standard.scale_range
        data[standard.target_column] = np.clip(
            rng.uniform(scale_lo, scale_hi, self.num_samples)
            + rng.normal(0, (scale_hi - scale_lo) * 0.02, self.num_samples),
            scale_lo,
            scale_hi,
        )

        # Quality column if the sensor profile defines one
        if profile.quality_column and profile.quality_min is not None:
            data[profile.quality_column] = np.full(
                self.num_samples, profile.quality_min
            )

        df = pd.DataFrame(data)
        logger.info("Generated %d synthetic samples (seed=%d)", self.num_samples, self.seed)

        return df
