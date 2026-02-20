"""Pluggable data sources for the training pipeline."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

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


class CSVDataSource(DataSource):
    """Loads training data from a CSV file."""

    def __init__(self, csv_path: str, field_mapping: Optional[dict] = None):
        self.csv_path = csv_path
        self._field_mapping = field_mapping

    @property
    def name(self) -> str:
        return f"CSV({self.csv_path})"

    @property
    def metadata(self) -> dict:
        return {
            "source_type": "csv",
            "csv_path": self.csv_path,
            "field_mapping": self._field_mapping or {},
        }

    @property
    def field_mapping(self) -> dict:
        if self._field_mapping is not None:
            return self._field_mapping
        cfg = settings.load_model_config()
        return cfg.get("sensor", {}).get("field_mapping", {})

    def validate(self) -> None:
        """Check that CSV file exists and is readable."""
        from pathlib import Path

        p = Path(self.csv_path)
        if not p.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        logger.info("Validated CSV source: %s", self.csv_path)

    def fetch(self) -> pd.DataFrame:
        """Read CSV, apply field mapping, validate required columns."""
        from app.profiles import get_iaq_standard, get_sensor_profile

        profile = get_sensor_profile()
        standard = get_iaq_standard()

        df = pd.read_csv(self.csv_path)
        raw_count = len(df)
        logger.info("Read %d rows from %s", raw_count, self.csv_path)

        # Apply field mapping (rename columns: external → internal)
        mapping = self.field_mapping
        if mapping:
            reverse = {ext: internal for ext, internal in mapping.items()}
            df = df.rename(columns=reverse)
            logger.info("Applied field mapping: %s", reverse)

        # Detect and set timestamp index
        ts_candidates = {"timestamp", "time", "datetime", "date", "ts"}
        for col in df.columns:
            if col.lower().strip() in ts_candidates:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df.set_index(col)
                logger.info("Set timestamp index: %s", col)
                break

        # Validate required columns
        required = list(profile.raw_features) + [standard.target_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Available: {list(df.columns)}. "
                f"Consider running 'python -m iaq4j map-fields --source {self.csv_path} --save' first."
            )

        # Filter by quality column if available
        if profile.quality_column and profile.quality_column in df.columns:
            if profile.quality_min is not None:
                df = df[df[profile.quality_column] >= profile.quality_min]

        df = df.dropna(subset=required)
        logger.info(
            "CSV: %d raw rows → %d after filtering (columns: %s)",
            raw_count, len(df), list(df.columns),
        )

        return df


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
