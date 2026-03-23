"""Pluggable data sources for the training pipeline."""

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.config import settings
from app.exceptions import ConfigurationError, SchemaMismatchError

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
    def provided_units(self) -> Dict[str, str]:
        """Map feature names to units this source provides (if non-canonical).

        Empty dict = assume canonical units (default).  When non-empty, the
        pipeline merges these over the sensor profile's ``feature_units`` so
        source-level declarations take precedence during unit conversion.
        """
        return {}

    @property
    def metadata(self) -> dict:
        """Source-specific metadata for data provenance tracking."""
        return {}


class InfluxDBSource(DataSource):
    """Fetches training data from InfluxDB."""

    def __init__(
        self,
        measurement="bme688",
        hours_back=168 * 8,
        min_iaq_accuracy=2,
        database=None,
        max_records=None,
        cache: bool = False,
        cache_dir: str = "cache",
        unit_overrides: Optional[Dict[str, str]] = None,
    ):
        self.measurement = measurement
        self.hours_back = hours_back
        self.min_iaq_accuracy = min_iaq_accuracy
        self._database = database
        self._max_records = max_records
        self._client = None
        self._cache_enabled = cache
        self._cache_dir = Path(cache_dir)
        self._unit_overrides = unit_overrides or {}

    def _get_cache_key(self) -> str:
        """Generate a unique cache key based on source parameters."""
        params = {
            "measurement": self.measurement,
            "hours_back": self.hours_back,
            "min_iaq_accuracy": self.min_iaq_accuracy,
            "database": self._database,
            "max_records": self._max_records,
        }
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]

    def _cache_path(self) -> Path:
        """Get the cache file path for this source."""
        key = self._get_cache_key()
        return self._cache_dir / f"influxdb_{key}.parquet"

    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache if available."""
        if not self._cache_enabled:
            return None
        cache_file = self._cache_path()
        if cache_file.exists():
            logger.info("Loading data from cache: %s", cache_file)
            return pd.read_parquet(cache_file)
        return None

    def _save_to_cache(self, df: pd.DataFrame) -> None:
        """Save DataFrame to cache."""
        if not self._cache_enabled:
            return
        cache_file = self._cache_path()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving data to cache: %s", cache_file)
        df.to_parquet(cache_file)

    @property
    def name(self) -> str:
        db = self._database or settings.INFLUX_DATABASE
        return f"InfluxDB({settings.INFLUX_HOST}:{settings.INFLUX_PORT}/{db})"

    @property
    def provided_units(self) -> Dict[str, str]:
        return self._unit_overrides

    def validate(self) -> None:
        """Connect to InfluxDB and verify it's reachable."""
        from influxdb import DataFrameClient

        db = self._database or settings.INFLUX_DATABASE
        logger.info(
            "Connecting to InfluxDB at %s:%s, database=%s",
            settings.INFLUX_HOST,
            settings.INFLUX_PORT,
            db,
        )

        try:
            self._client = DataFrameClient(
                host=settings.INFLUX_HOST,
                port=settings.INFLUX_PORT,
                username=settings.INFLUX_USERNAME,
                password=settings.INFLUX_PASSWORD,
                database=db,
            )
            self._client.ping()
        except Exception as e:
            from app.exceptions import InfluxUnreachableError

            raise InfluxUnreachableError(
                f"Failed to connect to InfluxDB at {settings.INFLUX_HOST}:{settings.INFLUX_PORT}: {e}",
                suggestion="Check host/port/network",
            ) from e

    def fetch(self) -> pd.DataFrame:
        """Fetch sensor data from InfluxDB, filter by quality column."""
        cached = self._load_from_cache()
        if cached is not None:
            logger.info("Using cached data: %d rows", len(cached))
            return cached

        if self._client is None:
            raise ConfigurationError(
                "validate() must be called before fetch()",
                suggestion="Call source.validate() before source.fetch()",
            )

        from app.profiles import get_iaq_standard, get_sensor_profile

        profile = get_sensor_profile()
        standard = get_iaq_standard()

        cfg = settings.load_model_config()
        field_mapping = cfg.get("sensor", {}).get("field_mapping", {})
        reverse_mapping = {v: k for k, v in field_mapping.items()}

        columns = list(profile.raw_features) + [standard.target_column]
        if profile.quality_column:
            columns.append(profile.quality_column)

        external_columns = [reverse_mapping.get(c, c) for c in columns]
        select_clause = ", ".join(external_columns)

        hours = self.hours_back if self.hours_back else 168 * 8  # default ~56 days
        query = f"""
        SELECT {select_clause}
        FROM {self.measurement}
        WHERE time > now() - {hours}h
        """
        if self._max_records:
            query += f" LIMIT {self._max_records}"

        result = self._client.query(query)

        if self.measurement not in result:
            from app.exceptions import NoDataError

            raise NoDataError(
                f"No data found in measurement '{self.measurement}'",
                suggestion="Check measurement name or widen time range",
            )

        df = result[self.measurement]

        if field_mapping:
            df = df.rename(columns=field_mapping)
            logger.info("Applied field mapping: %s", field_mapping)

        raw_count = len(df)

        if profile.quality_column and profile.quality_min is not None:
            df = df[df[profile.quality_column] >= profile.quality_min]
        df = df.dropna()

        logger.info(
            "Fetched %d raw points, %d after filtering, date range: %s to %s",
            raw_count,
            len(df),
            df.index.min(),
            df.index.max(),
        )

        self._save_to_cache(df)
        return df

    @property
    def metadata(self) -> dict:
        identity = settings.get_sensor_identity()
        db = self._database or settings.INFLUX_DATABASE
        meta = {
            "source_type": "influxdb",
            "measurement": self.measurement,
            "hours_back": self.hours_back,
            "min_iaq_accuracy": self.min_iaq_accuracy,
            "host": settings.INFLUX_HOST,
            "database": db,
        }
        if self._max_records:
            meta["max_records"] = self._max_records
        if identity.get("sensor_id"):
            meta["sensor_id"] = identity["sensor_id"]
        if identity.get("firmware_version"):
            meta["firmware_version"] = identity["firmware_version"]
        return meta

    def close(self) -> None:
        """Close the InfluxDB client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None


class CSVDataSource(DataSource):
    """Loads training data from a CSV file."""

    def __init__(
        self,
        csv_path: str,
        field_mapping: Optional[dict] = None,
        unit_overrides: Optional[Dict[str, str]] = None,
    ):
        self.csv_path = csv_path
        self._field_mapping = field_mapping
        self._unit_overrides = unit_overrides or {}

    @property
    def name(self) -> str:
        return f"CSV({self.csv_path})"

    @property
    def metadata(self) -> dict:
        identity = settings.get_sensor_identity()
        meta = {
            "source_type": "csv",
            "csv_path": self.csv_path,
            "field_mapping": self._field_mapping or {},
        }
        if identity.get("sensor_id"):
            meta["sensor_id"] = identity["sensor_id"]
        if identity.get("firmware_version"):
            meta["firmware_version"] = identity["firmware_version"]
        return meta

    @property
    def provided_units(self) -> Dict[str, str]:
        return self._unit_overrides

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
            from app.exceptions import ConfigurationError
            raise ConfigurationError(f"CSV file not found: {self.csv_path}")
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
                raw_na = int(df[col].isna().sum())
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Epoch seconds (values > 1e9 ≈ post-2001)
                    if df[col].dropna().min() > 1e9:
                        df[col] = pd.to_datetime(df[col], unit="s", errors="coerce", utc=True)
                    else:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                else:
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
                coerced = int(df[col].isna().sum()) - raw_na
                if coerced > 0:
                    logger.warning(
                        "%d timestamp(s) could not be parsed in column '%s'",
                        coerced, col,
                    )
                df = df.set_index(col)
                logger.info("Set timestamp index: %s", col)
                break

        # Validate required columns
        required = list(profile.raw_features) + [standard.target_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SchemaMismatchError(
                f"CSV missing required columns: {missing}. "
                f"Available: {list(df.columns)}.",
                suggestion=f"Run 'python -m iaq4j map-fields --source {self.csv_path} --save' to create a field mapping.",
            )

        # Filter by quality column if available
        if profile.quality_column and profile.quality_column in df.columns:
            if profile.quality_min is not None:
                df = df[df[profile.quality_column] >= profile.quality_min]

        df = df.dropna(subset=required)
        logger.info(
            "CSV: %d raw rows → %d after filtering (columns: %s)",
            raw_count,
            len(df),
            list(df.columns),
        )

        return df


# LabelStudioDataSource moved to integrations.label_studio.data_source
# Backward-compat import:
from integrations.label_studio.data_source import LabelStudioDataSource  # noqa: F401


class ParquetSource(DataSource):
    """Data source from a local Parquet file."""

    def __init__(self, path, end_date=None):
        self._path = Path(path)
        self._end_date = end_date

    @property
    def name(self) -> str:
        return f"Parquet({self._path.name})"

    @property
    def provided_units(self) -> Dict[str, str]:
        return {}

    def validate(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Not found: {self._path}")

    def fetch(self) -> pd.DataFrame:
        df = pd.read_parquet(self._path)
        if self._end_date is not None:
            df = df[df.index < self._end_date]
        return df

    def close(self):
        pass


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

    # BME680 features that have physics-based generation
    _PHYSICS_FEATURES = {"temperature", "rel_humidity", "pressure", "voc_resistance"}

    def fetch(self) -> pd.DataFrame:
        """Generate synthetic sensor data with physics-based correlations.

        Uses a three-layer generative model:
        1. Temporal backbone — dual-peak occupancy signal drives VOC emissions
        2. Correlated environmental features with AR(1) autocorrelation
        3. IAQ target from physics-inspired transfer function

        Features not in the physics model fall back to uniform random.
        """
        from app.profiles import get_iaq_standard, get_sensor_profile

        profile = get_sensor_profile()
        standard = get_iaq_standard()
        rng = np.random.default_rng(self.seed)
        n = self.num_samples

        # ── DatetimeIndex: evenly-spaced across one week ──────────────────
        week_start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        week_seconds = 7 * 24 * 3600
        interval_seconds = max(1, week_seconds // n)
        index = pd.date_range(
            start=week_start, periods=n, freq=f"{interval_seconds}s", tz="UTC"
        )

        hours = np.array([ts.hour + ts.minute / 60 for ts in index])
        dow = np.array([ts.dayofweek for ts in index])  # 0=Mon, 6=Sun

        # ── Layer 1: Occupancy signal ─────────────────────────────────────
        # Dual Gaussian peaks: morning ~9am, evening ~7pm, trough at ~3am
        occ_morning = np.exp(-0.5 * ((hours - 9) / 2.5) ** 2)
        occ_evening = np.exp(-0.5 * ((hours - 19) / 2.0) ** 2)
        occupancy = 0.6 * occ_morning + occ_evening
        # Weekend dampening
        weekend_mask = (dow >= 5).astype(float)
        occupancy *= 1.0 - 0.4 * weekend_mask
        # Normalize to [0, 1]
        occupancy = occupancy / (occupancy.max() + 1e-8)

        # ── Helper: AR(1) smooth drift ────────────────────────────────────
        def ar1_drift(decay: float, scale: float) -> np.ndarray:
            """Generate AR(1) process: x[t] = decay * x[t-1] + noise."""
            noise = rng.normal(0, scale, n)
            drift = np.empty(n)
            drift[0] = noise[0]
            for i in range(1, n):
                drift[i] = decay * drift[i - 1] + noise[i]
            return drift

        data = {}

        # ── Layer 2: Correlated environmental features ────────────────────
        has_physics = self._PHYSICS_FEATURES.issubset(set(profile.raw_features))

        if has_physics:
            # Temperature: 22°C base + diurnal swing (peak ~2pm) + slow drift
            diurnal_temp = 2.0 * np.sin(2 * np.pi * (hours - 8) / 24)
            temp_drift = ar1_drift(decay=0.995, scale=0.05)
            temperature = 22.0 + diurnal_temp + temp_drift
            temperature = np.clip(temperature, 18.0, 30.0)
            data["temperature"] = temperature

            # Humidity: 50% base, anti-correlated with temperature + drift
            humidity = 50.0 - 0.8 * (temperature - 22.0) + ar1_drift(0.995, 0.08)
            humidity = np.clip(humidity, 30.0, 70.0)
            data["rel_humidity"] = humidity

            # Pressure: 1013 hPa base + very slow weather drift
            pressure = 1013.0 + ar1_drift(decay=0.999, scale=0.02)
            pressure = np.clip(pressure, 995.0, 1030.0)
            data["pressure"] = pressure

            # VOC resistance: log-space, clean air ~200k, degraded by occupancy
            log_voc_clean = np.log(200_000)
            # Occupancy degrades VOC (lower resistance), temperature slightly too
            log_voc = (
                log_voc_clean
                - 2.5 * occupancy  # occupancy drives pollution
                - 0.05 * (temperature - 22.0)  # warm air slightly degrades
                + ar1_drift(0.99, 0.03)  # sensor noise
            )
            voc_resistance = np.exp(log_voc)
            voc_resistance = np.clip(voc_resistance, 5_000, 1_500_000)
            data["voc_resistance"] = voc_resistance
        else:
            voc_resistance = None

        # Fallback: any feature not in the physics model gets uniform random
        for feat in profile.raw_features:
            if feat not in data:
                lo, hi = profile.valid_ranges.get(feat, (0, 1))
                margin = (hi - lo) * 0.1
                data[feat] = rng.uniform(lo + margin, hi - margin, n)

        # ── Layer 3: IAQ from physics-inspired transfer function ──────────
        scale_lo, scale_hi = standard.scale_range
        if has_physics:
            # IAQ = f(voc_resistance, humidity)
            iaq = 25.0 + 120.0 * (np.log(200_000) - np.log(voc_resistance))
            # Humidity penalty: discomfort at extremes
            iaq += np.where(humidity > 60, 0.5 * (humidity - 60), 0.0)
            iaq += np.where(humidity < 35, 0.3 * (35 - humidity), 0.0)
            # Measurement noise
            iaq += rng.normal(0, 8.0, n)
            iaq = np.clip(iaq, scale_lo, scale_hi)
        else:
            # No physics model — uniform target (same as old behavior)
            iaq = np.clip(
                rng.uniform(scale_lo, scale_hi, n)
                + rng.normal(0, (scale_hi - scale_lo) * 0.02, n),
                scale_lo,
                scale_hi,
            )
        data[standard.target_column] = iaq

        # Quality column if the sensor profile defines one
        if profile.quality_column and profile.quality_min is not None:
            data[profile.quality_column] = np.full(n, profile.quality_min)

        df = pd.DataFrame(data, index=index)
        logger.info(
            "Generated %d synthetic samples (seed=%d, physics=%s) with "
            "DatetimeIndex spanning %s to %s",
            n,
            self.seed,
            has_physics,
            index[0],
            index[-1],
        )

        return df
