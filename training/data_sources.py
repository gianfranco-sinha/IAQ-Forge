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

        Must return a DataFrame with columns:
            temperature, rel_humidity, pressure, gas_resistance, iaq, iaq_accuracy
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label for logging."""
        ...


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
        """Fetch sensor data from InfluxDB, filter by iaq_accuracy."""
        if self._client is None:
            raise RuntimeError("validate() must be called before fetch()")

        query = f"""
        SELECT temperature, rel_humidity, pressure, gas_resistance, iaq, iaq_accuracy
        FROM {self.measurement}
        WHERE time > now() - {self.hours_back}h
        """

        result = self._client.query(query)

        if self.measurement not in result:
            raise ValueError(f"No data found in measurement '{self.measurement}'")

        df = result[self.measurement]
        raw_count = len(df)

        df = df[df["iaq_accuracy"] >= self.min_iaq_accuracy]
        df = df.dropna()

        logger.info(
            "Fetched %d raw points, %d after filtering (iaq_accuracy >= %d), "
            "date range: %s to %s",
            raw_count,
            len(df),
            self.min_iaq_accuracy,
            df.index.min(),
            df.index.max(),
        )

        return df

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

    def validate(self) -> None:
        """No-op — synthetic data is always available."""
        logger.info("Validating %s", self.name)

    def fetch(self) -> pd.DataFrame:
        """Generate synthetic sensor data with correlated IAQ values."""
        rng = np.random.default_rng(self.seed)

        temperature = rng.uniform(18, 30, self.num_samples)
        rel_humidity = rng.uniform(30, 70, self.num_samples)
        pressure = rng.uniform(980, 1020, self.num_samples)
        gas_resistance = rng.uniform(50_000, 500_000, self.num_samples)

        # IAQ derived from gas_resistance with noise:
        # lower gas_resistance → higher IAQ (worse air quality)
        iaq = 500 * (1 - (gas_resistance - 50_000) / 450_000) + rng.normal(0, 10, self.num_samples)
        iaq = np.clip(iaq, 0, 500)

        iaq_accuracy = np.full(self.num_samples, 3)

        df = pd.DataFrame({
            "temperature": temperature,
            "rel_humidity": rel_humidity,
            "pressure": pressure,
            "gas_resistance": gas_resistance,
            "iaq": iaq,
            "iaq_accuracy": iaq_accuracy,
        })

        logger.info("Generated %d synthetic samples (seed=%d)", self.num_samples, self.seed)

        return df
