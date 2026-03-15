"""Tests for InfluxDBSource cache functionality."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.data_sources import InfluxDBSource


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    yield str(cache_dir)
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "temperature": np.random.randn(100) + 20,
            "rel_humidity": np.random.rand(100) * 50 + 30,
            "voc_resistance": np.random.rand(100) * 1000 + 100,
            "pressure": np.random.rand(100) * 20 + 1000,
            "iaq": np.random.rand(100) * 100,
        },
        index=dates,
    )
    return df


class TestInfluxDBSourceCache:
    """Tests for InfluxDBSource cache functionality."""

    def test_cache_key_changes_with_params(self, temp_cache_dir):
        """Different parameters should produce different cache keys."""
        source1 = InfluxDBSource(
            measurement="bme688",
            hours_back=168,
            database="test_db",
            cache=True,
            cache_dir=temp_cache_dir,
        )
        source2 = InfluxDBSource(
            measurement="bme688",
            hours_back=336,  # different hours_back
            database="test_db",
            cache=True,
            cache_dir=temp_cache_dir,
        )
        source3 = InfluxDBSource(
            measurement="bme688",
            hours_back=168,
            database="test_db",
            cache=True,
            cache_dir=temp_cache_dir,
        )

        key1 = source1._get_cache_key()
        key2 = source2._get_cache_key()
        key3 = source3._get_cache_key()

        assert key1 != key2, "Different hours_back should produce different keys"
        assert key1 == key3, "Same params should produce same key"

    def test_load_from_cache_returns_none_when_empty(self, temp_cache_dir):
        """Loading from empty cache should return None."""
        source = InfluxDBSource(
            measurement="bme688",
            hours_back=168,
            database="test_db",
            cache=True,
            cache_dir=temp_cache_dir,
        )
        result = source._load_from_cache()
        assert result is None

    def test_save_and_load_roundtrip(self, temp_cache_dir, sample_dataframe):
        """Saving and loading should preserve DataFrame exactly."""
        source = InfluxDBSource(
            measurement="bme688",
            hours_back=168,
            database="test_db",
            cache=True,
            cache_dir=temp_cache_dir,
        )

        source._save_to_cache(sample_dataframe)
        loaded = source._load_from_cache()

        assert loaded is not None
        assert len(loaded) == len(sample_dataframe)
        assert list(loaded.columns) == list(sample_dataframe.columns)
        np.testing.assert_array_almost_equal(
            loaded.values, sample_dataframe.values, decimal=5
        )

    def test_cache_disabled_does_not_save(self, temp_cache_dir, sample_dataframe):
        """When cache=False, _save_to_cache should not write files."""
        source = InfluxDBSource(
            measurement="bme688",
            hours_back=168,
            database="test_db",
            cache=False,  # disabled
            cache_dir=temp_cache_dir,
        )

        source._save_to_cache(sample_dataframe)

        cache_files = list(Path(temp_cache_dir).glob("*.parquet"))
        assert len(cache_files) == 0

    def test_cache_disabled_load_returns_none(self, temp_cache_dir, sample_dataframe):
        """When cache=False, _load_from_cache should return None."""
        # First save with cache enabled
        source1 = InfluxDBSource(
            measurement="bme688",
            hours_back=168,
            database="test_db",
            cache=True,
            cache_dir=temp_cache_dir,
        )
        source1._save_to_cache(sample_dataframe)

        # Then try to load with cache disabled
        source2 = InfluxDBSource(
            measurement="bme688",
            hours_back=168,
            database="test_db",
            cache=False,  # disabled
            cache_dir=temp_cache_dir,
        )
        result = source2._load_from_cache()
        assert result is None

    def test_different_database_different_key(self, temp_cache_dir):
        """Different database should produce different cache key."""
        source1 = InfluxDBSource(
            measurement="bme688",
            hours_back=168,
            database="db1",
            cache=True,
            cache_dir=temp_cache_dir,
        )
        source2 = InfluxDBSource(
            measurement="bme688",
            hours_back=168,
            database="db2",
            cache=True,
            cache_dir=temp_cache_dir,
        )

        key1 = source1._get_cache_key()
        key2 = source2._get_cache_key()

        assert key1 != key2, "Different databases should produce different keys"
