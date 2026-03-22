"""Tests for app/profiles.py — ABC concrete methods, cyclical encoding, registry."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.profiles import (
    SensorProfile,
    IAQStandard,
    register_sensor,
    register_standard,
    _SENSOR_REGISTRY,
    _STANDARD_REGISTRY,
)


# ── _cyclical_encode ───────────────────────────────────────────────────────


class TestCyclicalEncode:
    def test_period_24_wraparound(self):
        """Hour 0 and hour 24 produce identical values."""
        s0, c0 = SensorProfile._cyclical_encode(np.array([0.0]), 24.0)
        s24, c24 = SensorProfile._cyclical_encode(np.array([24.0]), 24.0)
        np.testing.assert_allclose(s0, s24, atol=1e-10)
        np.testing.assert_allclose(c0, c24, atol=1e-10)

    def test_period_24_hour_6(self):
        """Hour 6 → sin≈1, cos≈0."""
        s, c = SensorProfile._cyclical_encode(np.array([6.0]), 24.0)
        assert abs(s[0] - 1.0) < 1e-10
        assert abs(c[0] - 0.0) < 1e-10

    def test_period_24_hour_12(self):
        """Hour 12 → sin≈0, cos≈-1."""
        s, c = SensorProfile._cyclical_encode(np.array([12.0]), 24.0)
        assert abs(s[0] - 0.0) < 1e-10
        assert abs(c[0] - (-1.0)) < 1e-10

    def test_period_7_wraparound(self):
        """Day 0 and day 7 produce identical values."""
        s0, c0 = SensorProfile._cyclical_encode(np.array([0.0]), 7.0)
        s7, c7 = SensorProfile._cyclical_encode(np.array([7.0]), 7.0)
        np.testing.assert_allclose(s0, s7, atol=1e-10)
        np.testing.assert_allclose(c0, c7, atol=1e-10)

    def test_sin_cos_norm(self):
        """sin²+cos² ≈ 1 for all inputs."""
        vals = np.array([0.0, 3.0, 6.0, 12.0, 18.0, 23.5])
        s, c = SensorProfile._cyclical_encode(vals, 24.0)
        np.testing.assert_allclose(s ** 2 + c ** 2, 1.0, atol=1e-10)

    def test_array_output_shape(self):
        vals = np.arange(10, dtype=float)
        s, c = SensorProfile._cyclical_encode(vals, 24.0)
        assert s.shape == (10,)
        assert c.shape == (10,)


# ── IAQStandard concrete methods ──────────────────────────────────────────


class TestIAQStandardMethods:
    """Test ABC concrete methods via BSECStandard."""

    def test_clamp_below(self, bsec_standard):
        assert bsec_standard.clamp(-10) == 0.0

    def test_clamp_above(self, bsec_standard):
        assert bsec_standard.clamp(600) == 500.0

    def test_clamp_in_range(self, bsec_standard):
        assert bsec_standard.clamp(250) == 250.0

    def test_categorize_boundary_50(self, bsec_standard):
        assert bsec_standard.categorize(50) == "Excellent"

    def test_categorize_boundary_100(self, bsec_standard):
        assert bsec_standard.categorize(100) == "Good"

    def test_categorize_between(self, bsec_standard):
        assert bsec_standard.categorize(75) == "Good"

    def test_category_distribution_empty(self, bsec_standard):
        dist = bsec_standard.category_distribution([])
        assert all(v == 0 for v in dist.values())

    def test_category_distribution_mixed(self, bsec_standard):
        values = [25, 75, 150, 250, 400]
        dist = bsec_standard.category_distribution(values)
        assert dist["Excellent"] == 1
        assert dist["Good"] == 1
        assert dist["Moderate"] == 1
        assert dist["Poor"] == 1
        assert dist["Very Poor"] == 1


# ── Registry ──────────────────────────────────────────────────────────────


class TestRegistry:
    def test_register_sensor(self):
        import app.builtin_profiles  # noqa: F401
        assert "bme680" in _SENSOR_REGISTRY
        assert "sps30" in _SENSOR_REGISTRY

    def test_register_standard(self):
        import app.builtin_profiles  # noqa: F401
        assert "bsec" in _STANDARD_REGISTRY
        assert "epa_aqi" in _STANDARD_REGISTRY

    def test_get_sensor_profile(self, monkeypatch):
        import app.builtin_profiles  # noqa: F401
        from app.profiles import get_sensor_profile
        from app.config import Settings

        monkeypatch.setattr(
            Settings, "load_model_config",
            lambda self: {"sensor": {"type": "bme680"}},
        )
        profile = get_sensor_profile()
        assert profile.name == "bme680"

    def test_get_iaq_standard(self, monkeypatch):
        import app.builtin_profiles  # noqa: F401
        from app.profiles import get_iaq_standard
        from app.config import Settings

        monkeypatch.setattr(
            Settings, "load_model_config",
            lambda self: {"iaq_standard": {"type": "bsec"}},
        )
        standard = get_iaq_standard()
        assert standard.name == "bsec"


# ── BME680 Envelope Detection ────────────────────────────────────────────


class TestBME680EnvelopeFeatures:
    """Tests for baseline envelope detection features on BME680Profile."""

    @pytest.fixture
    def sample_raw(self):
        """100 samples of plausible BME680 readings."""
        rng = np.random.RandomState(42)
        n = 100
        return np.column_stack([
            rng.uniform(20, 30, n),       # temperature
            rng.uniform(30, 70, n),       # rel_humidity
            rng.uniform(990, 1020, n),    # pressure
            rng.uniform(50000, 500000, n),  # voc_resistance
        ])

    @pytest.fixture
    def sample_timestamps(self):
        return pd.date_range("2026-01-01", periods=100, freq="3s").values

    def test_engineered_feature_names(self, bme680_profile):
        expected = [
            "abs_humidity",
            "baseline_24h", "gas_ratio_24h", "log_ratio_24h",
            "baseline_7d", "gas_ratio_7d", "log_ratio_7d",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        ]
        assert bme680_profile.engineered_feature_names == expected

    def test_total_features_is_15(self, bme680_profile):
        assert bme680_profile.total_features == 15

    def test_engineer_features_shape(self, bme680_profile, sample_raw, sample_timestamps):
        result = bme680_profile.engineer_features(sample_raw, timestamps=sample_timestamps)
        assert result.shape == (100, 15)

    def test_engineer_features_shape_no_timestamps(self, bme680_profile, sample_raw):
        result = bme680_profile.engineer_features(sample_raw)
        assert result.shape == (100, 15)

    def test_engineer_features_single_shape(self, bme680_profile):
        reading = {
            "temperature": 25.0, "rel_humidity": 50.0,
            "pressure": 1013.0, "voc_resistance": 200000.0,
        }
        baselines = {"voc_resistance": 300000.0, "baseline_24h": 350000.0, "baseline_7d": 320000.0}
        result = bme680_profile.engineer_features_single(
            reading, baselines, timestamp=datetime(2026, 3, 15, 14, 30),
        )
        assert result.shape == (15,)

    def test_envelope_baselines_computed(self, bme680_profile, sample_raw, sample_timestamps):
        baselines = bme680_profile.compute_baselines(sample_raw, timestamps=sample_timestamps)
        assert "baseline_24h" in baselines
        assert "baseline_7d" in baselines
        assert "voc_resistance" in baselines
        assert baselines["baseline_24h"] > 0
        assert baselines["baseline_7d"] > 0

    def test_envelope_baselines_no_timestamps(self, bme680_profile, sample_raw):
        baselines = bme680_profile.compute_baselines(sample_raw)
        assert "baseline_24h" in baselines
        assert "baseline_7d" in baselines

    def test_gas_ratio_bounded(self, bme680_profile, sample_raw, sample_timestamps):
        result = bme680_profile.engineer_features(sample_raw, timestamps=sample_timestamps)
        # gas_ratio_24h is column index 6 (4 raw + abs_humidity + baseline_24h + gas_ratio_24h)
        gas_ratio_24h = result[:, 6]
        log_ratio_24h = result[:, 7]
        assert np.all(gas_ratio_24h > 0)
        # EWM smoothing means ratio can transiently exceed 1 during spikes,
        # but should be roughly in (0, ~2] for realistic data
        assert np.all(gas_ratio_24h < 10.0)
        # log_ratio is finite (no NaN/Inf from clipping)
        assert np.all(np.isfinite(log_ratio_24h))

    def test_backward_compat_old_baselines(self, bme680_profile):
        """Old model artifacts with only voc_resistance key still work."""
        reading = {
            "temperature": 25.0, "rel_humidity": 50.0,
            "pressure": 1013.0, "voc_resistance": 200000.0,
        }
        old_baselines = {"voc_resistance": 300000.0}  # no baseline_24h/7d
        result = bme680_profile.engineer_features_single(reading, old_baselines)
        assert result.shape == (15,)
        # Both ratios should use voc_resistance as fallback
        gas_ratio_24h = result[6]
        gas_ratio_7d = result[9]
        expected_ratio = 200000.0 / 300000.0
        assert abs(gas_ratio_24h - expected_ratio) < 1e-6
        assert abs(gas_ratio_7d - expected_ratio) < 1e-6
