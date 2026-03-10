"""Tests for app/profiles.py — ABC concrete methods, cyclical encoding, registry."""
import numpy as np
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
