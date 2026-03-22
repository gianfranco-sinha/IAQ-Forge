"""Tests for app/builtin_profiles.py — BME680Profile, BSECStandard, SPS30, EPA AQI."""
from datetime import datetime

import numpy as np
import pytest


# ── BME680Profile ──────────────────────────────────────────────────────────


class TestBME680Profile:
    def test_name(self, bme680_profile):
        assert bme680_profile.name == "bme680"

    def test_raw_features(self, bme680_profile):
        assert bme680_profile.raw_features == [
            "temperature", "rel_humidity", "pressure", "voc_resistance"
        ]

    def test_engineered_feature_names(self, bme680_profile):
        assert bme680_profile.engineered_feature_names == [
            "abs_humidity",
            "baseline_24h", "gas_ratio_24h", "log_ratio_24h",
            "baseline_7d", "gas_ratio_7d", "log_ratio_7d",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        ]

    def test_total_features(self, bme680_profile):
        assert bme680_profile.total_features == 15

    def test_all_feature_names_length(self, bme680_profile):
        assert len(bme680_profile.all_feature_names) == 15

    def test_all_feature_names_order(self, bme680_profile):
        names = bme680_profile.all_feature_names
        assert names[:4] == ["temperature", "rel_humidity", "pressure", "voc_resistance"]
        assert names[4:] == [
            "abs_humidity",
            "baseline_24h", "gas_ratio_24h", "log_ratio_24h",
            "baseline_7d", "gas_ratio_7d", "log_ratio_7d",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        ]

    def test_quality_column(self, bme680_profile):
        assert bme680_profile.quality_column == "iaq_accuracy"

    def test_quality_min(self, bme680_profile):
        assert bme680_profile.quality_min == 2

    def test_valid_ranges_includes_raw_features(self, bme680_profile):
        vr = bme680_profile.valid_ranges
        assert "temperature" in vr
        assert "rel_humidity" in vr
        assert "pressure" in vr
        assert "voc_resistance" in vr

    def test_valid_ranges_includes_quality(self, bme680_profile):
        vr = bme680_profile.valid_ranges
        assert "iaq_accuracy" in vr
        assert vr["iaq_accuracy"] == (2, 3)

    def test_compute_baselines(self, bme680_profile, sample_raw_data):
        baselines = bme680_profile.compute_baselines(sample_raw_data)
        assert "voc_resistance" in baselines
        assert isinstance(baselines["voc_resistance"], float)

    def test_engineer_features_shape_with_timestamps(
        self, bme680_profile, sample_raw_data, sample_timestamps
    ):
        result = bme680_profile.engineer_features(
            sample_raw_data, timestamps=sample_timestamps.values
        )
        assert result.shape == (100, 15)

    def test_engineer_features_shape_without_timestamps(
        self, bme680_profile, sample_raw_data
    ):
        result = bme680_profile.engineer_features(sample_raw_data)
        assert result.shape == (100, 15)

    def test_engineer_features_cyclical_range(
        self, bme680_profile, sample_raw_data, sample_timestamps
    ):
        result = bme680_profile.engineer_features(
            sample_raw_data, timestamps=sample_timestamps.values
        )
        # Columns 11-14 are cyclical features (hour_sin, hour_cos, dow_sin, dow_cos)
        for col in range(11, 15):
            assert np.all(result[:, col] >= -1.0 - 1e-10)
            assert np.all(result[:, col] <= 1.0 + 1e-10)

    def test_engineer_features_sin_cos_norm(
        self, bme680_profile, sample_raw_data, sample_timestamps
    ):
        result = bme680_profile.engineer_features(
            sample_raw_data, timestamps=sample_timestamps.values
        )
        # hour: sin²+cos² ≈ 1
        hour_norm = result[:, 11] ** 2 + result[:, 12] ** 2
        np.testing.assert_allclose(hour_norm, 1.0, atol=1e-10)
        # dow: sin²+cos² ≈ 1
        dow_norm = result[:, 13] ** 2 + result[:, 14] ** 2
        np.testing.assert_allclose(dow_norm, 1.0, atol=1e-10)

    def test_engineer_features_voc_ratio(
        self, bme680_profile, sample_raw_data
    ):
        """gas_ratio_24h (col 6) should be positive and finite."""
        result = bme680_profile.engineer_features(sample_raw_data)
        gas_ratio_24h = result[:, 6]
        assert np.all(gas_ratio_24h > 0)
        assert np.all(np.isfinite(gas_ratio_24h))

    def test_engineer_features_abs_humidity_positive(
        self, bme680_profile, sample_raw_data
    ):
        result = bme680_profile.engineer_features(sample_raw_data)
        assert np.all(result[:, 4] > 0)  # abs_humidity is now col 4

    def test_engineer_features_single_length(
        self, bme680_profile, sample_reading
    ):
        result = bme680_profile.engineer_features_single(sample_reading)
        assert result.shape == (15,)

    def test_engineer_features_single_with_timestamp(
        self, bme680_profile, sample_reading
    ):
        ts = datetime(2026, 1, 15, 12, 0, 0)
        result = bme680_profile.engineer_features_single(sample_reading, timestamp=ts)
        # hour_sin (idx 11) and hour_cos (idx 12) should be non-trivial for hour=12
        # sin(2π*12/24)=0, cos(2π*12/24)=-1
        assert abs(result[11] - 0.0) < 1e-10  # hour_sin at noon
        assert abs(result[12] - (-1.0)) < 1e-10  # hour_cos at noon

    def test_engineer_features_single_without_timestamp(
        self, bme680_profile, sample_reading
    ):
        result = bme680_profile.engineer_features_single(sample_reading)
        # Without timestamp: hour=0, dow=0 → sin(0)=0, cos(0)=1
        assert abs(result[11] - 0.0) < 1e-10  # hour_sin
        assert abs(result[12] - 1.0) < 1e-10  # hour_cos
        assert abs(result[13] - 0.0) < 1e-10  # dow_sin
        assert abs(result[14] - 1.0) < 1e-10  # dow_cos


# ── BSECStandard ───────────────────────────────────────────────────────────


class TestBSECStandard:
    def test_name(self, bsec_standard):
        assert bsec_standard.name == "bsec"

    def test_target_column(self, bsec_standard):
        assert bsec_standard.target_column == "iaq"

    def test_scale_range(self, bsec_standard):
        assert bsec_standard.scale_range == (0.0, 500.0)

    def test_categories_count(self, bsec_standard):
        assert len(bsec_standard.categories) == 5

    def test_categorize_excellent(self, bsec_standard):
        assert bsec_standard.categorize(25) == "Excellent"

    def test_categorize_good(self, bsec_standard):
        assert bsec_standard.categorize(75) == "Good"

    def test_categorize_moderate(self, bsec_standard):
        assert bsec_standard.categorize(150) == "Moderate"

    def test_categorize_poor(self, bsec_standard):
        assert bsec_standard.categorize(250) == "Poor"

    def test_categorize_very_poor(self, bsec_standard):
        assert bsec_standard.categorize(400) == "Very Poor"

    def test_categorize_boundary_50(self, bsec_standard):
        assert bsec_standard.categorize(50) == "Excellent"

    def test_categorize_boundary_100(self, bsec_standard):
        assert bsec_standard.categorize(100) == "Good"

    def test_clamp_below(self, bsec_standard):
        assert bsec_standard.clamp(-10) == 0.0

    def test_clamp_above(self, bsec_standard):
        assert bsec_standard.clamp(600) == 500.0

    def test_clamp_in_range(self, bsec_standard):
        assert bsec_standard.clamp(250) == 250.0

    def test_category_distribution(self, bsec_standard):
        values = [10, 20, 60, 80, 120, 250, 350]
        dist = bsec_standard.category_distribution(values)
        assert dist["Excellent"] == 2
        assert dist["Good"] == 2
        assert dist["Moderate"] == 1
        assert dist["Poor"] == 1
        assert dist["Very Poor"] == 1


# ── SPS30Profile ──────────────────────────────────────────────────────────


class TestSPS30Profile:
    def test_name(self):
        from app.builtin_profiles import SPS30Profile
        p = SPS30Profile()
        assert p.name == "sps30"

    def test_raw_features(self):
        from app.builtin_profiles import SPS30Profile
        p = SPS30Profile()
        assert p.raw_features == ["pm1_0", "pm2_5", "pm4_0", "pm10"]

    def test_total_features(self):
        from app.builtin_profiles import SPS30Profile
        p = SPS30Profile()
        assert p.total_features == 6  # 4 raw + 2 engineered


# ── EPAAQIStandard ────────────────────────────────────────────────────────


class TestEPAAQIStandard:
    def test_name(self):
        import app.builtin_profiles  # noqa: F401
        from app.profiles import _STANDARD_REGISTRY
        s = _STANDARD_REGISTRY["epa_aqi"]()
        assert s.name == "epa_aqi"

    def test_categories_count(self):
        import app.builtin_profiles  # noqa: F401
        from app.profiles import _STANDARD_REGISTRY
        s = _STANDARD_REGISTRY["epa_aqi"]()
        assert len(s.categories) == 6

    def test_target_column(self):
        import app.builtin_profiles  # noqa: F401
        from app.profiles import _STANDARD_REGISTRY
        s = _STANDARD_REGISTRY["epa_aqi"]()
        assert s.target_column == "aqi"
