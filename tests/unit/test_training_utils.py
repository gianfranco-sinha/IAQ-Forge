"""Tests for training/utils.py — pure functions only (no model training)."""
import numpy as np
import pandas as pd
import pytest

from training.utils import (
    create_sliding_windows,
    find_contiguous_segments,
    calculate_absolute_humidity,
    compute_schema_fingerprint,
    compute_semver,
)


# ── create_sliding_windows ─────────────────────────────────────────────────


class TestCreateSlidingWindows:
    def test_basic_count(self):
        features = np.random.rand(20, 3)
        targets = np.random.rand(20)
        X, y = create_sliding_windows(features, targets, window_size=5)
        assert len(X) == 16  # 20 - 5 + 1

    def test_shape(self):
        features = np.random.rand(20, 3)
        targets = np.random.rand(20)
        X, y = create_sliding_windows(features, targets, window_size=5)
        assert X.shape == (16, 5 * 3)  # flattened

    def test_target_alignment(self):
        features = np.arange(10).reshape(-1, 1).astype(float)
        targets = np.arange(10).astype(float)
        X, y = create_sliding_windows(features, targets, window_size=3)
        # y[0] = targets[0 + 3 - 1] = targets[2]
        assert y[0] == 2.0
        assert y[-1] == 9.0

    def test_window_equals_length(self):
        features = np.random.rand(5, 2)
        targets = np.random.rand(5)
        X, y = create_sliding_windows(features, targets, window_size=5)
        assert len(X) == 1

    def test_window_exceeds_length(self):
        features = np.random.rand(3, 2)
        targets = np.random.rand(3)
        X, y = create_sliding_windows(features, targets, window_size=5)
        assert len(X) == 0

    def test_single_feature_column(self):
        features = np.arange(10).reshape(-1, 1).astype(float)
        targets = np.arange(10).astype(float)
        X, y = create_sliding_windows(features, targets, window_size=3)
        assert X.shape == (8, 3)  # window_size * 1 feature


# ── find_contiguous_segments ───────────────────────────────────────────────


class TestFindContiguousSegments:
    def test_no_gaps(self):
        idx = pd.date_range("2026-01-01", periods=100, freq="3s")
        segments, info = find_contiguous_segments(idx)
        assert info["gaps_found"] == 0
        assert info["segments"] == 1
        assert segments == [(0, 100)]

    def test_one_gap(self):
        idx1 = pd.date_range("2026-01-01 00:00", periods=50, freq="3s")
        idx2 = pd.date_range("2026-01-01 01:00", periods=50, freq="3s")
        idx = idx1.union(idx2)
        segments, info = find_contiguous_segments(idx)
        assert info["gaps_found"] == 1
        assert info["segments"] == 2
        assert len(segments) == 2

    def test_multiple_gaps(self):
        idx1 = pd.date_range("2026-01-01 00:00", periods=20, freq="3s")
        idx2 = pd.date_range("2026-01-01 01:00", periods=20, freq="3s")
        idx3 = pd.date_range("2026-01-01 02:00", periods=20, freq="3s")
        idx = idx1.union(idx2).union(idx3)
        segments, info = find_contiguous_segments(idx)
        assert info["gaps_found"] == 2
        assert info["segments"] == 3

    def test_non_datetime_index(self):
        idx = pd.RangeIndex(100)
        segments, info = find_contiguous_segments(idx)
        assert segments == [(0, 100)]
        assert info["skipped"] is True

    def test_single_element(self):
        idx = pd.DatetimeIndex(["2026-01-01"])
        segments, info = find_contiguous_segments(idx)
        assert segments == [(0, 1)]
        assert info["skipped"] is True

    def test_gap_info_keys(self):
        idx = pd.date_range("2026-01-01", periods=100, freq="3s")
        _, info = find_contiguous_segments(idx)
        assert "gaps_found" in info
        assert "segments" in info
        assert "median_interval_seconds" in info


# ── calculate_absolute_humidity ────────────────────────────────────────────


class TestCalculateAbsoluteHumidity:
    def test_known_value(self):
        """25°C, 50% RH → positive value (formula uses simplified Magnus)."""
        result = calculate_absolute_humidity(np.array([25.0]), np.array([50.0]))
        assert 0.05 < result[0] < 0.5  # ~0.115 g/m³ per formula

    def test_positive(self):
        result = calculate_absolute_humidity(np.array([20.0]), np.array([60.0]))
        assert result[0] > 0

    def test_array_shape(self):
        temps = np.array([20.0, 25.0, 30.0])
        rhs = np.array([50.0, 60.0, 70.0])
        result = calculate_absolute_humidity(temps, rhs)
        assert result.shape == (3,)

    def test_scalar_like(self):
        result = calculate_absolute_humidity(25.0, 50.0)
        assert float(result) > 0


# ── compute_schema_fingerprint ─────────────────────────────────────────────


class TestComputeSchemaFingerprint:
    def test_deterministic(self):
        fp1 = compute_schema_fingerprint("bme680", "bsec", 10, 10, "mlp")
        fp2 = compute_schema_fingerprint("bme680", "bsec", 10, 10, "mlp")
        assert fp1 == fp2

    def test_different_inputs(self):
        fp1 = compute_schema_fingerprint("bme680", "bsec", 10, 10, "mlp")
        fp2 = compute_schema_fingerprint("sps30", "bsec", 10, 10, "mlp")
        assert fp1 != fp2

    def test_length(self):
        fp = compute_schema_fingerprint("bme680", "bsec", 10, 10, "mlp")
        assert len(fp) == 12

    def test_hex_string(self):
        fp = compute_schema_fingerprint("bme680", "bsec", 10, 10, "mlp")
        int(fp, 16)  # should not raise

    def test_changing_single_field(self):
        base = compute_schema_fingerprint("bme680", "bsec", 10, 10, "mlp")
        assert compute_schema_fingerprint("bme680", "bsec", 10, 10, "kan") != base
        assert compute_schema_fingerprint("bme680", "bsec", 60, 10, "mlp") != base
        assert compute_schema_fingerprint("bme680", "bsec", 10, 6, "mlp") != base


# ── compute_semver ─────────────────────────────────────────────────────────


class TestComputeSemver:
    def test_no_previous_runs(self):
        ver = compute_semver("mlp", "fp1", "dfp1", [])
        assert ver == "1.0.0"

    def test_schema_change_major_bump(self):
        runs = [{
            "model_type": "mlp", "is_active": True,
            "version": "mlp-1.0.0",
            "schema_fingerprint": "aaa", "data_fingerprint": "bbb",
        }]
        ver = compute_semver("mlp", "zzz", "bbb", runs)
        assert ver == "2.0.0"

    def test_data_change_minor_bump(self):
        runs = [{
            "model_type": "mlp", "is_active": True,
            "version": "mlp-1.0.0",
            "schema_fingerprint": "aaa", "data_fingerprint": "bbb",
        }]
        ver = compute_semver("mlp", "aaa", "ccc", runs)
        assert ver == "1.1.0"

    def test_metrics_change_minor_bump(self):
        runs = [{
            "model_type": "mlp", "is_active": True,
            "version": "mlp-1.0.0",
            "schema_fingerprint": "aaa", "data_fingerprint": "bbb",
            "metrics": {"mae": 10.0, "rmse": 15.0, "r2": 0.8},
        }]
        ver = compute_semver("mlp", "aaa", "bbb", runs,
                             metrics={"mae": 9.0, "rmse": 14.0, "r2": 0.85})
        assert ver == "1.1.0"

    def test_nothing_changed_patch_bump(self):
        runs = [{
            "model_type": "mlp", "is_active": True,
            "version": "mlp-1.0.0",
            "schema_fingerprint": "aaa", "data_fingerprint": "bbb",
            "metrics": {"mae": 10.0, "rmse": 15.0, "r2": 0.8},
        }]
        ver = compute_semver("mlp", "aaa", "bbb", runs,
                             metrics={"mae": 10.0, "rmse": 15.0, "r2": 0.8})
        assert ver == "1.0.1"

    def test_legacy_version_ignored(self):
        runs = [{
            "model_type": "mlp", "is_active": True,
            "version": "mlp-v6",
            "schema_fingerprint": "aaa", "data_fingerprint": "bbb",
        }]
        ver = compute_semver("mlp", "aaa", "bbb", runs)
        assert ver == "1.0.0"
