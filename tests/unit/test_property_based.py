"""Property-based tests using Hypothesis.

These tests declare invariants that must hold for ALL valid inputs,
not just hand-picked examples. Hypothesis generates adversarial inputs
(including edge cases, boundary values, and shrunk counterexamples)
to find violations.

Bug classes caught: NaN propagation, shape mismatches, off-by-one errors,
silent data reinterpretation, domain errors at boundary values.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings as h_settings, assume, HealthCheck, Phase
from hypothesis import strategies as st

# Fixtures are stateless profile/standard singletons — safe to reuse across inputs.
_FIXTURE_OK = [HealthCheck.function_scoped_fixture, HealthCheck.too_slow]

import app.builtin_profiles  # noqa: F401 — register profiles


# ── Strategies ────────────────────────────────────────────────────────────

# Valid BME680 sensor ranges (from quantities.yaml)
st_temperature = st.floats(min_value=-40.0, max_value=85.0)
st_humidity = st.floats(min_value=0.5, max_value=100.0)  # >0 to avoid log(0)
st_pressure = st.floats(min_value=300.0, max_value=1100.0)
st_voc_resistance = st.floats(min_value=1000.0, max_value=2_000_000.0)

st_bme680_reading = st.fixed_dictionaries({
    "temperature": st_temperature,
    "rel_humidity": st_humidity,
    "pressure": st_pressure,
    "voc_resistance": st_voc_resistance,
})

st_hour = st.floats(min_value=0.0, max_value=23.0)
st_dow = st.floats(min_value=0.0, max_value=6.0)
st_datetime = st.datetimes(
    min_value=pd.Timestamp("2020-01-01").to_pydatetime(),
    max_value=pd.Timestamp("2030-12-31").to_pydatetime(),
)

st_window_size = st.integers(min_value=1, max_value=60)
st_n_samples = st.integers(min_value=2, max_value=200)


# ── A. Feature engineering invariants ─────────────────────────────────────


class TestEngineerFeaturesSingleProperties:
    """Properties of BME680Profile.engineer_features_single()."""

    @given(reading=st_bme680_reading)
    @h_settings(max_examples=200, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_output_length_equals_total_features(self, reading, bme680_profile):
        """Output array length must always equal profile.total_features."""
        result = bme680_profile.engineer_features_single(reading)
        assert len(result) == bme680_profile.total_features

    @given(reading=st_bme680_reading)
    @h_settings(max_examples=200, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_no_nan_in_output(self, reading, bme680_profile):
        """No NaN values for any valid-range input."""
        result = bme680_profile.engineer_features_single(reading)
        assert not np.any(np.isnan(result)), f"NaN in output: {result}"

    @given(reading=st_bme680_reading)
    @h_settings(max_examples=200, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_no_inf_in_output(self, reading, bme680_profile):
        """No Inf values for any valid-range input."""
        result = bme680_profile.engineer_features_single(reading)
        assert np.all(np.isfinite(result)), f"Inf in output: {result}"

    @given(reading=st_bme680_reading, dt=st_datetime)
    @h_settings(max_examples=100, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_timestamp_does_not_change_length(self, reading, dt, bme680_profile):
        """Output length is the same regardless of timestamp."""
        without_ts = bme680_profile.engineer_features_single(reading)
        with_ts = bme680_profile.engineer_features_single(reading, timestamp=dt)
        assert len(without_ts) == len(with_ts)

    @given(reading=st_bme680_reading)
    @h_settings(max_examples=100, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_raw_features_preserved_in_order(self, reading, bme680_profile):
        """First N elements of output match the raw feature values in order."""
        result = bme680_profile.engineer_features_single(reading)
        for i, feat in enumerate(bme680_profile.raw_features):
            assert result[i] == pytest.approx(reading[feat]), \
                f"Raw feature {feat} at index {i}: expected {reading[feat]}, got {result[i]}"


class TestEngineerFeaturesBatchProperties:
    """Properties of BME680Profile.engineer_features() (batch)."""

    @given(n=st.integers(min_value=1, max_value=100))
    @h_settings(max_examples=50, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_output_shape(self, n, bme680_profile):
        """Output shape is always (n_samples, total_features)."""
        rng = np.random.default_rng(42)
        raw = np.column_stack([
            rng.uniform(-40, 85, n),
            rng.uniform(0.5, 100, n),
            rng.uniform(300, 1100, n),
            rng.uniform(1000, 2_000_000, n),
        ])
        result = bme680_profile.engineer_features(raw)
        assert result.shape == (n, bme680_profile.total_features)

    @given(n=st.integers(min_value=10, max_value=50))
    @h_settings(max_examples=30, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_batch_single_consistency(self, n, bme680_profile):
        """Non-envelope columns must match between batch and single-reading.

        Envelope columns (baseline_24h/7d, gas_ratio_24h/7d, log_ratio_24h/7d)
        use rolling windows in batch mode but static final baselines in single
        mode, so they intentionally differ for all rows except the last.
        """
        rng = np.random.default_rng(42)
        raw = np.column_stack([
            rng.uniform(-40, 85, n),
            rng.uniform(0.5, 100, n),
            rng.uniform(300, 1100, n),
            rng.uniform(1000, 2_000_000, n),
        ])
        baselines = bme680_profile.compute_baselines(raw)
        batch = bme680_profile.engineer_features(raw, baselines=baselines)

        # Columns that are identical between batch and single:
        # 0-3: raw, 4: abs_humidity, 11-14: cyclical
        stable_cols = list(range(5)) + list(range(11, 15))

        for idx in [0, n // 2, n - 1]:
            reading = {
                feat: float(raw[idx, i])
                for i, feat in enumerate(bme680_profile.raw_features)
            }
            single = bme680_profile.engineer_features_single(
                reading, baselines=baselines
            )
            np.testing.assert_allclose(
                batch[idx, stable_cols], single[stable_cols], rtol=1e-6,
                err_msg=f"Mismatch at row {idx}",
            )


# ── B. Cyclical encoding invariants ──────────────────────────────────────


class TestCyclicalEncodeProperties:

    @given(value=st.floats(min_value=0.0, max_value=1000.0),
           period=st.floats(min_value=0.1, max_value=1000.0))
    @h_settings(max_examples=200)
    def test_output_bounded(self, value, period):
        """sin/cos are always in [-1, 1]."""
        from app.profiles import SensorProfile
        sin_val, cos_val = SensorProfile._cyclical_encode(
            np.array([value]), period
        )
        assert -1.0 <= sin_val[0] <= 1.0
        assert -1.0 <= cos_val[0] <= 1.0

    @given(value=st.floats(min_value=0.0, max_value=1000.0),
           period=st.floats(min_value=0.1, max_value=1000.0))
    @h_settings(max_examples=200)
    def test_unit_circle(self, value, period):
        """sin^2 + cos^2 = 1 for all inputs."""
        from app.profiles import SensorProfile
        sin_val, cos_val = SensorProfile._cyclical_encode(
            np.array([value]), period
        )
        assert sin_val[0] ** 2 + cos_val[0] ** 2 == pytest.approx(1.0, abs=1e-10)

    @given(period=st.floats(min_value=0.1, max_value=1000.0))
    @h_settings(max_examples=100)
    def test_period_wraparound(self, period):
        """Value and value+period produce the same encoding."""
        from app.profiles import SensorProfile
        v = 5.0
        sin_a, cos_a = SensorProfile._cyclical_encode(np.array([v]), period)
        sin_b, cos_b = SensorProfile._cyclical_encode(np.array([v + period]), period)
        assert sin_a[0] == pytest.approx(sin_b[0], abs=1e-10)
        assert cos_a[0] == pytest.approx(cos_b[0], abs=1e-10)


# ── C. IAQ standard invariants ───────────────────────────────────────────


class TestIAQStandardProperties:

    @given(value=st.floats(min_value=-1000, max_value=2000))
    @h_settings(max_examples=200, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_clamp_idempotent(self, value, bsec_standard):
        """Clamping twice gives the same result as clamping once."""
        clamped = bsec_standard.clamp(value)
        assert bsec_standard.clamp(clamped) == clamped

    @given(value=st.floats(min_value=-1000, max_value=2000))
    @h_settings(max_examples=200, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_clamp_within_range(self, value, bsec_standard):
        """Clamped output is always within scale_range."""
        lo, hi = bsec_standard.scale_range
        clamped = bsec_standard.clamp(value)
        assert lo <= clamped <= hi

    @given(value=st.floats(min_value=0, max_value=500))
    @h_settings(max_examples=200, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_categorize_returns_valid_category(self, value, bsec_standard):
        """categorize() always returns one of the defined category names."""
        valid_names = {name for _, name in bsec_standard.categories}
        assert bsec_standard.categorize(value) in valid_names

    @given(values=st.lists(st.floats(min_value=0, max_value=500), min_size=1, max_size=50))
    @h_settings(max_examples=50, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_category_distribution_sums_to_n(self, values, bsec_standard):
        """category_distribution counts sum to len(values)."""
        dist = bsec_standard.category_distribution(values)
        assert sum(dist.values()) == len(values)


# ── D. Schema fingerprint invariants ─────────────────────────────────────


class TestSchemaFingerprintProperties:

    @given(
        sensor=st.text(min_size=1, max_size=20),
        std=st.text(min_size=1, max_size=20),
        ws=st.integers(min_value=1, max_value=100),
        nf=st.integers(min_value=1, max_value=50),
        mt=st.sampled_from(["mlp", "kan", "lstm", "cnn", "bnn"]),
    )
    @h_settings(max_examples=200, deadline=None)
    def test_deterministic(self, sensor, std, ws, nf, mt):
        """Same inputs always produce the same fingerprint."""
        from training.utils import compute_schema_fingerprint
        fp1 = compute_schema_fingerprint(sensor, std, ws, nf, mt)
        fp2 = compute_schema_fingerprint(sensor, std, ws, nf, mt)
        assert fp1 == fp2

    @given(
        ws=st.integers(min_value=1, max_value=100),
        nf=st.integers(min_value=1, max_value=50),
    )
    @h_settings(max_examples=100)
    def test_model_type_sensitivity(self, ws, nf):
        """Different model types with same other params produce different fingerprints."""
        from training.utils import compute_schema_fingerprint
        fps = set()
        for mt in ["mlp", "kan", "lstm", "cnn", "bnn"]:
            fps.add(compute_schema_fingerprint("bme680", "bsec", ws, nf, mt))
        assert len(fps) == 5


# ── E. Sliding window invariants ─────────────────────────────────────────


class TestSlidingWindowProperties:

    @given(
        n=st.integers(min_value=2, max_value=200),
        nf=st.integers(min_value=1, max_value=10),
        ws=st.integers(min_value=1, max_value=20),
    )
    @h_settings(max_examples=100)
    def test_output_shape(self, n, nf, ws):
        """Output shape follows the formula: (n - ws + 1, ws * nf)."""
        assume(n >= ws)
        from training.utils import create_sliding_windows
        features = np.random.randn(n, nf)
        targets = np.random.randn(n)
        X, y = create_sliding_windows(features, targets, window_size=ws)
        assert X.shape == (n - ws + 1, ws * nf)
        assert y.shape == (n - ws + 1,)

    @given(
        n=st.integers(min_value=5, max_value=50),
        ws=st.integers(min_value=1, max_value=5),
    )
    @h_settings(max_examples=50)
    def test_target_alignment(self, n, ws):
        """Target y[i] corresponds to the last timestep in window i."""
        assume(n >= ws)
        from training.utils import create_sliding_windows
        features = np.arange(n).reshape(-1, 1).astype(float)
        targets = np.arange(n).astype(float) * 10
        _, y = create_sliding_windows(features, targets, window_size=ws)
        for i in range(len(y)):
            assert y[i] == targets[i + ws - 1]

    @given(n=st.integers(min_value=1, max_value=10))
    @h_settings(max_examples=20)
    def test_insufficient_data_returns_empty(self, n):
        """When n < window_size, output has 0 rows."""
        from training.utils import create_sliding_windows
        ws = n + 1
        features = np.random.randn(n, 3)
        targets = np.random.randn(n)
        X, y = create_sliding_windows(features, targets, window_size=ws)
        assert X.shape[0] == 0
        assert y.shape[0] == 0


# ── F. Absolute humidity domain ──────────────────────────────────────────


class TestAbsoluteHumidityProperties:

    @given(
        temp=st.floats(min_value=-40, max_value=85),
        rh=st.floats(min_value=0.5, max_value=100),
    )
    @h_settings(max_examples=200)
    def test_positive_output(self, temp, rh):
        """Absolute humidity is always positive for valid inputs."""
        from training.utils import calculate_absolute_humidity
        result = calculate_absolute_humidity(np.array([temp]), np.array([rh]))
        assert result[0] > 0, f"Non-positive abs humidity: {result[0]} at T={temp}, RH={rh}"

    @given(
        temp=st.floats(min_value=-40, max_value=85),
        rh=st.floats(min_value=0.5, max_value=100),
    )
    @h_settings(max_examples=200)
    def test_finite_output(self, temp, rh):
        """Output is always finite for valid inputs."""
        from training.utils import calculate_absolute_humidity
        result = calculate_absolute_humidity(np.array([temp]), np.array([rh]))
        assert np.isfinite(result[0])

    def test_zero_humidity_domain_error(self):
        """rel_humidity=0 causes log(0) — documents known domain boundary.

        This is a known mathematical singularity at the boundary of the
        valid range [0, 100]. Not a bug in practice (0% RH is physically
        impossible indoors) but documents the domain constraint.
        """
        from training.utils import calculate_absolute_humidity
        with np.errstate(divide="raise"):
            with pytest.raises(FloatingPointError):
                calculate_absolute_humidity(np.array([25.0]), np.array([0.0]))

    @given(temp=st.floats(min_value=0, max_value=85))
    @h_settings(max_examples=50)
    def test_monotonic_in_humidity(self, temp):
        """At fixed temperature, absolute humidity increases with relative humidity."""
        from training.utils import calculate_absolute_humidity
        rh_values = np.array([20.0, 40.0, 60.0, 80.0])
        results = calculate_absolute_humidity(
            np.full_like(rh_values, temp), rh_values
        )
        assert all(results[i] < results[i + 1] for i in range(len(results) - 1))


# ── G. Model forward pass shape assertions ───────────────────────────────


class TestModelForwardShapeInvariants:
    """Verify that model forward passes produce correct output shapes."""

    @given(
        batch=st.integers(min_value=1, max_value=16),
        ws=st.integers(min_value=2, max_value=10),
        nf=st.integers(min_value=1, max_value=8),
    )
    @h_settings(max_examples=30, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_mlp_output_shape(self, batch, ws, nf):
        """MLP always produces (batch, 1) output."""
        import torch
        from app.models import MLPRegressor
        model = MLPRegressor(input_dim=ws * nf)
        model.eval()
        x = torch.randn(batch, ws * nf)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, 1)

    @given(
        batch=st.integers(min_value=1, max_value=16),
        ws=st.integers(min_value=3, max_value=10),
        nf=st.integers(min_value=1, max_value=8),
    )
    @h_settings(max_examples=30, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_cnn_output_shape(self, batch, ws, nf):
        """CNN always produces (batch, 1) output."""
        import torch
        from app.models import CNNRegressor
        model = CNNRegressor(window_size=ws, num_features=nf)
        model.eval()
        x = torch.randn(batch, ws * nf)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, 1)

    @given(
        batch=st.integers(min_value=1, max_value=16),
        ws=st.integers(min_value=2, max_value=10),
        nf=st.integers(min_value=1, max_value=8),
    )
    @h_settings(max_examples=30, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_lstm_output_shape(self, batch, ws, nf):
        """LSTM always produces (batch, 1) output."""
        import torch
        from app.models import LSTMRegressor
        model = LSTMRegressor(window_size=ws, num_features=nf, num_layers=1)
        model.eval()
        x = torch.randn(batch, ws * nf)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, 1)

    @given(
        batch=st.integers(min_value=1, max_value=16),
        ws=st.integers(min_value=2, max_value=10),
        nf=st.integers(min_value=1, max_value=8),
    )
    @h_settings(max_examples=30, suppress_health_check=_FIXTURE_OK, deadline=None)
    def test_bnn_output_shape(self, batch, ws, nf):
        """BNN always produces (batch, 1) output."""
        import torch
        from app.models import BNNRegressor
        model = BNNRegressor(input_dim=ws * nf)
        model.eval()
        x = torch.randn(batch, ws * nf)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, 1)


# ── H. Scaler round-trip ─────────────────────────────────────────────────


class TestScalerRoundTrip:

    @given(
        n=st.integers(min_value=5, max_value=50),
        nf=st.integers(min_value=1, max_value=10),
    )
    @h_settings(max_examples=30)
    def test_standard_scaler_roundtrip(self, n, nf):
        """StandardScaler inverse_transform(transform(X)) ≈ X."""
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, nf)) * 100 + 50
        scaler = StandardScaler().fit(X)
        X_rt = scaler.inverse_transform(scaler.transform(X))
        np.testing.assert_allclose(X_rt, X, rtol=1e-6)

    @given(
        n=st.integers(min_value=5, max_value=50),
    )
    @h_settings(max_examples=30)
    def test_minmax_scaler_roundtrip(self, n):
        """MinMaxScaler inverse_transform(transform(y)) ≈ y."""
        from sklearn.preprocessing import MinMaxScaler
        rng = np.random.default_rng(42)
        y = rng.uniform(0, 500, (n, 1))
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(y)
        y_rt = scaler.inverse_transform(scaler.transform(y))
        np.testing.assert_allclose(y_rt, y, rtol=1e-6)
