import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def bme680_profile():
    import app.builtin_profiles  # noqa: F401 — registers profiles
    from app.builtin_profiles import BME680Profile
    return BME680Profile()


@pytest.fixture
def bsec_standard():
    import app.builtin_profiles  # noqa: F401
    from app.profiles import get_iaq_standard
    return get_iaq_standard()


@pytest.fixture
def sample_raw_data():
    """100 rows of realistic BME680 sensor data."""
    rng = np.random.default_rng(42)
    return np.column_stack([
        rng.uniform(18, 28, 100),       # temperature
        rng.uniform(40, 80, 100),       # rel_humidity
        rng.uniform(990, 1020, 100),    # pressure
        rng.uniform(5000, 500000, 100), # voc_resistance
    ])


@pytest.fixture
def sample_reading():
    """Single BME680 reading as dict."""
    return {
        "temperature": 22.5,
        "rel_humidity": 55.0,
        "pressure": 1013.25,
        "voc_resistance": 50000.0,
    }


@pytest.fixture
def sample_timestamps():
    """100 timestamps spanning 1 day at ~15min intervals."""
    return pd.date_range("2026-01-15 00:00", periods=100, freq="15min")


# ── Tier 2 shared fixtures ──────────────────────────────────────────────

@pytest.fixture
def patched_models_base(tmp_path, monkeypatch):
    """Monkeypatch settings.TRAINED_MODELS_BASE to tmp_path for filesystem isolation."""
    from app.config import settings
    monkeypatch.setattr(settings, "TRAINED_MODELS_BASE", str(tmp_path))
    return tmp_path


@pytest.fixture
def model_artifact_dir(tmp_path, monkeypatch):
    """Build MLP, save full artifact set to tmp_path/mlp, return dir path."""
    import app.builtin_profiles  # noqa: F401
    from app.config import settings
    from app.models import build_model
    from training.utils import save_trained_model
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    monkeypatch.setattr(settings, "TRAINED_MODELS_BASE", str(tmp_path))

    # Use a controlled config so build_model uses our window_size/num_features
    test_cfg = settings._get_default_model_config()
    test_cfg["mlp"]["window_size"] = 5
    test_cfg["mlp"]["num_features"] = 15
    test_cfg["global"]["num_features"] = 15
    monkeypatch.setattr(settings, "_model_config_cache", test_cfg)

    model = build_model("mlp", window_size=5, num_features=15)
    model_dir = tmp_path / "mlp"

    rng = np.random.default_rng(99)
    n_features = 5 * 15  # window_size * num_features
    X = rng.standard_normal((20, n_features))
    y = rng.uniform(0, 500, 20)

    feature_scaler = StandardScaler().fit(X)
    target_scaler = MinMaxScaler(feature_range=(0, 1)).fit(y.reshape(-1, 1))

    save_trained_model(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        model_type="mlp",
        window_size=5,
        model_dir=str(model_dir),
        metrics={"mae": 10.0, "rmse": 15.0, "r2": 0.85},
        sensor_type="bme680",
        iaq_standard="bsec",
    )
    return model_dir


@pytest.fixture
def fast_pipeline_kwargs():
    """Fast pipeline kwargs for e2e tests."""
    return {"epochs": 2, "window_size": 5, "min_samples": 50}
