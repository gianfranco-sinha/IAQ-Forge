"""Tier 2: Model construction, forward pass shapes, IAQPredictor buffering & load."""
import json
import re

import numpy as np
import pytest
import torch
import torch.nn as nn

import app.builtin_profiles  # noqa: F401
from app.config import settings
from app.exceptions import IAQError
from app.models import (
    BNNRegressor,
    BayesianLinear,
    CNNRegressor,
    LSTMRegressor,
    MLPRegressor,
    build_model,
    IAQPredictor,
    MODEL_REGISTRY,
    _KAN_AVAILABLE,
)

if _KAN_AVAILABLE:
    from app.models import KANRegressor

_skip_no_kan = pytest.mark.skipif(not _KAN_AVAILABLE, reason="KAN requires Python 3.9")


# ── helpers ──────────────────────────────────────────────────────────────

WINDOW_SIZE = 5
NUM_FEATURES = 15
INPUT_DIM = WINDOW_SIZE * NUM_FEATURES
ALL_TYPES = list(MODEL_REGISTRY.keys())  # mlp, lstm, cnn, kan, bnn


def _make_input(batch=4):
    """Random input tensor (batch, window*features)."""
    return torch.randn(batch, INPUT_DIM)


# ── TestBuildModel ───────────────────────────────────────────────────────

class TestBuildModel:
    def test_build_mlp(self):
        model = build_model("mlp", WINDOW_SIZE, NUM_FEATURES)
        assert isinstance(model, MLPRegressor)

    @_skip_no_kan
    def test_build_kan(self):
        model = build_model("kan", WINDOW_SIZE, NUM_FEATURES)
        assert isinstance(model, KANRegressor)

    def test_build_lstm(self):
        model = build_model("lstm", WINDOW_SIZE, NUM_FEATURES)
        assert isinstance(model, LSTMRegressor)

    def test_build_cnn(self):
        model = build_model("cnn", WINDOW_SIZE, NUM_FEATURES)
        assert isinstance(model, CNNRegressor)

    def test_build_bnn(self):
        model = build_model("bnn", WINDOW_SIZE, NUM_FEATURES)
        assert isinstance(model, BNNRegressor)

    def test_build_unknown_raises(self):
        with pytest.raises(IAQError, match="Unknown model type"):
            build_model("xgboost", WINDOW_SIZE, NUM_FEATURES)

    def test_build_uses_config_window_size(self, monkeypatch):
        """Per-model window_size from config overrides the arg."""
        cfg = settings._get_default_model_config()
        cfg["mlp"]["window_size"] = 7
        cfg["mlp"]["num_features"] = NUM_FEATURES
        monkeypatch.setattr(settings, "_model_config_cache", cfg)
        # Explicit args override config
        model = build_model("mlp", window_size=99, num_features=NUM_FEATURES)
        first_layer = model.network[0]
        assert first_layer.in_features == 99 * NUM_FEATURES
        # Config used as fallback when args are None
        model2 = build_model("mlp")
        first_layer2 = model2.network[0]
        assert first_layer2.in_features == 7 * NUM_FEATURES
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_build_bnn_no_batch_norm_by_default(self):
        model = build_model("bnn", WINDOW_SIZE, NUM_FEATURES)
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == 0


# ── TestForwardPass ──────────────────────────────────────────────────────

class TestForwardPass:
    @pytest.mark.parametrize("model_type", ALL_TYPES)
    def test_output_shape(self, model_type, monkeypatch):
        """Input (batch, window*features) → output (batch, 1)."""
        # Override config to use our test dimensions
        cfg = settings._get_default_model_config()
        for mt in ALL_TYPES:
            cfg.setdefault(mt, {})
            cfg[mt]["window_size"] = WINDOW_SIZE
            cfg[mt]["num_features"] = NUM_FEATURES
        monkeypatch.setattr(settings, "_model_config_cache", cfg)

        model = build_model(model_type, WINDOW_SIZE, NUM_FEATURES)
        model.eval()
        x = _make_input(batch=4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1)
        monkeypatch.setattr(settings, "_model_config_cache", None)

    @pytest.mark.parametrize("model_type", ALL_TYPES)
    def test_single_sample(self, model_type, monkeypatch):
        """Works with batch=1 in eval mode."""
        cfg = settings._get_default_model_config()
        for mt in ALL_TYPES:
            cfg.setdefault(mt, {})
            cfg[mt]["window_size"] = WINDOW_SIZE
            cfg[mt]["num_features"] = NUM_FEATURES
        monkeypatch.setattr(settings, "_model_config_cache", cfg)

        model = build_model(model_type, WINDOW_SIZE, NUM_FEATURES)
        model.eval()
        x = _make_input(batch=1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)
        monkeypatch.setattr(settings, "_model_config_cache", None)


# ── TestBNNSpecific ──────────────────────────────────────────────────────

class TestBNNSpecific:
    def _make_bnn(self):
        return BNNRegressor(input_dim=INPUT_DIM, hidden_dims=[16, 8])

    def test_kl_loss_positive(self):
        model = self._make_bnn()
        x = _make_input(batch=2)
        model(x)
        assert model.kl_loss().item() > 0

    def test_kl_loss_zero_before_forward(self):
        model = self._make_bnn()
        # Before any forward pass, individual BayesianLinear._kl is 0
        for m in model.modules():
            if isinstance(m, BayesianLinear):
                assert m._kl.item() == 0.0

    def test_bayesian_linear_weight_sampling(self):
        """Two forward passes produce different outputs (stochastic weights)."""
        model = self._make_bnn()
        model.eval()
        x = _make_input(batch=1)
        with torch.no_grad():
            out1 = model(x).clone()
            out2 = model(x).clone()
        # Extremely unlikely to be identical due to weight sampling
        assert not torch.allclose(out1, out2)

    def test_use_batch_norm_true(self):
        model = BNNRegressor(input_dim=INPUT_DIM, hidden_dims=[16, 8], use_batch_norm=True)
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == 2  # one per hidden layer


# ── TestKANSpecific ──────────────────────────────────────────────────────

@_skip_no_kan
class TestKANSpecific:
    def test_regularization_loss_callable(self):
        model = KANRegressor(input_dim=INPUT_DIM, hidden_dims=[16, 8])
        x = _make_input(batch=2)
        model(x)
        reg = model.regularization_loss()
        assert reg.dim() == 0  # scalar

    def test_grid_size_from_config(self):
        m1 = KANRegressor(input_dim=INPUT_DIM, hidden_dims=[16], grid_size=5)
        m2 = KANRegressor(input_dim=INPUT_DIM, hidden_dims=[16], grid_size=10)
        # Different grid_size → different parameter counts
        p1 = sum(p.numel() for p in m1.parameters())
        p2 = sum(p.numel() for p in m2.parameters())
        assert p1 != p2

    def test_different_hidden_dims(self):
        m1 = KANRegressor(input_dim=INPUT_DIM, hidden_dims=[64, 32])
        m2 = KANRegressor(input_dim=INPUT_DIM, hidden_dims=[32, 16])
        p1 = sum(p.numel() for p in m1.parameters())
        p2 = sum(p.numel() for p in m2.parameters())
        assert p1 != p2


# ── TestIAQPredictor ─────────────────────────────────────────────────────

class TestIAQPredictor:
    def test_init_sets_defaults(self):
        p = IAQPredictor(model_type="mlp", window_size=10)
        assert p.model is None
        assert p.buffer == []
        assert p.window_size == 10

    def test_load_model_success(self, model_artifact_dir, monkeypatch):
        """Round-trip: build → save → load returns True."""
        p = IAQPredictor(model_type="mlp", window_size=5)
        assert p.load_model(str(model_artifact_dir)) is True
        assert p.model is not None

    def test_load_model_missing_config(self, tmp_path):
        """Returns False when config.json is missing."""
        (tmp_path / "model.pt").touch()
        p = IAQPredictor(model_type="mlp")
        assert p.load_model(str(tmp_path)) is False

    def test_load_model_missing_weights(self, tmp_path):
        """Returns False when model.pt is missing."""
        (tmp_path / "config.json").write_text(json.dumps({"window_size": 5}))
        p = IAQPredictor(model_type="mlp")
        assert p.load_model(str(tmp_path)) is False

    def test_load_model_sets_window_size(self, model_artifact_dir, monkeypatch):
        p = IAQPredictor(model_type="mlp", window_size=99)
        p.load_model(str(model_artifact_dir))
        # window_size should be read from config.json (saved as 5)
        assert p.window_size == 5

    def test_load_model_sets_baselines(self, model_artifact_dir, monkeypatch):
        # Patch config.json to include baselines
        config_path = model_artifact_dir / "config.json"
        cfg = json.loads(config_path.read_text())
        cfg["baselines"] = {"voc_resistance": 100000.0}
        config_path.write_text(json.dumps(cfg))

        p = IAQPredictor(model_type="mlp", window_size=5)
        p.load_model(str(model_artifact_dir))
        assert p._baselines == {"voc_resistance": 100000.0}

    def test_predict_before_load(self):
        p = IAQPredictor(model_type="mlp")
        result = p.predict(readings={"temperature": 22})
        assert result["status"] == "error"

    def test_predict_buffering(self, model_artifact_dir, sample_reading, monkeypatch):
        p = IAQPredictor(model_type="mlp", window_size=5)
        p.load_model(str(model_artifact_dir))
        result = p.predict(readings=sample_reading)
        assert result["status"] == "buffering"
        assert result["buffer_size"] == 1

    def test_predict_ready(self, model_artifact_dir, sample_reading, monkeypatch):
        p = IAQPredictor(model_type="mlp", window_size=5)
        p.load_model(str(model_artifact_dir))
        # Fill the buffer
        for _ in range(5):
            result = p.predict(readings=sample_reading)
        assert result["status"] == "ready"
        assert isinstance(result["iaq"], float)

    def test_predict_buffer_slides(self, model_artifact_dir, sample_reading, monkeypatch):
        p = IAQPredictor(model_type="mlp", window_size=5)
        p.load_model(str(model_artifact_dir))
        for _ in range(10):
            p.predict(readings=sample_reading)
        assert len(p.buffer) == 5  # never exceeds window_size

    def test_reset_buffer(self):
        p = IAQPredictor(model_type="mlp")
        p.buffer = [1, 2, 3]
        p.reset_buffer()
        assert p.buffer == []

    def test_predict_with_mc_samples(self, model_artifact_dir, sample_reading, monkeypatch):
        p = IAQPredictor(model_type="mlp", window_size=5)
        p.load_model(str(model_artifact_dir))
        for _ in range(5):
            result = p.predict(readings=sample_reading, n_mc_samples=10)
        assert result["status"] == "ready"
        assert "uncertainty" in result["predicted"]
        assert "std" in result["predicted"]["uncertainty"]

    def test_schema_fingerprint_mismatch_warns(
        self, model_artifact_dir, monkeypatch, caplog
    ):
        config_path = model_artifact_dir / "config.json"
        cfg = json.loads(config_path.read_text())
        cfg["schema_fingerprint"] = "000000000000"  # fake mismatch
        config_path.write_text(json.dumps(cfg))

        import logging
        with caplog.at_level(logging.WARNING):
            p = IAQPredictor(model_type="mlp", window_size=5)
            loaded = p.load_model(str(model_artifact_dir))
        assert loaded is True  # still loads despite mismatch
        assert "Schema fingerprint mismatch" in caplog.text

    def test_predict_result_structure(self, model_artifact_dir, sample_reading, monkeypatch):
        p = IAQPredictor(model_type="mlp", window_size=5)
        p.load_model(str(model_artifact_dir))
        for _ in range(5):
            result = p.predict(readings=sample_reading)
        expected_keys = {"iaq", "category", "status", "model_type", "raw_inputs",
                         "buffer_size", "required", "observation", "predicted", "inference"}
        assert expected_keys.issubset(result.keys())
