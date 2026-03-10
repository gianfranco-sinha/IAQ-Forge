"""Tier 2: Settings config loading, caching, per-model merge, training defaults."""
import pytest

from app.config import Settings, settings


class TestLoadModelConfig:
    def test_loads_yaml(self, monkeypatch):
        """Returns dict when YAML file exists (using real config)."""
        monkeypatch.setattr(settings, "_model_config_cache", None)
        result = settings.load_model_config()
        assert isinstance(result, dict)

    def test_caches_result(self, monkeypatch):
        """Second call returns same object (identity check)."""
        monkeypatch.setattr(settings, "_model_config_cache", None)
        first = settings.load_model_config()
        second = settings.load_model_config()
        assert first is second

    def test_cache_invalidation(self, monkeypatch):
        """Setting _model_config_cache = None forces reload."""
        monkeypatch.setattr(settings, "_model_config_cache", None)
        first = settings.load_model_config()
        monkeypatch.setattr(settings, "_model_config_cache", None)
        second = settings.load_model_config()
        # Same content but different object after cache invalidation
        assert first == second

    def test_missing_yaml_returns_defaults(self, monkeypatch):
        """Falls back to _get_default_model_config() when YAML missing."""
        monkeypatch.setattr(settings, "_model_config_cache", None)
        monkeypatch.setattr(settings, "MODEL_CONFIG_PATH", "/nonexistent/path.yaml")
        result = settings.load_model_config()
        default = settings._get_default_model_config()
        assert result == default
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_default_config_has_all_models(self):
        default = settings._get_default_model_config()
        for key in ("global", "mlp", "kan", "lstm", "cnn", "bnn"):
            assert key in default

    def test_default_config_window_sizes(self):
        default = settings._get_default_model_config()
        assert default["mlp"]["window_size"] == 10
        assert default["lstm"]["window_size"] == 60
        assert default["cnn"]["window_size"] == 30

    def test_default_mlp_has_hidden_dims(self):
        default = settings._get_default_model_config()
        assert "hidden_dims" in default["mlp"]

    def test_default_bnn_has_prior_sigma(self):
        default = settings._get_default_model_config()
        assert "prior_sigma" in default["bnn"]


class TestGetModelConfig:
    def _with_config(self, monkeypatch, cfg):
        monkeypatch.setattr(settings, "_model_config_cache", cfg)

    def test_global_merged(self, monkeypatch):
        cfg = {"global": {"device": "cpu", "window_size": 10}, "mlp": {"dropout": 0.2}}
        self._with_config(monkeypatch, cfg)
        result = settings.get_model_config("mlp")
        assert result["device"] == "cpu"
        assert result["dropout"] == 0.2
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_model_overrides_global(self, monkeypatch):
        cfg = {"global": {"window_size": 10}, "mlp": {"window_size": 20}}
        self._with_config(monkeypatch, cfg)
        result = settings.get_model_config("mlp")
        assert result["window_size"] == 20
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_mlp_defaults(self):
        result = settings.get_model_config("mlp")
        assert "window_size" in result

    def test_lstm_defaults(self):
        default = settings._get_default_model_config()
        assert "hidden_size" in default["lstm"]
        assert "num_layers" in default["lstm"]
        assert "bidirectional" in default["lstm"]

    def test_cnn_defaults(self):
        default = settings._get_default_model_config()
        assert "num_filters" in default["cnn"]
        assert "kernel_sizes" in default["cnn"]

    def test_bnn_defaults(self):
        default = settings._get_default_model_config()
        assert "prior_sigma" in default["bnn"]
        assert "kl_weight" in default["bnn"]

    def test_kan_defaults(self):
        default = settings._get_default_model_config()
        assert "hidden_dims" in default["kan"]

    def test_unknown_model_type(self, monkeypatch):
        """Returns only global keys for unknown model type (no crash)."""
        cfg = {"global": {"window_size": 10}}
        self._with_config(monkeypatch, cfg)
        result = settings.get_model_config("nonexistent")
        assert result == {"window_size": 10}
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_each_model_has_window_size(self, monkeypatch):
        default = settings._get_default_model_config()
        self._with_config(monkeypatch, default)
        for mt in ("mlp", "kan", "lstm", "cnn", "bnn"):
            cfg = settings.get_model_config(mt)
            assert "window_size" in cfg, f"{mt} missing window_size"
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_window_size_per_model_values(self, monkeypatch):
        default = settings._get_default_model_config()
        self._with_config(monkeypatch, default)
        assert settings.get_model_config("mlp")["window_size"] == 10
        assert settings.get_model_config("kan")["window_size"] == 10
        assert settings.get_model_config("lstm")["window_size"] == 60
        assert settings.get_model_config("cnn")["window_size"] == 30
        assert settings.get_model_config("bnn")["window_size"] == 10
        monkeypatch.setattr(settings, "_model_config_cache", None)


class TestGetTrainingConfig:
    def test_default_epochs(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {})
        result = settings.get_training_config()
        assert result["epochs"] == 200
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_default_batch_size(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {})
        result = settings.get_training_config()
        assert result["batch_size"] == 32
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_default_learning_rate(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {})
        result = settings.get_training_config()
        assert result["learning_rate"] == 0.001
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_default_test_size(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {})
        result = settings.get_training_config()
        assert result["test_size"] == 0.2
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_yaml_overrides_default(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {
            "training": {"epochs": 50, "batch_size": 64}
        })
        result = settings.get_training_config()
        assert result["epochs"] == 50
        assert result["batch_size"] == 64
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_checkpoint_freq_present(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {
            "training": {"checkpoint_freq": 10}
        })
        result = settings.get_training_config()
        assert result["checkpoint_freq"] == 10
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_all_expected_keys(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {})
        result = settings.get_training_config()
        for key in ("epochs", "batch_size", "learning_rate", "test_size",
                     "random_state", "min_samples", "lr_scheduler_patience",
                     "lr_scheduler_factor"):
            assert key in result, f"Missing key: {key}"
        monkeypatch.setattr(settings, "_model_config_cache", None)
