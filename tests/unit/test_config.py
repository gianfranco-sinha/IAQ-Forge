"""Tier 2: Settings config loading, caching, per-model merge, training defaults."""
import pytest

from app.config import (
    APIConfig,
    DatabaseConfig,
    KANSidecarConfig,
    ModelConfig,
    Settings,
    settings,
)


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


# ---------------------------------------------------------------------------
# Sub-config access tests (Config Decomposition R2)
# ---------------------------------------------------------------------------


class TestSubConfigAccess:
    """Verify sub-config instances are accessible and correctly populated."""

    def test_api_config_exists(self):
        assert settings.api is not None
        assert isinstance(settings.api, APIConfig)

    def test_api_config_fields(self):
        assert settings.api.title == settings.API_TITLE
        assert settings.api.version == settings.API_VERSION
        assert settings.api.host == settings.HOST
        assert settings.api.port == settings.PORT
        assert settings.api.api_key == settings.API_KEY
        assert settings.api.environment == settings.ENVIRONMENT

    def test_model_config_exists(self):
        assert settings.model is not None
        assert isinstance(settings.model, ModelConfig)

    def test_model_config_fields(self):
        assert settings.model.config_path == settings.MODEL_CONFIG_PATH
        assert settings.model.default_model == settings.DEFAULT_MODEL
        assert settings.model.trained_models_base == settings.TRAINED_MODELS_BASE

    def test_database_config_exists(self):
        assert settings.database is not None
        assert isinstance(settings.database, DatabaseConfig)

    def test_database_config_fields(self):
        assert settings.database.config_path == settings.DATABASE_CONFIG_PATH
        assert settings.database.influx_enabled == settings.INFLUX_ENABLED
        assert settings.database.influx_host == settings.INFLUX_HOST
        assert settings.database.influx_port == settings.INFLUX_PORT

    def test_kan_sidecar_config_exists(self):
        assert settings.kan_sidecar is not None
        assert isinstance(settings.kan_sidecar, KANSidecarConfig)

    def test_kan_sidecar_config_fields(self):
        assert settings.kan_sidecar.remote_url == settings.KAN_REMOTE_URL
        assert settings.kan_sidecar.remote_timeout == settings.KAN_REMOTE_TIMEOUT


class TestModelSubConfigMethods:
    """ModelConfig owns its own load/get/invalidate methods."""

    def test_load_model_config(self):
        result = settings.model.load_model_config()
        assert isinstance(result, dict)

    def test_get_model_config(self):
        result = settings.model.get_model_config("mlp")
        assert "window_size" in result

    def test_invalidate_cache(self):
        settings.model.load_model_config()
        settings.model.invalidate_cache()
        assert settings.model._cache is None

    def test_default_model_config(self):
        default = ModelConfig._get_default_model_config()
        for key in ("global", "mlp", "kan", "lstm", "cnn", "bnn"):
            assert key in default


class TestDatabaseSubConfigMethods:
    """DatabaseConfig owns its own load/get/invalidate methods."""

    def test_load_database_config(self):
        result = settings.database.load_database_config()
        assert isinstance(result, dict)

    def test_get_database_config(self):
        result = settings.database.get_database_config()
        assert "host" in result
        assert "port" in result

    def test_invalidate_cache(self):
        settings.database.load_database_config()
        settings.database.invalidate_cache()
        assert settings.database._cache is None

    def test_default_database_config(self):
        default = DatabaseConfig._get_default_database_config()
        assert "influxdb" in default
        assert "database" in default
        assert "logging" in default


class TestBackwardCompatDelegation:
    """Settings facade still delegates correctly to sub-configs."""

    def test_invalidate_clears_both(self):
        # Load to populate caches
        settings.load_model_config()
        settings.load_database_config()
        settings.invalidate_config_cache()
        assert settings._model_config_cache is None
        assert settings._database_config_cache is None
        assert settings.model._cache is None
        assert settings.database._cache is None

    def test_load_model_config_delegates(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", None)
        result = settings.load_model_config()
        assert isinstance(result, dict)

    def test_get_model_config_delegates(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", None)
        result = settings.get_model_config("mlp")
        assert "window_size" in result

    def test_get_database_config_delegates(self):
        result = settings.get_database_config()
        assert "host" in result
