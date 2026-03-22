# ============================================================================
# File: app/config.py
# ============================================================================
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class APIConfig:
    """API server settings."""

    title: str = "iaq4j - IAQ Prediction Platform"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    environment: str = "development"


@dataclass
class ModelConfig:
    """Model YAML loading and per-model config merge."""

    config_path: str = "model_config.yaml"
    default_model: str = "mlp"
    trained_models_base: str = "trained_models"
    _cache: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        if self._cache is not None:
            return self._cache

        path = Path(self.config_path)
        if not path.exists():
            default_config = self._get_default_model_config()
            self._cache = default_config
            return default_config

        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            self._cache = config
            return config
        except Exception as e:
            print(f"Warning: Failed to load model config from {path}: {e}")
            default_config = self._get_default_model_config()
            self._cache = default_config
            return default_config

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type."""
        config = self.load_model_config()
        global_config = config.get("global", {})
        model_config = config.get(model_type, {})
        merged_config = {}
        merged_config.update(global_config)
        merged_config.update(model_config)
        return merged_config

    def invalidate_cache(self) -> None:
        """Clear cached config so next access re-reads from YAML."""
        self._cache = None

    @staticmethod
    def _get_default_model_config() -> Dict[str, Any]:
        """Get default model configuration when YAML is not available."""
        return {
            "global": {
                "window_size": 10,
                "num_features": 15,
                "device": "cpu",
                "default_dropout": 0.2,
            },
            "mlp": {"hidden_dims": [64, 32, 16], "dropout": 0.2, "input_dim": 15, "window_size": 10},
            "kan": {"hidden_dims": [64, 32], "input_dim": 15, "window_size": 10},
            "lstm": {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.3,
                "bidirectional": True,
                "window_size": 60,
                "num_features": 15,
                "fc_layers": [64, 32],
            },
            "cnn": {
                "num_filters": [64, 128, 256],
                "kernel_sizes": [3, 3, 3],
                "dropout": 0.3,
                "window_size": 30,
                "num_features": 15,
                "fc_layers": [128, 64],
            },
            "bnn": {
                "hidden_dims": [64, 32, 16],
                "prior_sigma": 1.0,
                "kl_weight": 0.001,
                "input_dim": 15,
                "window_size": 10,
            },
        }


@dataclass
class DatabaseConfig:
    """Database YAML loading and InfluxDB config merge."""

    config_path: str = "database_config.yaml"
    influx_enabled: bool = False
    influx_host: str = "localhost"
    influx_port: int = 8086
    influx_database: str = "iaq_predictions"
    influx_username: str = ""
    influx_password: str = ""
    influx_timeout: int = 60
    _cache: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def load_database_config(self) -> Dict[str, Any]:
        """Load database configuration from YAML file."""
        if self._cache is not None:
            return self._cache

        path = Path(self.config_path)
        if not path.exists():
            default_config = self._get_default_database_config()
            self._cache = default_config
            return default_config

        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            self._cache = config
            return config
        except Exception as e:
            print(f"Warning: Failed to load database config from {path}: {e}")
            default_config = self._get_default_database_config()
            self._cache = default_config
            return default_config

    def get_database_config(self) -> Dict[str, Any]:
        """Get merged database configuration."""
        yaml_config = self.load_database_config()
        influx_config = yaml_config.get("influxdb", {})
        merged_config = {
            "version": influx_config.get("version", "1.x"),
            "client_type": influx_config.get("client_type", "influxdb"),
            "host": influx_config.get("host", self.influx_host),
            "port": influx_config.get("port", self.influx_port),
            "database": influx_config.get("database", self.influx_database),
            "username": influx_config.get("username", self.influx_username),
            "password": influx_config.get("password", self.influx_password),
            "timeout": influx_config.get("timeout", self.influx_timeout),
            "enabled": influx_config.get("enabled", self.influx_enabled),
            "ssl": influx_config.get("ssl", False),
            "verify_ssl": influx_config.get("verify_ssl", True),
            "org": influx_config.get("org", ""),
            "bucket": influx_config.get("bucket", ""),
            "token": influx_config.get("token", ""),
            "batch_size": yaml_config.get("database", {}).get("batch_size", 1000),
            "max_retries": yaml_config.get("database", {}).get("max_retries", 3),
            "retry_delay": yaml_config.get("database", {}).get("retry_delay", 1),
            "retention_policy": yaml_config.get("database", {}).get(
                "retention_policy", "autogen"
            ),
            "data_retention_days": yaml_config.get("database", {}).get(
                "data_retention_days", 30
            ),
            "log_queries": yaml_config.get("logging", {}).get("log_queries", False),
            "log_performance": yaml_config.get("logging", {}).get(
                "log_performance", False
            ),
            "query_timeout_threshold": yaml_config.get("logging", {}).get(
                "query_timeout_threshold", 5
            ),
        }
        return merged_config

    def invalidate_cache(self) -> None:
        """Clear cached config so next access re-reads from YAML."""
        self._cache = None

    @staticmethod
    def _get_default_database_config() -> Dict[str, Any]:
        """Get default database configuration when YAML is not available."""
        return {
            "influxdb": {
                "host": "localhost",
                "port": 8086,
                "database": "iaq_predictions",
                "username": "",
                "password": "",
                "timeout": 60,
                "enabled": False,
                "ssl": False,
                "verify_ssl": True,
                "version": "1.x",
                "client_type": "influxdb",
                "org": "",
                "bucket": "",
                "token": "",
            },
            "database": {
                "batch_size": 1000,
                "max_retries": 3,
                "retry_delay": 1,
                "retention_policy": "autogen",
                "data_retention_days": 30,
            },
            "logging": {
                "log_queries": False,
                "log_performance": False,
                "query_timeout_threshold": 5,
            },
        }


@dataclass
class KANSidecarConfig:
    """KAN remote sidecar settings."""

    remote_url: str = ""
    remote_timeout: float = 0.5


# ---------------------------------------------------------------------------
# Settings facade — backward-compatible singleton
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    # API settings
    API_TITLE: str = "iaq4j - IAQ Prediction Platform"
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_KEY: str = ""  # empty = auth disabled (dev mode)
    ENVIRONMENT: str = "development"  # set to "production" in production deployments

    # Model settings
    DEFAULT_MODEL: str = "mlp"
    TRAINED_MODELS_BASE: str = "trained_models"
    MLP_MODEL_PATH: str = "trained_models/mlp"
    KAN_MODEL_PATH: str = "trained_models/kan"
    CNN_MODEL_PATH: str = "trained_models/cnn"
    LSTM_MODEL_PATH: str = "trained_models/lstm"
    BNN_MODEL_PATH: str = "trained_models/bnn"

    # YAML configuration paths
    MODEL_CONFIG_PATH: str = "model_config.yaml"
    DATABASE_CONFIG_PATH: str = "database_config.yaml"
    _model_config_cache: Optional[Dict[str, Any]] = None
    _database_config_cache: Optional[Dict[str, Any]] = None

    # KAN sidecar (remote inference)
    KAN_REMOTE_URL: str = ""        # e.g. "http://kan-sidecar:8001"
    KAN_REMOTE_TIMEOUT: float = 0.5  # seconds

    # InfluxDB defaults
    INFLUX_ENABLED: bool = False
    INFLUX_HOST: str = "localhost"
    INFLUX_PORT: int = 8086
    INFLUX_DATABASE: str = "iaq_predictions"
    INFLUX_USERNAME: str = ""
    INFLUX_PASSWORD: str = ""
    INFLUX_TIMEOUT: int = 60

    # Sub-config instances (populated in model_post_init)
    _api: Optional[APIConfig] = None
    _model: Optional[ModelConfig] = None
    _database: Optional[DatabaseConfig] = None
    _kan_sidecar: Optional[KANSidecarConfig] = None

    def model_post_init(self, __context: Any) -> None:
        self._api = APIConfig(
            title=self.API_TITLE,
            version=self.API_VERSION,
            host=self.HOST,
            port=self.PORT,
            api_key=self.API_KEY,
            environment=self.ENVIRONMENT,
        )
        self._model = ModelConfig(
            config_path=self.MODEL_CONFIG_PATH,
            default_model=self.DEFAULT_MODEL,
            trained_models_base=self.TRAINED_MODELS_BASE,
        )
        self._database = DatabaseConfig(
            config_path=self.DATABASE_CONFIG_PATH,
            influx_enabled=self.INFLUX_ENABLED,
            influx_host=self.INFLUX_HOST,
            influx_port=self.INFLUX_PORT,
            influx_database=self.INFLUX_DATABASE,
            influx_username=self.INFLUX_USERNAME,
            influx_password=self.INFLUX_PASSWORD,
            influx_timeout=self.INFLUX_TIMEOUT,
        )
        self._kan_sidecar = KANSidecarConfig(
            remote_url=self.KAN_REMOTE_URL,
            remote_timeout=self.KAN_REMOTE_TIMEOUT,
        )

    # -- Sub-config accessors -----------------------------------------------

    @property
    def api(self) -> APIConfig:
        return self._api

    @property
    def model(self) -> ModelConfig:
        return self._model

    @property
    def database(self) -> DatabaseConfig:
        return self._database

    @property
    def kan_sidecar(self) -> KANSidecarConfig:
        return self._kan_sidecar

    # -- Backward-compat delegation -----------------------------------------

    def load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        if self._model_config_cache is not None:
            return self._model_config_cache

        config_path = Path(self.MODEL_CONFIG_PATH)
        if not config_path.exists():
            default_config = ModelConfig._get_default_model_config()
            self._model_config_cache = default_config
            return default_config

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self._model_config_cache = config
            return config
        except Exception as e:
            print(f"Warning: Failed to load model config from {config_path}: {e}")
            default_config = ModelConfig._get_default_model_config()
            self._model_config_cache = default_config
            return default_config

    def _get_default_model_config(self) -> Dict[str, Any]:
        return ModelConfig._get_default_model_config()

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type."""
        config = self.load_model_config()
        global_config = config.get("global", {})
        model_config = config.get(model_type, {})
        merged_config = {}
        merged_config.update(global_config)
        merged_config.update(model_config)
        return merged_config

    def invalidate_config_cache(self) -> None:
        """Clear cached config so next access re-reads from YAML."""
        self._model_config_cache = None
        self._database_config_cache = None
        self._model.invalidate_cache()
        self._database.invalidate_cache()

    def load_database_config(self) -> Dict[str, Any]:
        """Load database configuration from YAML file."""
        if self._database_config_cache is not None:
            return self._database_config_cache

        config_path = Path(self.DATABASE_CONFIG_PATH)
        if not config_path.exists():
            default_config = DatabaseConfig._get_default_database_config()
            self._database_config_cache = default_config
            return default_config

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self._database_config_cache = config
            return config
        except Exception as e:
            print(f"Warning: Failed to load database config from {config_path}: {e}")
            default_config = DatabaseConfig._get_default_database_config()
            self._database_config_cache = default_config
            return default_config

    def _get_default_database_config(self) -> Dict[str, Any]:
        return DatabaseConfig._get_default_database_config()

    def get_database_config(self) -> Dict[str, Any]:
        """Get merged database configuration."""
        return self._database.get_database_config()

    # -- Domain accessors (stay on Settings) --------------------------------

    def get_sensor_identity(self) -> Dict[str, Any]:
        """Get sensor identity (sensor_id, firmware_version) from config."""
        config = self.load_model_config()
        identity = config.get("sensor", {}).get("identity", {})
        return {
            "sensor_id": identity.get("sensor_id"),
            "firmware_version": identity.get("firmware_version"),
        }

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration with defaults."""
        config = self.load_model_config()
        defaults = {
            "epochs": 200,
            "batch_size": 32,
            "learning_rate": 0.001,
            "test_size": 0.2,
            "random_state": 42,
            "min_samples": 100,
            "lr_scheduler_patience": 10,
            "lr_scheduler_factor": 0.5,
            "tensorboard_enabled": True,
            "tensorboard_log_dir": "runs",
            "tensorboard_histogram_freq": 50,
        }
        defaults.update(config.get("training", {}))
        return defaults

    def get_sensor_config(self) -> Dict[str, Any]:
        """Get sensor configuration with defaults (BME680 datasheet ranges)."""
        config = self.load_model_config()
        defaults = {
            "type": "bme680",
            "features": ["temperature", "rel_humidity", "pressure", "voc_resistance"],
            "target": "iaq",
            "valid_ranges": {
                "temperature": [-40, 85],
                "rel_humidity": [0, 100],
                "pressure": [300, 1100],
                "voc_resistance": [1000, 2000000],
                "iaq_accuracy": [2, 3],
            },
        }
        sensor_cfg = config.get("sensor", {})
        for key in ("type", "features", "target"):
            if key in sensor_cfg:
                defaults[key] = sensor_cfg[key]
        if "valid_ranges" in sensor_cfg:
            defaults["valid_ranges"].update(sensor_cfg["valid_ranges"])
        return defaults

    def get_label_studio_config(self) -> Dict[str, Any]:
        """Get Label Studio connection config from integrations.yaml."""
        import os
        import logging
        try:
            from integrations.config import get_integration_config
            ls = get_integration_config("label_studio")
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Could not load Label Studio config from integrations.yaml: %s — using defaults", e
            )
            ls = {}
        return {
            "url": ls.get("url", "http://localhost:8080").rstrip("/"),
            "api_key": os.environ.get("LABEL_STUDIO_API_KEY", ls.get("api_key", "")),
            "project_id": ls.get("project_id"),
        }

    def get_prior_variables_config(self) -> Dict[str, Any]:
        """Get user-declared prior variables from bnn.prior_variables config."""
        config = self.load_model_config()
        return config.get("bnn", {}).get("prior_variables", {})


settings = Settings()
