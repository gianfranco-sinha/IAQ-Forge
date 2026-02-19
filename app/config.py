# ============================================================================
# File: app/config.py
# ============================================================================
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API settings
    API_TITLE: str = "iaq4j - IAQ Prediction Platform"
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model settings
    DEFAULT_MODEL: str = "mlp"
    TRAINED_MODELS_BASE: str = "trained_models"
    MLP_MODEL_PATH: str = "trained_models/mlp"
    KAN_MODEL_PATH: str = "trained_models/kan"
    CNN_MODEL_PATH: str = "trained_models/cnn"
    LSTM_MODEL_PATH: str = "trained_models/lstm"
    BNN_MODEL_PATH: str = "trained_models/bnn"
    WINDOW_SIZE: int = 10

    # YAML configuration paths
    MODEL_CONFIG_PATH: str = "model_config.yaml"
    DATABASE_CONFIG_PATH: str = "database_config.yaml"
    _model_config_cache: Optional[Dict[str, Any]] = None
    _database_config_cache: Optional[Dict[str, Any]] = None

    # InfluxDB defaults
    INFLUX_ENABLED: bool = False
    INFLUX_HOST: str = "87.106.102.14"
    INFLUX_PORT: int = 8086
    INFLUX_DATABASE: str = "home_study_room_iaq"
    INFLUX_USERNAME: str = ""
    INFLUX_PASSWORD: str = ""
    INFLUX_TIMEOUT: int = 60

    class Config:
        env_file = ".env"

    def load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        if self._model_config_cache is not None:
            return self._model_config_cache

        config_path = Path(self.MODEL_CONFIG_PATH)
        if not config_path.exists():
            default_config = self._get_default_model_config()
            self._model_config_cache = default_config
            return default_config

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self._model_config_cache = config
            return config
        except Exception as e:
            print(f"Warning: Failed to load model config from {config_path}: {e}")
            default_config = self._get_default_model_config()
            self._model_config_cache = default_config
            return default_config

    def _get_default_model_config(self) -> Dict[str, Any]:
        """Get default model configuration when YAML is not available."""
        return {
            "global": {
                "window_size": 10,
                "num_features": 4,
                "device": "cpu",
                "default_dropout": 0.2,
            },
            "mlp": {"hidden_dims": [64, 32, 16], "dropout": 0.2, "input_dim": 4},
            "kan": {"hidden_dims": [32, 16], "input_dim": 4},
            "lstm": {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.3,
                "bidirectional": True,
                "window_size": 10,
                "num_features": 4,
                "fc_layers": [64, 32],
            },
            "cnn": {
                "num_filters": [64, 128, 256],
                "kernel_sizes": [3, 3, 3],
                "dropout": 0.3,
                "window_size": 10,
                "num_features": 4,
                "fc_layers": [128, 64],
            },
            "bnn": {
                "hidden_dims": [64, 32, 16],
                "prior_sigma": 1.0,
                "kl_weight": 1.0,
                "input_dim": 6,
            },
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

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type."""
        config = self.load_model_config()
        global_config = config.get("global", {})
        model_config = config.get(model_type, {})
        merged_config = {}
        merged_config.update(global_config)
        merged_config.update(model_config)
        return merged_config

    def load_database_config(self) -> Dict[str, Any]:
        """Load database configuration from YAML file."""
        if self._database_config_cache is not None:
            return self._database_config_cache

        config_path = Path(self.DATABASE_CONFIG_PATH)
        if not config_path.exists():
            default_config = self._get_default_database_config()
            self._database_config_cache = default_config
            return default_config

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self._database_config_cache = config
            return config
        except Exception as e:
            print(f"Warning: Failed to load database config from {config_path}: {e}")
            default_config = self._get_default_database_config()
            self._database_config_cache = default_config
            return default_config

    def _get_default_database_config(self) -> Dict[str, Any]:
        """Get default database configuration when YAML is not available."""
        return {
            "influxdb": {
                "host": "87.106.102.14",
                "port": 8086,
                "database": "home_study_room_iaq",
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

    def get_database_config(self) -> Dict[str, Any]:
        """Get merged database configuration."""
        yaml_config = self.load_database_config()
        influx_config = yaml_config.get("influxdb", {})
        merged_config = {
            "version": influx_config.get("version", "1.x"),
            "client_type": influx_config.get("client_type", "influxdb"),
            "host": influx_config.get("host", self.INFLUX_HOST),
            "port": influx_config.get("port", self.INFLUX_PORT),
            "database": influx_config.get("database", self.INFLUX_DATABASE),
            "username": influx_config.get("username", self.INFLUX_USERNAME),
            "password": influx_config.get("password", self.INFLUX_PASSWORD),
            "timeout": influx_config.get("timeout", self.INFLUX_TIMEOUT),
            "enabled": influx_config.get("enabled", self.INFLUX_ENABLED),
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


settings = Settings()
