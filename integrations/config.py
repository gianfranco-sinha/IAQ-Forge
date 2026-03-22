"""Central integration configuration loader.

Reads ``integrations.yaml`` and merges environment variable overrides.
Each integration module reads its own section via ``load_integration_config()``.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "kan_sidecar": {
        "enabled": False,
        "url": "http://kan-sidecar:8001",
        "timeout_ms": 500,
        "healthcheck_interval_s": 15,
    },
    "mlflow": {
        "enabled": False,
        "tracking_uri": "mlruns",
        "experiment_name": "iaq4j",
        "log_models": True,
        "log_artifacts": True,
    },
    "label_studio": {
        "enabled": False,
        "url": "http://localhost:8080",
        "project_id": None,
    },
}

# Environment variable overrides (env var name → (section, key))
_ENV_OVERRIDES = {
    "KAN_SIDECAR_URL": ("kan_sidecar", "url"),
    "KAN_SIDECAR_ENABLED": ("kan_sidecar", "enabled"),
    "MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
    "MLFLOW_ENABLED": ("mlflow", "enabled"),
    "LABEL_STUDIO_URL": ("label_studio", "url"),
    "LABEL_STUDIO_API_KEY": ("label_studio", "api_key"),
    "LABEL_STUDIO_ENABLED": ("label_studio", "enabled"),
}

_cache: Optional[Dict[str, Dict[str, Any]]] = None


def load_integration_config(
    config_path: str = "integrations.yaml",
) -> Dict[str, Dict[str, Any]]:
    """Load and cache integration configuration.

    Reads ``integrations.yaml`` (falls back to defaults if missing),
    then merges environment variable overrides.

    Returns:
        Dict keyed by integration name, each containing its config dict.
    """
    global _cache
    if _cache is not None:
        return _cache

    config = {section: dict(defaults) for section, defaults in _DEFAULTS.items()}

    path = Path(config_path)
    if not path.exists():
        logger.warning("%s not found — using built-in defaults for all integrations", config_path)
    if path.exists():
        try:
            with open(path) as f:
                yaml_config = yaml.safe_load(f) or {}
            for section, values in yaml_config.items():
                if section in config and isinstance(values, dict):
                    config[section].update(values)
        except Exception as e:
            logger.warning("Failed to load %s: %s — using defaults", config_path, e)

    # Apply env var overrides
    for env_var, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Convert "true"/"false" strings to booleans
            if value.lower() in ("true", "1", "yes"):
                value = True
            elif value.lower() in ("false", "0", "no"):
                value = False
            config[section][key] = value

    _cache = config
    return config


def get_integration_config(integration: str) -> Dict[str, Any]:
    """Get config for a specific integration."""
    all_config = load_integration_config()
    return all_config.get(integration, {})


def invalidate_cache() -> None:
    """Clear the cached config (for testing)."""
    global _cache
    _cache = None
