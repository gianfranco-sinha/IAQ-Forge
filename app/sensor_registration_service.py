"""SensorRegistrationService — owns field mapping proposals, confirmation, and persistence.

Extracted from app/main.py (R3) to decouple business logic from route handlers.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings
from app.schemas import configure_field_mapping

logger = logging.getLogger(__name__)

# Default TTL for pending mapping proposals (seconds)
_DEFAULT_TTL = 30 * 60  # 30 minutes


class SensorRegistrationService:
    """Manages sensor field mapping proposals and persistence."""

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL) -> None:
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._pending.items() if now - v["created_at"] > self._ttl]
        for k in expired:
            del self._pending[k]

    def propose_mapping(self, fields, sample_values=None, backend="fuzzy",
                        sensor_id=None, firmware_version=None):
        """Run the field mapper and store the proposal.  Returns (mapping_id, result)."""
        from app.field_mapper import FieldMapper
        from app.profiles import get_sensor_profile

        self._evict_expired()

        profile = get_sensor_profile()
        mapper = FieldMapper(profile)
        result = mapper.map_fields(fields, sample_values=sample_values, backend=backend)

        mapping_id = str(uuid.uuid4())
        self._pending[mapping_id] = {
            "result": result,
            "sensor_id": sensor_id,
            "firmware_version": firmware_version,
            "created_at": time.monotonic(),
        }
        return mapping_id, result

    def confirm_mapping(self, mapping_id: str, overrides: Optional[Dict[str, str]] = None) -> dict:
        """Persist a proposed mapping to model_config.yaml.

        Returns dict with status, field_mapping, sensor_id, firmware_version.
        Raises KeyError if mapping_id not found or expired.
        """
        import yaml

        self._evict_expired()

        if mapping_id not in self._pending:
            raise KeyError(f"Mapping '{mapping_id}' not found or expired")

        pending = self._pending.pop(mapping_id)
        result = pending["result"]
        sensor_id = pending.get("sensor_id")
        firmware_version = pending.get("firmware_version")

        field_mapping = {m.source_field: m.target_feature for m in result.matches}
        if overrides:
            field_mapping.update(overrides)

        # Save to model_config.yaml
        config_path = Path(__file__).resolve().parent.parent / "model_config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        cfg.setdefault("sensor", {})["field_mapping"] = field_mapping

        if sensor_id or firmware_version:
            cfg.setdefault("sensor", {}).setdefault("identity", {})
            if sensor_id:
                cfg["sensor"]["identity"]["sensor_id"] = sensor_id
            if firmware_version:
                cfg["sensor"]["identity"]["firmware_version"] = firmware_version

        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        settings.invalidate_config_cache()
        configure_field_mapping(field_mapping)
        logger.info("Field mapping saved: %s", field_mapping)

        return {
            "field_mapping": field_mapping,
            "sensor_id": sensor_id,
            "firmware_version": firmware_version,
        }

    @staticmethod
    def get_mapping() -> dict:
        """Return the active field mapping from config."""
        cfg = settings.load_model_config()
        return {
            "sensor_type": cfg.get("sensor", {}).get("type", "unknown"),
            "field_mapping": cfg.get("sensor", {}).get("field_mapping", {}),
            "identity": cfg.get("sensor", {}).get("identity", {}),
        }

    @staticmethod
    def delete_mapping() -> dict:
        """Remove the active field mapping from model_config.yaml."""
        import yaml

        config_path = Path(__file__).resolve().parent.parent / "model_config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        sensor = cfg.get("sensor", {})
        if "field_mapping" not in sensor:
            return {"status": "no_mapping", "message": "No field mapping configured"}

        del sensor["field_mapping"]

        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        settings.invalidate_config_cache()
        configure_field_mapping({})
        logger.info("Field mapping removed from config")
        return {"status": "removed"}
