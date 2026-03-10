"""Tests for app/schemas.py — Pydantic models for API request/response."""
import pytest
from pydantic import ValidationError


# ── SensorReading ──────────────────────────────────────────────────────────


class TestSensorReading:
    def _make(self, monkeypatch, **kwargs):
        """Create a SensorReading with settings mocked to return empty config."""
        from app.config import Settings
        monkeypatch.setattr(Settings, "load_model_config", lambda self: {})
        from app.schemas import SensorReading
        return SensorReading(**kwargs)

    def test_new_format(self, monkeypatch):
        r = self._make(
            monkeypatch,
            readings={"temperature": 25.0, "rel_humidity": 50.0},
            timestamp="2026-01-15T00:00:00Z",
        )
        assert r.get_readings() == {"temperature": 25.0, "rel_humidity": 50.0}

    def test_legacy_format(self, monkeypatch):
        r = self._make(
            monkeypatch,
            temperature=25.0,
            rel_humidity=50.0,
            pressure=1013.0,
            voc_resistance=50000.0,
            timestamp="2026-01-15T00:00:00Z",
        )
        readings = r.get_readings()
        assert readings["temperature"] == 25.0
        assert readings["voc_resistance"] == 50000.0

    def test_readings_wins_over_legacy(self, monkeypatch):
        r = self._make(
            monkeypatch,
            readings={"temperature": 99.0},
            temperature=25.0,
            timestamp="2026-01-15T00:00:00Z",
        )
        assert r.get_readings()["temperature"] == 99.0

    def test_timestamp_required(self, monkeypatch):
        from app.config import Settings
        monkeypatch.setattr(Settings, "load_model_config", lambda self: {})
        from app.schemas import SensorReading
        with pytest.raises(ValidationError):
            SensorReading(readings={"temperature": 25.0})

    def test_optional_fields_default_none(self, monkeypatch):
        r = self._make(
            monkeypatch,
            readings={"temperature": 25.0},
            timestamp="2026-01-15T00:00:00Z",
        )
        assert r.sensor_id is None
        assert r.firmware_version is None
        assert r.sequence_number is None
        assert r.iaq_actual is None
        assert r.prior_variables is None

    def test_get_readings_empty(self, monkeypatch):
        r = self._make(monkeypatch, timestamp="2026-01-15T00:00:00Z")
        assert r.get_readings() == {}

    def test_field_mapping(self, monkeypatch):
        from app.config import Settings
        monkeypatch.setattr(
            Settings, "load_model_config",
            lambda self: {"sensor": {"field_mapping": {"gas_resistance": "voc_resistance"}}},
        )
        from app.schemas import SensorReading
        r = SensorReading(
            readings={"gas_resistance": 50000.0, "temperature": 25.0},
            timestamp="2026-01-15T00:00:00Z",
        )
        readings = r.get_readings()
        assert "voc_resistance" in readings
        assert readings["voc_resistance"] == 50000.0

    def test_with_all_optional_fields(self, monkeypatch):
        r = self._make(
            monkeypatch,
            readings={"temperature": 25.0},
            timestamp="2026-01-15T00:00:00Z",
            sensor_id="sensor-001",
            firmware_version="1.2.3",
            sequence_number=42,
            iaq_actual=75.0,
            prior_variables={"presence": 1.0},
        )
        assert r.sensor_id == "sensor-001"
        assert r.sequence_number == 42
        assert r.iaq_actual == 75.0
        assert r.prior_variables == {"presence": 1.0}


# ── IAQResponse ────────────────────────────────────────────────────────────


class TestIAQResponse:
    def test_minimal(self):
        from app.schemas import IAQResponse
        r = IAQResponse(status="ok")
        assert r.status == "ok"
        assert r.iaq is None
        assert r.category is None

    def test_full(self):
        from app.schemas import IAQResponse
        r = IAQResponse(
            status="ok",
            iaq=75.0,
            category="Good",
            model_type="mlp",
            buffer_size=10,
            required=10,
        )
        assert r.iaq == 75.0
        assert r.model_type == "mlp"

    def test_model_config_namespace(self):
        """Ensure ConfigDict(protected_namespaces=()) prevents Pydantic warning."""
        from app.schemas import IAQResponse
        # Just constructing with model_type should not raise
        r = IAQResponse(status="ok", model_type="mlp")
        assert r.model_type == "mlp"


# ── Other schemas ──────────────────────────────────────────────────────────


class TestOtherSchemas:
    def test_observation(self):
        from app.schemas import Observation
        o = Observation(
            sensor_type="bme680",
            readings={"temperature": 25.0},
            timestamp="2026-01-15T00:00:00Z",
        )
        assert o.sensor_type == "bme680"

    def test_predicted(self):
        from app.schemas import Predicted
        p = Predicted(mean=75.0, category="Good")
        assert p.mean == 75.0
        assert p.iaq_standard == "bsec"

    def test_uncertainty_estimate(self):
        from app.schemas import UncertaintyEstimate
        u = UncertaintyEstimate(std=5.0, ci_lower=65.0, ci_upper=85.0, method="mc_dropout")
        assert u.method == "mc_dropout"

    def test_prior(self):
        from app.schemas import Prior
        p = Prior(mean=70.0, std=10.0, source="history_window", n_observations=50)
        assert p.n_observations == 50

    def test_health_response(self):
        from app.schemas import HealthResponse
        h = HealthResponse(
            status="healthy",
            models_available={"mlp": True, "kan": False},
            active_model="mlp",
        )
        assert h.status == "healthy"

    def test_model_info(self):
        from app.schemas import ModelInfo
        m = ModelInfo(
            model_type="mlp", window_size=10, loaded=True, config={"layers": [64, 32]},
        )
        assert m.loaded is True

    def test_model_selection(self):
        from app.schemas import ModelSelection
        s = ModelSelection(model_type="kan")
        assert s.model_type == "kan"
