"""Tests for PredictionService and SensorRegistrationService (R3 extraction)."""

import time
from unittest.mock import MagicMock, patch

import pytest

import app.builtin_profiles  # noqa: F401
from app.prediction_service import PredictionService
from app.sensor_registration_service import SensorRegistrationService


# ===================================================================
# PredictionService
# ===================================================================


class TestPredictionServiceInit:
    def test_starts_empty(self):
        svc = PredictionService()
        assert not svc.has_models
        assert svc.available_models == []
        assert svc.active_model == "mlp"  # settings.DEFAULT_MODEL

    def test_models_available_map_all_false_when_empty(self):
        svc = PredictionService()
        m = svc.models_available_map()
        assert all(v is False for v in m.values())
        assert "mlp" in m

    def test_list_models_empty(self):
        svc = PredictionService()
        result = svc.list_models()
        assert result["available"] == {}


class TestPredictionServiceSelectModel:
    def test_select_unknown_raises(self):
        svc = PredictionService()
        with pytest.raises(KeyError, match="not loaded"):
            svc.select_model("nonexistent")

    def test_select_loaded_model(self):
        svc = PredictionService()
        # Inject a fake predictor
        mock_predictor = MagicMock()
        mock_predictor.window_size = 10
        mock_predictor.config = {}
        svc._predictors["mlp"] = mock_predictor
        svc._engines["mlp"] = MagicMock()

        result = svc.select_model("mlp")
        assert result == "mlp"
        assert svc.active_model == "mlp"


class TestPredictionServiceReset:
    def test_reset_model_unknown_raises(self):
        svc = PredictionService()
        with pytest.raises(KeyError, match="not found"):
            svc.reset_model("nonexistent")

    def test_reset_all_empty(self):
        svc = PredictionService()
        result = svc.reset_all()
        assert result["status"] == "all buffers reset"
        assert result["models"] == []

    def test_reset_model_clears_buffer(self):
        svc = PredictionService()
        mock_predictor = MagicMock()
        mock_predictor.buffer = [1, 2, 3]
        mock_predictor.window_size = 10
        svc._predictors["mlp"] = mock_predictor

        result = svc.reset_model("mlp")
        assert result["status"] == "buffer reset"
        assert mock_predictor.buffer == []


class TestPredictionServiceStatistics:
    def test_get_statistics_no_model_raises(self):
        svc = PredictionService()
        with pytest.raises(KeyError):
            svc.get_statistics()

    def test_get_statistics_specific_model(self):
        svc = PredictionService()
        mock_engine = MagicMock()
        mock_engine.get_statistics.return_value = {"count": 42}
        svc._engines["mlp"] = mock_engine

        result = svc.get_statistics("mlp")
        assert result == {"count": 42}


# ===================================================================
# SensorRegistrationService
# ===================================================================


class TestSensorRegistrationServiceTTL:
    def test_proposal_expires_after_ttl(self):
        svc = SensorRegistrationService(ttl_seconds=0)  # instant expiry

        # Inject a fake pending entry manually
        svc._pending["test-id"] = {
            "result": MagicMock(),
            "sensor_id": None,
            "firmware_version": None,
            "created_at": time.monotonic() - 1,  # already expired
        }

        svc._evict_expired()
        assert "test-id" not in svc._pending

    def test_confirm_expired_raises(self):
        svc = SensorRegistrationService(ttl_seconds=0)

        svc._pending["test-id"] = {
            "result": MagicMock(),
            "sensor_id": None,
            "firmware_version": None,
            "created_at": time.monotonic() - 1,
        }

        with pytest.raises(KeyError, match="not found or expired"):
            svc.confirm_mapping("test-id")

    def test_confirm_nonexistent_raises(self):
        svc = SensorRegistrationService()
        with pytest.raises(KeyError, match="not found or expired"):
            svc.confirm_mapping("does-not-exist")


class TestSensorRegistrationServiceGetMapping:
    def test_get_mapping_returns_dict(self):
        result = SensorRegistrationService.get_mapping()
        assert "sensor_type" in result
        assert "field_mapping" in result
        assert "identity" in result
