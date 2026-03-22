"""Tests for RemoteKANPredictor — buffering, scaling, and HTTP proxying."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import app.builtin_profiles  # noqa: F401
from app.remote_predictor import RemoteKANPredictor


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.Client for sidecar HTTP calls."""
    with patch("app.remote_predictor.httpx.Client") as MockClient:
        client_instance = MagicMock()
        MockClient.return_value = client_instance
        yield client_instance


@pytest.fixture
def kan_model_dir(tmp_path):
    """Create a minimal KAN model directory with config and scalers."""
    model_dir = tmp_path / "kan"
    model_dir.mkdir()

    config = {
        "model_type": "kan",
        "window_size": 3,
        "num_features": 15,
        "model_params": {"input_dim": 45, "hidden_dims": [32, 16]},
        "baselines": {"voc_resistance": 150000},
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    # Create dummy scalers
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    import joblib

    feature_scaler = StandardScaler()
    feature_scaler.fit(np.random.randn(10, 45))
    joblib.dump(feature_scaler, model_dir / "feature_scaler.pkl")

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(np.array([[0], [500]]))
    joblib.dump(target_scaler, model_dir / "target_scaler.pkl")

    return model_dir


class TestRemoteKANPredictor:
    def test_load_model_success(self, kan_model_dir, mock_httpx_client):
        """load_model loads config/scalers locally and health-checks sidecar."""
        health_resp = MagicMock()
        health_resp.json.return_value = {"status": "ok", "model_loaded": True}
        health_resp.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = health_resp

        predictor = RemoteKANPredictor(
            base_url="http://localhost:8001", window_size=3
        )
        # Replace the client with our mock
        predictor._client = mock_httpx_client

        result = predictor.load_model(str(kan_model_dir))

        assert result is True
        assert predictor.config is not None
        assert predictor.feature_scaler is not None
        assert predictor.target_scaler is not None
        assert predictor.window_size == 3
        mock_httpx_client.get.assert_called_once_with("/health")

    def test_load_model_triggers_reload(self, kan_model_dir, mock_httpx_client):
        """When sidecar reports model not loaded, /load is called."""
        health_resp = MagicMock()
        health_resp.json.return_value = {"status": "ok", "model_loaded": False}
        health_resp.raise_for_status = MagicMock()

        load_resp = MagicMock()
        load_resp.raise_for_status = MagicMock()

        mock_httpx_client.get.return_value = health_resp
        mock_httpx_client.post.return_value = load_resp

        predictor = RemoteKANPredictor(
            base_url="http://localhost:8001", window_size=3
        )
        predictor._client = mock_httpx_client

        result = predictor.load_model(str(kan_model_dir))
        assert result is True
        mock_httpx_client.post.assert_called_once_with("/load")

    def test_load_model_sidecar_unreachable(self, kan_model_dir, mock_httpx_client):
        """load_model returns False when sidecar is unreachable."""
        import httpx

        mock_httpx_client.get.side_effect = httpx.ConnectError("connection refused")

        predictor = RemoteKANPredictor(
            base_url="http://localhost:8001", window_size=3
        )
        predictor._client = mock_httpx_client

        result = predictor.load_model(str(kan_model_dir))
        assert result is False

    def test_buffering(self, kan_model_dir, mock_httpx_client):
        """Predictions return buffering status until window is filled."""
        health_resp = MagicMock()
        health_resp.json.return_value = {"status": "ok", "model_loaded": True}
        health_resp.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = health_resp

        predictor = RemoteKANPredictor(
            base_url="http://localhost:8001", window_size=3
        )
        predictor._client = mock_httpx_client
        predictor.load_model(str(kan_model_dir))

        # First reading — should buffer
        readings = {
            "temperature": 22.0,
            "rel_humidity": 50.0,
            "pressure": 1013.0,
            "voc_resistance": 150000.0,
        }
        result = predictor.predict(readings)
        assert result["status"] == "buffering"
        assert result["buffer_size"] == 1

    def test_predict_proxies_forward_pass(self, kan_model_dir, mock_httpx_client):
        """After filling buffer, predict sends tensor to /forward."""
        health_resp = MagicMock()
        health_resp.json.return_value = {"status": "ok", "model_loaded": True}
        health_resp.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = health_resp

        forward_resp = MagicMock()
        forward_resp.json.return_value = {"prediction": [[0.45]]}
        forward_resp.raise_for_status = MagicMock()
        mock_httpx_client.post.return_value = forward_resp

        predictor = RemoteKANPredictor(
            base_url="http://localhost:8001", window_size=3
        )
        predictor._client = mock_httpx_client
        predictor.load_model(str(kan_model_dir))

        readings = {
            "temperature": 22.0,
            "rel_humidity": 50.0,
            "pressure": 1013.0,
            "voc_resistance": 150000.0,
        }

        # Fill buffer
        for _ in range(2):
            predictor.predict(readings)

        # Third reading should trigger actual prediction
        result = predictor.predict(readings)
        assert result["status"] == "ready"
        assert result["iaq"] is not None
        assert result["model_type"] == "kan"
        assert "predicted" in result

        # Verify /forward was called
        calls = mock_httpx_client.post.call_args_list
        forward_calls = [c for c in calls if c[0][0] == "/forward"]
        assert len(forward_calls) == 1

    def test_predict_sidecar_error_returns_error_status(
        self, kan_model_dir, mock_httpx_client
    ):
        """When sidecar returns HTTP error, predict returns error status."""
        import httpx

        health_resp = MagicMock()
        health_resp.json.return_value = {"status": "ok", "model_loaded": True}
        health_resp.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = health_resp

        predictor = RemoteKANPredictor(
            base_url="http://localhost:8001", window_size=3
        )
        predictor._client = mock_httpx_client
        predictor.load_model(str(kan_model_dir))

        readings = {
            "temperature": 22.0,
            "rel_humidity": 50.0,
            "pressure": 1013.0,
            "voc_resistance": 150000.0,
        }

        # Fill buffer (window_size=3 from config)
        predictor.predict(readings)
        predictor.predict(readings)

        # Make forward call fail
        mock_httpx_client.post.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )

        result = predictor.predict(readings)
        assert result["status"] == "error"
        assert "sidecar" in result["message"].lower() or "error" in result["message"].lower()

    def test_reset_buffer(self, mock_httpx_client):
        """reset_buffer clears the sliding window."""
        predictor = RemoteKANPredictor(base_url="http://localhost:8001")
        predictor._client = mock_httpx_client
        predictor.buffer = [1, 2, 3]
        predictor.reset_buffer()
        assert predictor.buffer == []

    def test_model_type_is_kan(self, mock_httpx_client):
        """model_type is always 'kan'."""
        predictor = RemoteKANPredictor(base_url="http://localhost:8001")
        predictor._client = mock_httpx_client
        assert predictor.model_type == "kan"
