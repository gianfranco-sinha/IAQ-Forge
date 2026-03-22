"""Tests for training.experiment_tracker — ABC, NullTracker, MLflowTracker."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from integrations.mlflow.tracker import ExperimentTracker, MLflowTracker, NullTracker


class TestNullTracker:
    """NullTracker implements the ABC and all methods are safe no-ops."""

    def test_is_experiment_tracker(self):
        assert isinstance(NullTracker(), ExperimentTracker)

    def test_all_methods_callable(self):
        t = NullTracker()
        t.start_run("mlp")
        t.log_epoch(1, 0.5, 0.4, 0.001)
        t.log_params({"a": 1})
        t.log_metrics({"loss": 0.1})
        t.log_tags({"v": "1.0"})
        t.log_artifact("/nonexistent/path")
        t.log_model(torch.nn.Linear(1, 1))
        t.end_run()
        t.end_run(status="FAILED")
        t.end_run(status="KILLED")


class TestMLflowTracker:
    """MLflowTracker delegates to mlflow; raises ImportError when missing."""

    def _make_tracker(self):
        """Build an MLflowTracker with a mocked mlflow module."""
        mock_mlflow = MagicMock()
        mock_pytorch = MagicMock()
        mock_mlflow.pytorch = mock_pytorch
        with patch.dict(sys.modules, {"mlflow": mock_mlflow, "mlflow.pytorch": mock_pytorch}):
            tracker = MLflowTracker()
        return tracker, mock_mlflow, mock_pytorch

    def test_raises_when_mlflow_missing(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            with pytest.raises(ImportError, match="mlflow is not installed"):
                MLflowTracker()

    def test_start_run(self):
        t, ml, _ = self._make_tracker()
        t.start_run("mlp", "my_exp")
        ml.set_experiment.assert_called_once_with("my_exp")
        ml.start_run.assert_called_once_with(run_name="mlp")

    def test_log_epoch(self):
        t, ml, _ = self._make_tracker()
        t.log_epoch(5, 0.3, 0.2, 0.01)
        ml.log_metrics.assert_called_once_with(
            {"train_loss": 0.3, "val_loss": 0.2, "lr": 0.01}, step=5
        )

    def test_log_params(self):
        t, ml, _ = self._make_tracker()
        t.log_params({"lr": 0.01})
        ml.log_params.assert_called_once_with({"lr": 0.01})

    def test_log_metrics(self):
        t, ml, _ = self._make_tracker()
        t.log_metrics({"mae": 1.5})
        ml.log_metrics.assert_called_once_with({"mae": 1.5})

    def test_log_tags(self):
        t, ml, _ = self._make_tracker()
        t.log_tags({"version": "mlp-1.0.0"})
        ml.set_tags.assert_called_once_with({"version": "mlp-1.0.0"})

    def test_log_artifact_existing_file(self, tmp_path):
        t, ml, _ = self._make_tracker()
        f = tmp_path / "scaler.pkl"
        f.write_text("data")
        t.log_artifact(str(f))
        ml.log_artifact.assert_called_once_with(str(f))

    def test_log_artifact_missing_file(self):
        t, ml, _ = self._make_tracker()
        t.log_artifact("/does/not/exist.pkl")
        ml.log_artifact.assert_not_called()

    def test_log_model(self):
        t, _, pt = self._make_tracker()
        model = torch.nn.Linear(1, 1)
        t.log_model(model, name="my_model")
        pt.log_model.assert_called_once_with(model, name="my_model")

    def test_end_run(self):
        t, ml, _ = self._make_tracker()
        t.end_run()
        ml.end_run.assert_called_once_with(status="FINISHED")

    def test_end_run_failed(self):
        t, ml, _ = self._make_tracker()
        t.end_run(status="FAILED")
        ml.end_run.assert_called_once_with(status="FAILED")
