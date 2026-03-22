"""Experiment tracking abstraction — keeps MLflow out of core training code."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker(ABC):
    """ABC for experiment tracking backends."""

    @abstractmethod
    def start_run(self, model_type: str, experiment_name: str = "iaq4j") -> None: ...

    @abstractmethod
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float) -> None: ...

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]) -> None: ...

    @abstractmethod
    def log_tags(self, tags: Dict[str, str]) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str) -> None: ...

    @abstractmethod
    def log_model(self, model: Any, name: str = "model") -> None: ...

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None: ...


class NullTracker(ExperimentTracker):
    """No-op tracker — used when no tracking backend is available."""

    def start_run(self, model_type: str, experiment_name: str = "iaq4j") -> None:
        pass

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        pass

    def log_tags(self, tags: Dict[str, str]) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def log_model(self, model: Any, name: str = "model") -> None:
        pass

    def end_run(self, status: str = "FINISHED") -> None:
        pass


class MLflowTracker(ExperimentTracker):
    """Wraps all MLflow calls. Fails fast if mlflow is not installed."""

    def __init__(self) -> None:
        try:
            import mlflow
            import mlflow.pytorch

            self._mlflow = mlflow
            self._mlflow_pytorch = mlflow.pytorch
        except ImportError:
            raise ImportError(
                "mlflow is not installed. Install it with: pip install mlflow"
            )

        from integrations.config import get_integration_config

        cfg = get_integration_config("mlflow")
        if cfg.get("tracking_uri"):
            self._mlflow.set_tracking_uri(cfg["tracking_uri"])

    def start_run(self, model_type: str, experiment_name: str = "iaq4j") -> None:
        self._mlflow.set_experiment(experiment_name)
        self._mlflow.start_run(run_name=model_type)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float) -> None:
        self._mlflow.log_metrics(
            {"train_loss": train_loss, "val_loss": val_loss, "lr": lr},
            step=epoch,
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def log_tags(self, tags: Dict[str, str]) -> None:
        self._mlflow.set_tags(tags)

    def log_artifact(self, path: str) -> None:
        if not Path(path).exists():
            logger.debug("Skipping missing artifact: %s", path)
            return
        self._mlflow.log_artifact(path)

    def log_model(self, model: Any, name: str = "model") -> None:
        self._mlflow_pytorch.log_model(model, name=name)

    def end_run(self, status: str = "FINISHED") -> None:
        self._mlflow.end_run(status=status)
