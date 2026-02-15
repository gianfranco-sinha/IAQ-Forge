"""FSM-based training pipeline that orchestrates the full training lifecycle."""

import enum
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.config import settings
from app.models import MODEL_REGISTRY, build_model
from training.data_sources import DataSource
from training.utils import (
    calculate_absolute_humidity,
    create_sliding_windows,
    evaluate_model,
    get_device,
    save_trained_model,
    train_model,
)

logger = logging.getLogger("training.pipeline")


class PipelineState(enum.Enum):
    IDLE = "idle"
    SOURCE_ACCESS = "source_access"
    INGESTION = "ingestion"
    FEATURE_ENGINEERING = "feature_engineering"
    WINDOWING = "windowing"
    SCALING = "scaling"
    SPLITTING = "splitting"
    TRAINING = "training"
    EVALUATION = "evaluation"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StageResult:
    state: PipelineState
    duration_seconds: float
    rows_in: Optional[int] = None
    rows_out: Optional[int] = None
    columns: Optional[List[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureInfo:
    failed_state: PipelineState
    error: Exception
    stage_results: List[StageResult]


@dataclass
class PipelineResult:
    metrics: Dict[str, float]
    training_history: Dict
    model_dir: Path
    stage_results: List[StageResult]


class PipelineError(Exception):
    def __init__(self, message: str, failure_info: FailureInfo):
        super().__init__(message)
        self.failure_info = failure_info


StageCallback = Callable[[PipelineState, StageResult], None]
ErrorCallback = Callable[[FailureInfo], None]


class TrainingPipeline:
    """Orchestrates source -> training completion for a single data source.

    Usage:
        pipeline = TrainingPipeline(
            source=InfluxDBSource(hours_back=168*4),
            model_type="mlp",
            epochs=200,
            window_size=10,
        )
        pipeline.on_stage_complete(my_callback)
        pipeline.on_error(my_error_handler)
        result = pipeline.orchestrate()
    """

    def __init__(
        self,
        source: DataSource,
        model_type: str,
        epochs: int = None,
        window_size: int = 10,
        test_size: float = None,
        random_state: int = None,
        min_samples: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        lr_scheduler_patience: int = None,
        lr_scheduler_factor: float = None,
        output_dir: str = None,
    ):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}. Must be one of {list(MODEL_REGISTRY)}")

        tcfg = settings.get_training_config()

        self._source = source
        self._model_type = model_type
        self._epochs = epochs if epochs is not None else tcfg["epochs"]
        self._window_size = window_size
        self._test_size = test_size if test_size is not None else tcfg["test_size"]
        self._random_state = random_state if random_state is not None else tcfg["random_state"]
        self._min_samples = min_samples if min_samples is not None else tcfg["min_samples"]
        self._batch_size = batch_size if batch_size is not None else tcfg["batch_size"]
        self._learning_rate = learning_rate if learning_rate is not None else tcfg["learning_rate"]
        self._lr_scheduler_patience = lr_scheduler_patience if lr_scheduler_patience is not None else tcfg["lr_scheduler_patience"]
        self._lr_scheduler_factor = lr_scheduler_factor if lr_scheduler_factor is not None else tcfg["lr_scheduler_factor"]
        self._output_dir = self._resolve_output_dir(output_dir)

        self._state = PipelineState.IDLE
        self._failure: Optional[FailureInfo] = None
        self._stage_results: List[StageResult] = []

        self._on_stage_enter_callbacks: List[StageCallback] = []
        self._on_stage_complete_callbacks: List[StageCallback] = []
        self._on_error_callbacks: List[ErrorCallback] = []

        # Inter-stage data (populated as stages execute)
        self._df = None
        self._X = None
        self._y = None
        self._baseline_gas_resistance = None
        self._X_scaled = None
        self._y_scaled = None
        self._feature_scaler = None
        self._target_scaler = None
        self._X_train = None
        self._X_val = None
        self._y_train = None
        self._y_val = None
        self._model = None
        self._training_history = None
        self._metrics = None
        self._model_dir = None
        self._device = None

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def failure(self) -> Optional[FailureInfo]:
        return self._failure

    @staticmethod
    def _resolve_output_dir(output_dir: str = None) -> Path:
        """Resolve and validate that output_dir is under TRAINED_MODELS_BASE."""
        base = Path(settings.TRAINED_MODELS_BASE).resolve()
        if output_dir is None:
            return None  # will default to base/model_type at save time
        target = Path(output_dir).resolve()
        if not (target == base or base in target.parents):
            raise ValueError(
                f"output_dir must be under '{settings.TRAINED_MODELS_BASE}/'. "
                f"Got: {output_dir}"
            )
        return target

    def on_stage_enter(self, callback: StageCallback) -> "TrainingPipeline":
        self._on_stage_enter_callbacks.append(callback)
        return self

    def on_stage_complete(self, callback: StageCallback) -> "TrainingPipeline":
        self._on_stage_complete_callbacks.append(callback)
        return self

    def on_error(self, callback: ErrorCallback) -> "TrainingPipeline":
        self._on_error_callbacks.append(callback)
        return self

    def _fire_enter(self, state: PipelineState) -> None:
        self._state = state
        logger.info("[%s] Starting...", state.value)
        for cb in self._on_stage_enter_callbacks:
            cb(state, None)

    def _transition(self, new_state: PipelineState, result: StageResult) -> None:
        msg = "[%s] Completed in %.2fs"
        args: list = [new_state.value, result.duration_seconds]
        if result.rows_out is not None:
            msg += " (%d rows)"
            args.append(result.rows_out)
        logger.info(msg, *args)
        logger.debug("[%s] extra=%s", new_state.value, result.extra)
        self._stage_results.append(result)
        for cb in self._on_stage_complete_callbacks:
            cb(new_state, result)

    def _fail(self, state: PipelineState, error: Exception) -> FailureInfo:
        logger.error(
            "[%s] FAILED: %s: %s",
            state.value,
            type(error).__name__,
            error,
            exc_info=True,
        )
        self._state = PipelineState.FAILED
        info = FailureInfo(
            failed_state=state,
            error=error,
            stage_results=list(self._stage_results),
        )
        self._failure = info
        for cb in self._on_error_callbacks:
            cb(info)
        return info

    # ── Stage implementations ──────────────────────────────────────────

    def _do_source_access(self) -> StageResult:
        t0 = time.monotonic()
        self._source.validate()
        return StageResult(
            state=PipelineState.SOURCE_ACCESS,
            duration_seconds=time.monotonic() - t0,
            extra={"source": self._source.name},
        )

    def _do_ingestion(self) -> StageResult:
        t0 = time.monotonic()
        self._df = self._source.fetch()
        n = len(self._df)
        if n < self._min_samples:
            raise ValueError(
                f"Insufficient data: got {n} samples, need at least {self._min_samples}"
            )
        if n < self._min_samples * 2:
            logger.warning(
                "Low sample count: %d (minimum is %d, recommended >= %d)",
                n, self._min_samples, self._min_samples * 2,
            )
        extra = {}
        if hasattr(self._df, "index") and len(self._df) > 0:
            try:
                extra["date_range_start"] = str(self._df.index.min())
                extra["date_range_end"] = str(self._df.index.max())
            except Exception:
                pass
        return StageResult(
            state=PipelineState.INGESTION,
            duration_seconds=time.monotonic() - t0,
            rows_in=n,
            rows_out=n,
            columns=list(self._df.columns),
            extra=extra,
        )

    def _do_feature_engineering(self) -> StageResult:
        t0 = time.monotonic()
        rows_in = len(self._df)

        features = self._df[["temperature", "rel_humidity", "pressure", "gas_resistance"]].values

        self._baseline_gas_resistance = float(np.median(features[:, 3]))
        logger.info("Baseline gas resistance: %.0f Ohm", self._baseline_gas_resistance)

        gas_ratio = features[:, 3] / self._baseline_gas_resistance
        abs_humidity = calculate_absolute_humidity(features[:, 0], features[:, 1])

        features_enhanced = np.column_stack(
            [features, gas_ratio.reshape(-1, 1), abs_humidity.reshape(-1, 1)]
        )

        targets = self._df["iaq"].values

        # Store for next stage
        self._features_enhanced = features_enhanced
        self._targets = targets

        return StageResult(
            state=PipelineState.FEATURE_ENGINEERING,
            duration_seconds=time.monotonic() - t0,
            rows_in=rows_in,
            rows_out=len(features_enhanced),
            columns=["temperature", "rel_humidity", "pressure", "gas_resistance", "gas_ratio", "abs_humidity"],
            extra={"baseline_gas_resistance": self._baseline_gas_resistance},
        )

    def _do_windowing(self) -> StageResult:
        t0 = time.monotonic()
        rows_in = len(self._features_enhanced)

        self._X, self._y = create_sliding_windows(
            self._features_enhanced, self._targets, self._window_size
        )

        feature_dim = self._X.shape[1] if self._X.ndim > 1 else self._X.shape[0]
        logger.info(
            "Created %d windows (window_size=%d, feature_dim=%d)",
            len(self._X), self._window_size, feature_dim,
        )

        return StageResult(
            state=PipelineState.WINDOWING,
            duration_seconds=time.monotonic() - t0,
            rows_in=rows_in,
            rows_out=len(self._X),
            extra={"window_size": self._window_size, "feature_dim": feature_dim},
        )

    def _do_scaling(self) -> StageResult:
        t0 = time.monotonic()
        rows_in = len(self._X)

        self._feature_scaler = StandardScaler()
        self._target_scaler = MinMaxScaler(feature_range=(0, 1))

        self._X_scaled = self._feature_scaler.fit_transform(self._X)
        self._y_scaled = self._target_scaler.fit_transform(
            self._y.reshape(-1, 1)
        ).flatten()

        return StageResult(
            state=PipelineState.SCALING,
            duration_seconds=time.monotonic() - t0,
            rows_in=rows_in,
            rows_out=len(self._X_scaled),
            extra={
                "feature_scaler": "StandardScaler",
                "target_scaler": "MinMaxScaler(0,1)",
            },
        )

    def _do_splitting(self) -> StageResult:
        t0 = time.monotonic()
        rows_in = len(self._X_scaled)

        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
            self._X_scaled,
            self._y_scaled,
            test_size=self._test_size,
            random_state=self._random_state,
        )

        logger.info(
            "Train/val split: %d / %d (test_size=%.2f)",
            len(self._X_train), len(self._X_val), self._test_size,
        )

        return StageResult(
            state=PipelineState.SPLITTING,
            duration_seconds=time.monotonic() - t0,
            rows_in=rows_in,
            rows_out=rows_in,
            extra={
                "train_samples": len(self._X_train),
                "val_samples": len(self._X_val),
                "test_size": self._test_size,
            },
        )

    def _do_training(self) -> StageResult:
        t0 = time.monotonic()

        self._device = get_device()

        self._model = build_model(
            self._model_type,
            window_size=self._window_size,
            num_features=6,
        )

        history = train_model(
            self._model,
            self._X_train,
            self._y_train,
            self._X_val,
            self._y_val,
            self._model_type.upper(),
            epochs=self._epochs,
            device=self._device,
            batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            lr_scheduler_patience=self._lr_scheduler_patience,
            lr_scheduler_factor=self._lr_scheduler_factor,
        )

        self._training_history = history
        best_val_loss = history["best_val_loss"]

        logger.info("Best validation loss: %.6f", best_val_loss)

        return StageResult(
            state=PipelineState.TRAINING,
            duration_seconds=time.monotonic() - t0,
            extra={
                "epochs": self._epochs,
                "best_val_loss": best_val_loss,
                "device": str(self._device),
            },
        )

    def _do_evaluation(self) -> StageResult:
        t0 = time.monotonic()

        self._metrics = evaluate_model(
            self._model,
            self._X_val,
            self._y_val,
            self._target_scaler,
            device=self._device,
        )

        logger.info(
            "Metrics: MAE=%.2f, RMSE=%.2f, R2=%.4f",
            self._metrics["mae"],
            self._metrics["rmse"],
            self._metrics["r2"],
        )

        return StageResult(
            state=PipelineState.EVALUATION,
            duration_seconds=time.monotonic() - t0,
            extra={
                "mae": self._metrics["mae"],
                "rmse": self._metrics["rmse"],
                "r2": self._metrics["r2"],
            },
        )

    def _do_saving(self) -> StageResult:
        t0 = time.monotonic()

        base = Path(settings.TRAINED_MODELS_BASE)
        self._model_dir = self._output_dir if self._output_dir else base / self._model_type

        save_trained_model(
            model=self._model,
            feature_scaler=self._feature_scaler,
            target_scaler=self._target_scaler,
            model_type=self._model_type,
            window_size=self._window_size,
            baseline_gas_resistance=self._baseline_gas_resistance,
            model_dir=str(self._model_dir),
            metrics=self._metrics,
            training_history=self._training_history,
        )

        artifacts = []
        if self._model_dir.exists():
            artifacts = [f.name for f in self._model_dir.iterdir() if f.is_file()]

        return StageResult(
            state=PipelineState.SAVING,
            duration_seconds=time.monotonic() - t0,
            extra={
                "model_dir": str(self._model_dir),
                "artifacts": artifacts,
            },
        )

    # ── Orchestrator ───────────────────────────────────────────────────

    def orchestrate(self) -> PipelineResult:
        """Run the full pipeline from IDLE to COMPLETED."""
        stages = [
            (PipelineState.SOURCE_ACCESS, self._do_source_access),
            (PipelineState.INGESTION, self._do_ingestion),
            (PipelineState.FEATURE_ENGINEERING, self._do_feature_engineering),
            (PipelineState.WINDOWING, self._do_windowing),
            (PipelineState.SCALING, self._do_scaling),
            (PipelineState.SPLITTING, self._do_splitting),
            (PipelineState.TRAINING, self._do_training),
            (PipelineState.EVALUATION, self._do_evaluation),
            (PipelineState.SAVING, self._do_saving),
        ]

        for state, stage_fn in stages:
            self._fire_enter(state)
            try:
                result = stage_fn()
                self._transition(state, result)
            except Exception as e:
                info = self._fail(state, e)
                raise PipelineError(
                    f"Pipeline failed at {state.value}: {type(e).__name__}: {e}",
                    failure_info=info,
                ) from e

        self._state = PipelineState.COMPLETED
        return self._build_result()

    def _build_result(self) -> PipelineResult:
        return PipelineResult(
            metrics=self._metrics,
            training_history=self._training_history,
            model_dir=self._model_dir,
            stage_results=list(self._stage_results),
        )
