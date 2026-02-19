"""FSM-based training pipeline that orchestrates the full training lifecycle."""

import enum
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.config import settings
from app.models import MODEL_REGISTRY, build_model
import app.builtin_profiles  # noqa: F401  — registers sensor/standard profiles
from app.profiles import get_iaq_standard, get_sensor_profile
from training.data_sources import DataSource
from training.utils import (
    _compute_data_fingerprint,
    _compute_feature_statistics,
    _get_git_commit,
    create_sliding_windows,
    evaluate_model,
    find_contiguous_segments,
    get_device,
    save_trained_model,
    train_model,
    update_central_manifest,
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


class IssueSeverity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Issue:
    severity: IssueSeverity
    stage: str
    message: str
    rows_affected: Optional[int] = None


@dataclass
class PreprocessingReport:
    issues: List[Issue] = field(default_factory=list)

    def add(self, severity: IssueSeverity, stage: str, message: str, rows_affected: int = None):
        self.issues.append(Issue(severity, stage, message, rows_affected))

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def has_errors(self) -> bool:
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    def log_summary(self):
        if not self.issues:
            logger.info("[preprocessing_report] No issues found")
            return
        errors = self.errors
        warnings = self.warnings
        infos = [i for i in self.issues if i.severity == IssueSeverity.INFO]
        logger.info(
            "[preprocessing_report] %d issue(s): %d error, %d warning, %d info",
            len(self.issues), len(errors), len(warnings), len(infos),
        )
        for issue in self.issues:
            log_fn = {
                IssueSeverity.ERROR: logger.error,
                IssueSeverity.WARNING: logger.warning,
                IssueSeverity.INFO: logger.info,
            }[issue.severity]
            msg = "[preprocessing_report] [%s] %s"
            args = [issue.stage, issue.message]
            if issue.rows_affected is not None:
                msg += " (%d rows)"
                args.append(issue.rows_affected)
            log_fn(msg, *args)


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
    preprocessing_report: PreprocessingReport = field(default_factory=PreprocessingReport)
    version: str = ""


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
        self._sensor_cfg = settings.get_sensor_config()
        self._sensor_profile = get_sensor_profile()
        self._iaq_standard = get_iaq_standard()

        self._state = PipelineState.IDLE
        self._failure: Optional[FailureInfo] = None
        self._stage_results: List[StageResult] = []
        self._report = PreprocessingReport()

        self._on_stage_enter_callbacks: List[StageCallback] = []
        self._on_stage_complete_callbacks: List[StageCallback] = []
        self._on_error_callbacks: List[ErrorCallback] = []

        # Data provenance tracking
        self._data_fingerprint: Optional[str] = None
        self._feature_stats: Optional[Dict] = None
        self._target_stats: Optional[Dict] = None
        self._version: str = ""

        # Inter-stage data (populated as stages execute)
        self._df = None
        self._X = None
        self._y = None
        self._baselines: Dict[str, float] = {}
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
        self._tb_log_dir = None

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
        import pandas as pd

        t0 = time.monotonic()
        self._df = self._source.fetch()
        n_raw = len(self._df)
        report = self._report

        # ── Check: sufficient data ────────────────────────────────────
        if n_raw < self._min_samples:
            raise ValueError(
                f"Insufficient data: got {n_raw} samples, need at least {self._min_samples}"
            )
        if n_raw < self._min_samples * 2:
            report.add(IssueSeverity.WARNING, "ingestion",
                       f"Low sample count ({n_raw}), recommended >= {self._min_samples * 2}",
                       rows_affected=n_raw)

        has_timestamps = isinstance(self._df.index, pd.DatetimeIndex)

        # ── Check: timestamp presence ─────────────────────────────────
        if not has_timestamps:
            report.add(IssueSeverity.INFO, "ingestion",
                       "No DatetimeIndex — gap detection and ordering checks skipped")
        else:
            # ── Check: NaT values ─────────────────────────────────────
            nat_count = int(self._df.index.isna().sum())
            if nat_count > 0:
                report.add(IssueSeverity.ERROR, "ingestion",
                           f"Found {nat_count} missing timestamp(s) (NaT) — rows dropped",
                           rows_affected=nat_count)
                self._df = self._df[self._df.index.notna()]

            # ── Check: chronological order ────────────────────────────
            if not self._df.index.is_monotonic_increasing:
                report.add(IssueSeverity.WARNING, "ingestion",
                           "Data not in chronological order — sorted by timestamp")
                self._df = self._df.sort_index()

            # ── Check: duplicate timestamps ───────────────────────────
            dup_count = int(self._df.index.duplicated().sum())
            if dup_count > 0:
                report.add(IssueSeverity.WARNING, "ingestion",
                           f"Found {dup_count} duplicate timestamp(s) — kept first occurrence",
                           rows_affected=dup_count)
                self._df = self._df[~self._df.index.duplicated(keep="first")]

        # ── Check: NaN values in data columns ─────────────────────────
        nan_rows = int(self._df.isna().any(axis=1).sum())
        if nan_rows > 0:
            report.add(IssueSeverity.WARNING, "ingestion",
                       f"Found {nan_rows} row(s) with NaN values — dropped",
                       rows_affected=nan_rows)
            self._df = self._df.dropna()

        # ── Check: sensor range outliers ────────────────────────────────
        valid_ranges = self._sensor_profile.valid_ranges
        outlier_mask = np.zeros(len(self._df), dtype=bool)

        for col, (lo, hi) in valid_ranges.items():
            if col not in self._df.columns:
                continue
            col_outliers = (self._df[col] < lo) | (self._df[col] > hi)
            n_outliers = int(col_outliers.sum())
            if n_outliers > 0:
                report.add(
                    IssueSeverity.WARNING, "ingestion",
                    f"{col} has {n_outliers} value(s) outside valid range "
                    f"[{lo}, {hi}] — rows dropped",
                    rows_affected=n_outliers,
                )
            outlier_mask |= col_outliers.values

        total_outliers = int(outlier_mask.sum())
        if total_outliers > 0:
            self._df = self._df[~outlier_mask]
            logger.info(
                "Removed %d outlier row(s) based on %s sensor ranges",
                total_outliers, self._sensor_profile.name,
            )

        n_clean = len(self._df)
        if n_clean < self._min_samples:
            raise ValueError(
                f"Insufficient data after cleaning: {n_clean} samples "
                f"(started with {n_raw}, need at least {self._min_samples})"
            )

        # Data fingerprint — SHA256 of cleaned data before feature engineering
        self._data_fingerprint = _compute_data_fingerprint(self._df)

        extra = {"rows_before_cleaning": n_raw, "rows_after_cleaning": n_clean,
                 "data_fingerprint": self._data_fingerprint}
        if has_timestamps and n_clean > 0:
            extra["date_range_start"] = str(self._df.index.min())
            extra["date_range_end"] = str(self._df.index.max())

        return StageResult(
            state=PipelineState.INGESTION,
            duration_seconds=time.monotonic() - t0,
            rows_in=n_raw,
            rows_out=n_clean,
            columns=list(self._df.columns),
            extra=extra,
        )

    def _do_feature_engineering(self) -> StageResult:
        t0 = time.monotonic()
        rows_in = len(self._df)

        profile = self._sensor_profile
        target_col = self._iaq_standard.target_column

        raw = self._df[profile.raw_features].values

        self._baselines = profile.compute_baselines(raw)
        if self._baselines:
            logger.info("Baselines: %s", self._baselines)

        features_enhanced = profile.engineer_features(raw, self._baselines)
        targets = self._df[target_col].values

        # Store for next stage
        self._features_enhanced = features_enhanced
        self._targets = targets

        # Compute statistics for manifest
        self._feature_stats = _compute_feature_statistics(
            features_enhanced, profile.all_feature_names
        )
        self._target_stats = {
            target_col: {
                "mean": float(np.mean(targets)),
                "std": float(np.std(targets)),
                "min": float(np.min(targets)),
                "max": float(np.max(targets)),
            }
        }

        return StageResult(
            state=PipelineState.FEATURE_ENGINEERING,
            duration_seconds=time.monotonic() - t0,
            rows_in=rows_in,
            rows_out=len(features_enhanced),
            columns=profile.all_feature_names,
            extra={"baselines": self._baselines},
        )

    def _do_windowing(self) -> StageResult:
        t0 = time.monotonic()
        rows_in = len(self._features_enhanced)

        # Detect time gaps and window each contiguous segment independently
        segments, gap_info = find_contiguous_segments(self._df.index)

        if gap_info.get("gaps_found", 0) > 0:
            logger.info(
                "Detected %d gap(s) → %d contiguous segments "
                "(median_interval=%.0fs, threshold=%.0fs, largest_gap=%.0fs)",
                gap_info["gaps_found"],
                gap_info["segments"],
                gap_info["median_interval_seconds"],
                gap_info["gap_threshold_seconds"],
                gap_info.get("largest_gap_seconds", 0),
            )
            self._report.add(
                IssueSeverity.WARNING, "windowing",
                f"Detected {gap_info['gaps_found']} time gap(s) across "
                f"{gap_info['segments']} contiguous segments "
                f"(largest gap: {gap_info.get('largest_gap_seconds', 0):.0f}s)",
            )

        all_X, all_y = [], []
        skipped_segments = 0
        for start, end in segments:
            seg_len = end - start
            if seg_len < self._window_size:
                skipped_segments += 1
                continue
            seg_X, seg_y = create_sliding_windows(
                self._features_enhanced[start:end],
                self._targets[start:end],
                self._window_size,
            )
            all_X.append(seg_X)
            all_y.append(seg_y)

        if not all_X:
            raise ValueError(
                f"No contiguous segments long enough for window_size={self._window_size} "
                f"({gap_info['segments']} segments, {skipped_segments} too short)"
            )

        self._X = np.concatenate(all_X)
        self._y = np.concatenate(all_y)

        feature_dim = self._X.shape[1] if self._X.ndim > 1 else self._X.shape[0]
        logger.info(
            "Created %d windows (window_size=%d, feature_dim=%d)",
            len(self._X), self._window_size, feature_dim,
        )

        extra = {"window_size": self._window_size, "feature_dim": feature_dim}
        extra.update(gap_info)
        if skipped_segments:
            extra["skipped_short_segments"] = skipped_segments
            logger.warning(
                "Skipped %d segment(s) shorter than window_size=%d",
                skipped_segments, self._window_size,
            )
            self._report.add(
                IssueSeverity.WARNING, "windowing",
                f"Skipped {skipped_segments} segment(s) shorter than window_size={self._window_size}",
            )

        return StageResult(
            state=PipelineState.WINDOWING,
            duration_seconds=time.monotonic() - t0,
            rows_in=rows_in,
            rows_out=len(self._X),
            extra=extra,
        )

    def _do_splitting(self) -> StageResult:
        """Chronological split — no shuffling to prevent data leakage."""
        t0 = time.monotonic()
        rows_in = len(self._X)

        split_idx = int(len(self._X) * (1 - self._test_size))
        self._X_train = self._X[:split_idx]
        self._X_val = self._X[split_idx:]
        self._y_train = self._y[:split_idx]
        self._y_val = self._y[split_idx:]

        logger.info(
            "Chronological split: %d train / %d val (test_size=%.2f)",
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

    def _do_scaling(self) -> StageResult:
        """Fit scalers on training data only, then transform both splits."""
        t0 = time.monotonic()

        self._feature_scaler = StandardScaler()
        self._target_scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit on training data only to prevent leakage
        self._X_train = self._feature_scaler.fit_transform(self._X_train)
        self._X_val = self._feature_scaler.transform(self._X_val)

        self._y_train = self._target_scaler.fit_transform(
            self._y_train.reshape(-1, 1)
        ).flatten()
        self._y_val = self._target_scaler.transform(
            self._y_val.reshape(-1, 1)
        ).flatten()

        return StageResult(
            state=PipelineState.SCALING,
            duration_seconds=time.monotonic() - t0,
            rows_in=len(self._X_train) + len(self._X_val),
            rows_out=len(self._X_train) + len(self._X_val),
            extra={
                "feature_scaler": "StandardScaler",
                "target_scaler": "MinMaxScaler(0,1)",
                "fit_on": "train_only",
            },
        )

    def _resolve_tensorboard_dir(self) -> str:
        """Build a TensorBoard log_dir for this run, or return None if disabled."""
        tcfg = settings.get_training_config()
        if not tcfg.get("tensorboard_enabled", True):
            return None

        import datetime

        base = Path(tcfg.get("tensorboard_log_dir", "runs"))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / f"{self._model_type}_{timestamp}"
        return str(run_dir)

    def _do_training(self) -> StageResult:
        t0 = time.monotonic()

        self._device = get_device()

        self._model = build_model(
            self._model_type,
            window_size=self._window_size,
            num_features=self._sensor_profile.total_features,
        )

        if hasattr(self._model, 'kl_loss'):
            bnn_cfg = settings.get_model_config('bnn')
            self._model._kl_weight = bnn_cfg.get('kl_weight', 1.0)

        tcfg = settings.get_training_config()
        self._tb_log_dir = self._resolve_tensorboard_dir()
        histogram_freq = tcfg.get("tensorboard_histogram_freq", 50)

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
            log_dir=self._tb_log_dir,
            histogram_freq=histogram_freq,
        )

        self._training_history = history
        best_val_loss = history["best_val_loss"]

        logger.info("Best validation loss: %.6f", best_val_loss)

        extra = {
            "epochs": self._epochs,
            "best_val_loss": best_val_loss,
            "device": str(self._device),
        }
        if self._tb_log_dir:
            extra["tensorboard_log_dir"] = self._tb_log_dir

        return StageResult(
            state=PipelineState.TRAINING,
            duration_seconds=time.monotonic() - t0,
            extra=extra,
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

        # ── TensorBoard: log hyperparams + final metrics ─────────────
        if self._tb_log_dir:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=self._tb_log_dir)
            writer.add_hparams(
                hparam_dict={
                    "model_type": self._model_type,
                    "epochs": self._epochs,
                    "batch_size": self._batch_size,
                    "learning_rate": self._learning_rate,
                    "window_size": self._window_size,
                    "lr_scheduler_patience": self._lr_scheduler_patience,
                    "lr_scheduler_factor": self._lr_scheduler_factor,
                },
                metric_dict={
                    "hparam/mae": self._metrics["mae"],
                    "hparam/rmse": self._metrics["rmse"],
                    "hparam/r2": self._metrics["r2"],
                    "hparam/best_val_loss": self._training_history["best_val_loss"],
                },
            )
            writer.flush()
            writer.close()

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

        # Build data provenance manifest
        manifest = self._build_data_manifest()

        save_trained_model(
            model=self._model,
            feature_scaler=self._feature_scaler,
            target_scaler=self._target_scaler,
            model_type=self._model_type,
            window_size=self._window_size,
            baselines=self._baselines,
            sensor_type=self._sensor_profile.name,
            iaq_standard=self._iaq_standard.name,
            model_dir=str(self._model_dir),
            metrics=self._metrics,
            training_history=self._training_history,
            data_manifest=manifest,
        )

        # Update central manifest and get version
        import pandas as pd

        run_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_fingerprint": self._data_fingerprint,
            "model_dir": str(self._model_dir),
            "metrics": {k: float(v) for k, v in self._metrics.items()},
            "git_commit": manifest["git_commit"],
        }
        self._version = update_central_manifest(self._model_type, run_entry)
        manifest["version"] = self._version

        # Re-save manifest with version included
        from training.utils import save_data_manifest
        save_data_manifest(manifest, self._model_dir)

        logger.info("Registered as %s in central manifest", self._version)

        artifacts = []
        if self._model_dir.exists():
            artifacts = [f.name for f in self._model_dir.iterdir() if f.is_file()]

        return StageResult(
            state=PipelineState.SAVING,
            duration_seconds=time.monotonic() - t0,
            extra={
                "model_dir": str(self._model_dir),
                "version": self._version,
                "artifacts": artifacts,
            },
        )

    def _build_data_manifest(self) -> dict:
        """Assemble the full data provenance manifest from pipeline state."""
        import pandas as pd

        # Serialize stage results
        stages = []
        for sr in self._stage_results:
            stages.append({
                "stage": sr.state.value,
                "duration_seconds": round(sr.duration_seconds, 3),
                "rows_in": sr.rows_in,
                "rows_out": sr.rows_out,
                "columns": sr.columns,
                "extra": sr.extra,
            })

        # Serialize preprocessing issues
        issues = []
        for issue in self._report.issues:
            issues.append({
                "severity": issue.severity.value,
                "stage": issue.stage,
                "message": issue.message,
                "rows_affected": issue.rows_affected,
            })

        # Config snapshot
        config_snapshot = {
            "model_type": self._model_type,
            "epochs": self._epochs,
            "window_size": self._window_size,
            "batch_size": self._batch_size,
            "learning_rate": self._learning_rate,
            "test_size": self._test_size,
            "random_state": self._random_state,
            "min_samples": self._min_samples,
            "lr_scheduler_patience": self._lr_scheduler_patience,
            "lr_scheduler_factor": self._lr_scheduler_factor,
            "sensor_type": self._sensor_profile.name,
            "iaq_standard": self._iaq_standard.name,
        }

        return {
            "created_at": pd.Timestamp.now().isoformat(),
            "git_commit": _get_git_commit(),
            "data_fingerprint": self._data_fingerprint,
            "data_source": self._source.metadata,
            "config": config_snapshot,
            "feature_statistics": self._feature_stats,
            "target_statistics": self._target_stats,
            "stages": stages,
            "preprocessing_issues": issues,
            "metrics": {k: float(v) for k, v in self._metrics.items()} if self._metrics else {},
        }

    # ── Orchestrator ───────────────────────────────────────────────────

    def orchestrate(self) -> PipelineResult:
        """Run the full pipeline from IDLE to COMPLETED."""
        stages = [
            (PipelineState.SOURCE_ACCESS, self._do_source_access),
            (PipelineState.INGESTION, self._do_ingestion),
            (PipelineState.FEATURE_ENGINEERING, self._do_feature_engineering),
            (PipelineState.WINDOWING, self._do_windowing),
            (PipelineState.SPLITTING, self._do_splitting),
            (PipelineState.SCALING, self._do_scaling),
            (PipelineState.TRAINING, self._do_training),
            (PipelineState.EVALUATION, self._do_evaluation),
            (PipelineState.SAVING, self._do_saving),
        ]

        preprocessing_end = PipelineState.SCALING

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

            if state == preprocessing_end:
                self._report.log_summary()

        self._state = PipelineState.COMPLETED
        return self._build_result()

    def _build_result(self) -> PipelineResult:
        return PipelineResult(
            metrics=self._metrics,
            training_history=self._training_history,
            model_dir=self._model_dir,
            stage_results=list(self._stage_results),
            preprocessing_report=self._report,
            version=self._version,
        )
