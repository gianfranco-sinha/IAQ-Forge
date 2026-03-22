"""TrainingPipeline — orchestrates the full training lifecycle via StageMachine."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.config import settings
from app.models import MODEL_REGISTRY, build_model
import app.builtin_profiles  # noqa: F401  — registers sensor/standard profiles
from app.profiles import get_iaq_standard, get_sensor_profile
from app.exceptions import ConfigurationError, InsufficientDataError, NegativeR2Error
from training.data_sources import DataSource
from training.utils import (
    _compute_data_fingerprint,
    _compute_feature_statistics,
    _get_git_commit,
    compute_schema_fingerprint,
    create_sliding_windows,
    evaluate_model,
    find_contiguous_segments,
    get_device,
    patch_config_with_merkle_hash,
    patch_config_with_version,
    patch_manifest_merkle,
    save_trained_model,
    seed_everything,
    train_model,
    update_central_manifest,
)
from training.merkle import build_merkle_tree_from_context
from training.pipeline.types import (
    ErrorCallback,
    IssueSeverity,
    PipelineResult,
    PipelineState,
    PreprocessingReport,
    StageCallback,
    StageResult,
)
from training.pipeline.fsm import StageMachine

logger = logging.getLogger("training.pipeline")


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
        early_stopping_patience: int = None,
        output_dir: str = None,
        on_epoch: Optional[Callable] = None,
        resume: bool = False,
    ):
        if model_type not in MODEL_REGISTRY:
            raise ConfigurationError(
                f"Unsupported model type: {model_type}. Must be one of {list(MODEL_REGISTRY)}"
            )

        tcfg = settings.get_training_config()

        self._source = source
        self._model_type = model_type
        self._epochs = epochs if epochs is not None else tcfg["epochs"]
        self._window_size = window_size
        self._test_size = test_size if test_size is not None else tcfg["test_size"]
        self._random_state = (
            random_state if random_state is not None else tcfg["random_state"]
        )
        self._min_samples = (
            min_samples if min_samples is not None else tcfg["min_samples"]
        )
        self._batch_size = batch_size if batch_size is not None else tcfg["batch_size"]
        self._learning_rate = (
            learning_rate if learning_rate is not None else tcfg["learning_rate"]
        )
        self._lr_scheduler_patience = (
            lr_scheduler_patience
            if lr_scheduler_patience is not None
            else tcfg["lr_scheduler_patience"]
        )
        self._lr_scheduler_factor = (
            lr_scheduler_factor
            if lr_scheduler_factor is not None
            else tcfg["lr_scheduler_factor"]
        )
        self._early_stopping_patience = (
            early_stopping_patience
            if early_stopping_patience is not None
            else tcfg.get("early_stopping_patience", 0)
        )
        self._output_dir = self._resolve_output_dir(output_dir)
        self._sensor_cfg = settings.get_sensor_config()

        # Pass explicit types from config to avoid hidden config coupling
        model_cfg_yaml = settings.load_model_config()
        sensor_type = model_cfg_yaml.get("sensor", {}).get("type")
        standard_type = model_cfg_yaml.get("iaq_standard", {}).get("type")
        self._sensor_profile = get_sensor_profile(sensor_type)
        self._iaq_standard = get_iaq_standard(standard_type)

        self._report = PreprocessingReport()
        self._on_epoch_callback: Optional[Callable] = on_epoch
        self._resume = resume

        # Data provenance tracking
        self._data_fingerprint: Optional[str] = None
        self._feature_stats: Optional[Dict] = None
        self._target_stats: Optional[Dict] = None
        self._schema_fp: Optional[str] = None
        self._version: str = ""
        self._merkle_root = None
        self._raw_row_count: Optional[int] = None

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
        self._training_interrupted = False
        self._metrics = None
        self._model_dir = None
        self._device = None
        self._tb_log_dir = None

        # Wire up the FSM
        self._fsm = StageMachine()
        self._fsm.add_stage(PipelineState.SOURCE_ACCESS, self._do_source_access)
        self._fsm.add_stage(PipelineState.INGESTION, self._do_ingestion)
        self._fsm.add_stage(PipelineState.FEATURE_ENGINEERING, self._do_feature_engineering)
        self._fsm.add_stage(PipelineState.WINDOWING, self._do_windowing)
        self._fsm.add_stage(PipelineState.SPLITTING, self._do_splitting)
        self._fsm.add_stage(PipelineState.SCALING, self._do_scaling)
        self._fsm.add_stage(PipelineState.TRAINING, self._do_training)
        self._fsm.add_stage(PipelineState.EVALUATION, self._do_evaluation)
        self._fsm.add_stage(PipelineState.SAVING, self._do_saving)

    @property
    def state(self) -> PipelineState:
        return self._fsm.state or PipelineState.IDLE

    @property
    def failure(self):
        return self._fsm.failure

    @property
    def model(self):
        """The trained model (available after TRAINING stage)."""
        return self._model

    @staticmethod
    def _resolve_output_dir(output_dir: str = None) -> Path:
        """Resolve and validate that output_dir is under TRAINED_MODELS_BASE."""
        base = Path(settings.TRAINED_MODELS_BASE).resolve()
        if output_dir is None:
            return None  # will default to base/model_type at save time
        target = Path(output_dir).resolve()
        if not (target == base or base in target.parents):
            raise ConfigurationError(
                f"output_dir must be under '{settings.TRAINED_MODELS_BASE}/'. "
                f"Got: {output_dir}"
            )
        return target

    def on_stage_enter(self, callback: StageCallback) -> "TrainingPipeline":
        self._fsm.on_stage_enter(callback)
        return self

    def on_stage_complete(self, callback: StageCallback) -> "TrainingPipeline":
        self._fsm.on_stage_complete(callback)
        return self

    def on_error(self, callback: ErrorCallback) -> "TrainingPipeline":
        self._fsm.on_error(callback)
        return self

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
        self._raw_row_count = n_raw
        report = self._report

        # ── Check: sufficient data ────────────────────────────────────
        if n_raw < self._min_samples:
            raise InsufficientDataError(
                f"Insufficient data: got {n_raw} samples, need at least {self._min_samples}"
            )
        if n_raw < self._min_samples * 2:
            report.add(
                IssueSeverity.WARNING,
                "ingestion",
                f"Low sample count ({n_raw}), recommended >= {self._min_samples * 2}",
                rows_affected=n_raw,
            )

        has_timestamps = isinstance(self._df.index, pd.DatetimeIndex)

        # ── Check: timestamp presence ─────────────────────────────────
        if not has_timestamps:
            report.add(
                IssueSeverity.INFO,
                "ingestion",
                "No DatetimeIndex — gap detection and ordering checks skipped",
            )
        else:
            # ── Check: NaT values ─────────────────────────────────────
            nat_count = int(self._df.index.isna().sum())
            if nat_count > 0:
                report.add(
                    IssueSeverity.ERROR,
                    "ingestion",
                    f"Found {nat_count} missing timestamp(s) (NaT) — rows dropped",
                    rows_affected=nat_count,
                )
                self._df = self._df[self._df.index.notna()]

            # ── Check: timezone normalization ─────────────────────────
            if self._df.index.tz is None:
                self._df.index = self._df.index.tz_localize("UTC")
                report.add(IssueSeverity.WARNING, "ingestion", "Naive timestamps localized to UTC")
            elif str(self._df.index.tz) != "UTC":
                orig_tz = str(self._df.index.tz)
                self._df.index = self._df.index.tz_convert("UTC")
                report.add(IssueSeverity.WARNING, "ingestion", f"Converted timestamps from {orig_tz} to UTC")

            # ── Check: chronological order ────────────────────────────
            if not self._df.index.is_monotonic_increasing:
                report.add(
                    IssueSeverity.WARNING,
                    "ingestion",
                    "Data not in chronological order — sorted by timestamp",
                )
                self._df = self._df.sort_index()

            # ── Check: duplicate timestamps ───────────────────────────
            dup_count = int(self._df.index.duplicated().sum())
            if dup_count > 0:
                report.add(
                    IssueSeverity.WARNING,
                    "ingestion",
                    f"Found {dup_count} duplicate timestamp(s) — kept first occurrence",
                    rows_affected=dup_count,
                )
                self._df = self._df[~self._df.index.duplicated(keep="first")]

            # ── Check: sampling interval ──────────────────────────────
            interval_stats = None
            expected = self._sensor_profile.expected_interval_seconds
            if expected is not None and len(self._df) > 1:
                deltas = self._df.index.to_series().diff().dt.total_seconds().dropna()
                median_interval = float(deltas.median())
                ratio = median_interval / expected
                interval_stats = {
                    "median_seconds": round(median_interval, 2),
                    "p5_seconds": round(float(deltas.quantile(0.05)), 2),
                    "p95_seconds": round(float(deltas.quantile(0.95)), 2),
                    "expected_seconds": expected,
                }
                if ratio > 5.0 or ratio < 0.2:
                    report.add(IssueSeverity.WARNING, "ingestion",
                        f"Sampling interval mismatch: median {median_interval:.1f}s "
                        f"vs expected {expected:.1f}s (ratio {ratio:.1f}x)")
                # Detect variable-rate sampling (bimodal intervals)
                if median_interval > 0 and len(deltas) > 1:
                    cv = float(deltas.std() / median_interval)
                    interval_stats["cv"] = round(cv, 3)
                    if cv > 1.0:
                        report.add(IssueSeverity.WARNING, "ingestion",
                            f"High sampling interval variability (CV={cv:.2f}) — "
                            f"possible multi-rate or irregular sampling")

        # ── Check: NaN values in data columns ─────────────────────────
        nan_rows = int(self._df.isna().any(axis=1).sum())
        if nan_rows > 0:
            report.add(
                IssueSeverity.WARNING,
                "ingestion",
                f"Found {nan_rows} row(s) with NaN values — dropped",
                rows_affected=nan_rows,
            )
            self._df = self._df.dropna()

        # ── Unit auto-detection and conversion ─────────────────────────
        from app.quantities import get_quantity, convert_to_canonical

        feature_quantities = self._sensor_profile.feature_quantities
        provided_units = getattr(self._source, "provided_units", None) or {}

        for feat, qty_name in feature_quantities.items():
            if feat not in self._df.columns:
                continue

            qty = get_quantity(qty_name)

            # 1) Declared unit from source — convert directly
            declared_unit = provided_units.get(feat)
            if declared_unit and declared_unit != qty.canonical_unit:
                self._df[feat] = self._df[feat].apply(
                    lambda v, u=declared_unit, q=qty_name: convert_to_canonical(v, u, q)
                )
                report.add(
                    IssueSeverity.WARNING,
                    "ingestion",
                    f"{feat} converted from {declared_unit} to {qty.canonical_unit}",
                    rows_affected=len(self._df),
                )
                continue

            if declared_unit:
                continue  # declared as canonical — no auto-detection needed

            # 2) Auto-detect: sample min/max, check if canonical range fits
            if not qty.valid_range or not qty.alternate_units:
                continue

            sample = self._df[feat].iloc[:10_000]
            col_min, col_max = float(sample.min()), float(sample.max())
            lo, hi = qty.valid_range

            if lo <= col_min and col_max <= hi:
                continue  # already in canonical range

            # Try each alternate unit
            matched_unit = None
            for unit_name in qty.alternate_units:
                try:
                    conv_min = convert_to_canonical(col_min, unit_name, qty_name)
                    conv_max = convert_to_canonical(col_max, unit_name, qty_name)
                    if lo <= min(conv_min, conv_max) and max(conv_min, conv_max) <= hi:
                        matched_unit = unit_name
                        break
                except Exception:
                    continue

            if matched_unit:
                self._df[feat] = self._df[feat].apply(
                    lambda v, u=matched_unit, q=qty_name: convert_to_canonical(v, u, q)
                )
                report.add(
                    IssueSeverity.WARNING,
                    "ingestion",
                    f"{feat} auto-detected as {matched_unit}, "
                    f"converted to {qty.canonical_unit}",
                    rows_affected=len(self._df),
                )
                logger.info(
                    "Auto-converted %s from %s to %s",
                    feat, matched_unit, qty.canonical_unit,
                )
            else:
                report.add(
                    IssueSeverity.ERROR,
                    "ingestion",
                    f"Possible unit mismatch on {feat}: values "
                    f"[{col_min:.1f}, {col_max:.1f}] outside valid range "
                    f"[{lo}, {hi}] and no alternate unit matches",
                )

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
                    IssueSeverity.WARNING,
                    "ingestion",
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
                total_outliers,
                self._sensor_profile.name,
            )

        n_clean = len(self._df)
        if n_clean < self._min_samples:
            raise InsufficientDataError(
                f"Insufficient data after cleaning: {n_clean} samples "
                f"(started with {n_raw}, need at least {self._min_samples})"
            )

        # Data fingerprint — SHA256 of cleaned data before feature engineering
        self._data_fingerprint = _compute_data_fingerprint(self._df)

        extra = {
            "rows_before_cleaning": n_raw,
            "rows_after_cleaning": n_clean,
            "data_fingerprint": self._data_fingerprint,
        }
        if has_timestamps and n_clean > 0:
            extra["date_range_start"] = str(self._df.index.min())
            extra["date_range_end"] = str(self._df.index.max())
            if interval_stats:
                extra["interval_stats"] = interval_stats

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

        import pandas as pd

        raw = self._df[profile.raw_features].values

        self._baselines = profile.compute_baselines(raw)
        if self._baselines:
            logger.info("Baselines: %s", self._baselines)

        timestamps = (
            self._df.index.values
            if isinstance(self._df.index, pd.DatetimeIndex)
            else None
        )
        if timestamps is not None:
            logger.info("Temporal features enabled — DatetimeIndex detected")
        else:
            logger.warning(
                "No DatetimeIndex found — temporal features will be zero-valued"
            )

        features_enhanced = profile.engineer_features(raw, self._baselines, timestamps=timestamps)
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

        # Detect time gaps and find longest contiguous segment
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
                IssueSeverity.WARNING,
                "windowing",
                f"Detected {gap_info['gaps_found']} time gap(s) across "
                f"{gap_info['segments']} contiguous segments "
                f"(largest gap: {gap_info.get('largest_gap_seconds', 0):.0f}s)",
            )

        # Trim sensor warm-up from the start of each segment
        warmup_secs = self._sensor_profile.warmup_seconds
        if warmup_secs > 0 and len(segments) > 0:
            interval = self._sensor_profile.expected_interval_seconds or 3.0
            warmup_samples = int(warmup_secs / interval)
            trimmed = []
            trimmed_total = 0
            for s, e in segments:
                if (e - s) > warmup_samples:
                    trimmed.append((s + warmup_samples, e))
                    trimmed_total += warmup_samples
                # else: entire segment is within warm-up — skip it
            if trimmed_total > 0:
                logger.info(
                    "Trimmed %d warm-up samples (%.0fs) from %d segment starts",
                    trimmed_total, warmup_secs, len(segments),
                )
                self._report.add(
                    IssueSeverity.INFO,
                    "windowing",
                    f"Trimmed {trimmed_total} warm-up samples "
                    f"({warmup_secs:.0f}s) from segment starts",
                )
            # Only apply trimming if at least one segment survived
            if trimmed:
                segments = trimmed

        # Use all contiguous segments that are long enough for windowing
        valid_segments = [
            (s, e) for s, e in segments if (e - s) >= self._window_size
        ]

        if not valid_segments:
            longest = max(segments, key=lambda s: s[1] - s[0])
            raise InsufficientDataError(
                f"No contiguous segment is long enough for "
                f"window_size={self._window_size} "
                f"(longest has {longest[1] - longest[0]} samples)"
            )

        total_samples = sum(e - s for s, e in valid_segments)
        logger.info(
            "Using %d of %d contiguous segments (%d total samples)",
            len(valid_segments),
            len(segments),
            total_samples,
        )

        # Create sliding windows from all valid segments
        all_X, all_y = [], []
        for start, end in valid_segments:
            seg_X, seg_y = create_sliding_windows(
                self._features_enhanced[start:end],
                self._targets[start:end],
                self._window_size,
            )
            all_X.append(seg_X)
            all_y.append(seg_y)

        self._X = np.concatenate(all_X, axis=0)
        self._y = np.concatenate(all_y, axis=0)

        feature_dim = self._X.shape[1] if self._X.ndim > 1 else self._X.shape[0]
        logger.info(
            "Created %d windows (window_size=%d, feature_dim=%d)",
            len(self._X),
            self._window_size,
            feature_dim,
        )

        extra = {"window_size": self._window_size, "feature_dim": feature_dim}
        extra.update(gap_info)
        extra["valid_segments"] = len(valid_segments)
        extra["total_segment_samples"] = total_samples

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
            len(self._X_train),
            len(self._X_val),
            self._test_size,
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

        # Seed all RNGs before model creation for reproducible weight init
        seed_everything(self._random_state)

        self._model = build_model(
            self._model_type,
            window_size=self._window_size,
            num_features=self._sensor_profile.total_features,
        )

        if hasattr(self._model, "kl_loss"):
            bnn_cfg = settings.get_model_config("bnn")
            self._model._kl_weight = bnn_cfg.get("kl_weight", 1.0)

        if hasattr(self._model, "regularization_loss"):
            model_cfg = settings.get_model_config(self._model_type)
            self._model._reg_weight = model_cfg.get("reg_weight", 1.0)

        tcfg = settings.get_training_config()
        self._tb_log_dir = self._resolve_tensorboard_dir()
        histogram_freq = tcfg.get("tensorboard_histogram_freq", 50)
        checkpoint_freq = tcfg.get("checkpoint_freq", 20)

        # Resolve checkpoint path and model output dir
        base = Path(settings.TRAINED_MODELS_BASE)
        effective_model_dir = (
            self._output_dir if self._output_dir else base / self._model_type
        )
        effective_model_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = None
        if self._resume:
            ckpt_file = effective_model_dir / "checkpoint.pt"
            if ckpt_file.exists():
                checkpoint_path = str(ckpt_file)
            else:
                logger.info("--resume specified but no checkpoint found at %s", ckpt_file)

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
            seed=self._random_state,
            on_epoch=self._on_epoch_callback,
            checkpoint_path=checkpoint_path,
            checkpoint_freq=checkpoint_freq,
            model_dir=str(effective_model_dir),
            early_stopping_patience=self._early_stopping_patience,
        )

        self._training_history = history
        self._training_interrupted = history.get("interrupted", False)
        best_val_loss = history["best_val_loss"]

        logger.info("Best validation loss: %.6f", best_val_loss)

        extra = {
            "epochs": self._epochs,
            "best_val_loss": best_val_loss,
            "device": str(self._device),
        }
        if self._tb_log_dir:
            extra["tensorboard_log_dir"] = self._tb_log_dir
        if self._training_interrupted:
            extra["interrupted"] = True
            extra["epochs_completed"] = len(history["train_losses"])

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

        r2 = self._metrics["r2"]
        if r2 < -1.0:
            raise NegativeR2Error(
                f"Model evaluation produced R²={r2:.4f} (worse than mean predictor)",
                suggestion="Check data quality, reduce learning rate, or increase epochs",
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
                    "early_stopping_patience": self._early_stopping_patience,
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
        self._model_dir = (
            self._output_dir if self._output_dir else base / self._model_type
        )

        # Build data provenance manifest
        manifest = self._build_data_manifest()

        identity = settings.get_sensor_identity()

        save_trained_model(
            model=self._model,
            feature_scaler=self._feature_scaler,
            target_scaler=self._target_scaler,
            model_type=self._model_type,
            window_size=self._window_size,
            baselines=self._baselines,
            sensor_type=self._sensor_profile.name,
            iaq_standard=self._iaq_standard.name,
            sensor_id=identity.get("sensor_id"),
            firmware_version=identity.get("firmware_version"),
            model_dir=str(self._model_dir),
            metrics=self._metrics,
            training_history=self._training_history,
            data_manifest=manifest,
            feature_names=self._sensor_profile.all_feature_names,
        )

        # Compute schema fingerprint for semver
        self._schema_fp = compute_schema_fingerprint(
            sensor_type=self._sensor_profile.name,
            iaq_standard=self._iaq_standard.name,
            window_size=self._window_size,
            num_features=self._sensor_profile.total_features,
            model_type=self._model_type,
        )
        schema_fp = self._schema_fp

        # Update central manifest and get version
        import pandas as pd

        run_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_fingerprint": self._data_fingerprint,
            "model_dir": str(self._model_dir),
            "metrics": {k: float(v) for k, v in self._metrics.items()},
            "git_commit": manifest["git_commit"],
        }
        if identity.get("sensor_id"):
            run_entry["sensor_id"] = identity["sensor_id"]
        if identity.get("firmware_version"):
            run_entry["firmware_version"] = identity["firmware_version"]
        self._version = update_central_manifest(
            self._model_type,
            run_entry,
            schema_fingerprint=schema_fp,
        )
        manifest["version"] = self._version

        # Build Merkle provenance tree
        self._merkle_root = build_merkle_tree_from_context(
            sensor_id=identity.get("sensor_id"),
            firmware_version=identity.get("firmware_version"),
            sensor_type=self._sensor_profile.name,
            source_metadata=self._source.metadata,
            raw_row_count=self._raw_row_count or 0,
            clean_row_count=len(self._df),
            data_fingerprint=self._data_fingerprint or "",
            rows_dropped=(self._raw_row_count or 0) - len(self._df),
            issues_summary=(
                "; ".join(f"[{i.severity.value}] {i.message}" for i in self._report.issues)
                if self._report.issues
                else "none"
            ),
            schema_fingerprint=schema_fp,
            feature_stats=self._feature_stats or {},
            baselines=self._baselines,
            feature_scaler=self._feature_scaler,
            target_scaler=self._target_scaler,
            window_size=self._window_size,
            num_features=self._sensor_profile.total_features,
            sample_count=len(self._X),
            test_size=self._test_size,
            train_samples=len(self._X_train),
            val_samples=len(self._X_val),
            model_state_dict=self._model.state_dict(),
            metrics=self._metrics,
            training_config={
                "epochs": self._epochs,
                "learning_rate": self._learning_rate,
                "batch_size": self._batch_size,
            },
            model_type=self._model_type,
            version=self._version,
        )
        manifest["merkle_tree"] = self._merkle_root.to_dict()
        manifest["merkle_root_hash"] = self._merkle_root.content_hash

        # Patch central MANIFEST.json with merkle_root_hash
        patch_manifest_merkle(
            self._model_type, self._version, self._merkle_root.content_hash
        )

        # Patch config.json with version + schema fingerprint (Option B)
        patch_config_with_version(self._model_dir, self._version, schema_fp)
        patch_config_with_merkle_hash(self._model_dir, self._merkle_root.content_hash)

        # Re-save manifest with version included
        from training.utils import save_data_manifest

        save_data_manifest(manifest, self._model_dir)

        # Save data cleanse report
        report_path = self._model_dir / "data_cleanse_report.json"
        with open(report_path, "w") as f:
            json.dump(self._report.to_dict(), f, indent=2)

        logger.info(
            "Registered as %s in central manifest (merkle=%s)",
            self._version,
            self._merkle_root.content_hash[:16],
        )

        self._artifact_paths: List[Path] = []
        if self._model_dir.exists():
            self._artifact_paths = sorted(
                f for f in self._model_dir.iterdir() if f.is_file()
            )

        return StageResult(
            state=PipelineState.SAVING,
            duration_seconds=time.monotonic() - t0,
            extra={
                "model_dir": str(self._model_dir),
                "version": self._version,
                "artifacts": [f.name for f in self._artifact_paths],
            },
        )

    def collect_run_params(self) -> Dict[str, Any]:
        """Flat dict of all run parameters — ready for experiment tracking.

        Can be called at any point after __init__. Values that depend on
        pipeline execution (data_fingerprint, git_commit, num_features) are
        included only if already computed; missing keys are omitted.
        """
        identity = settings.get_sensor_identity()
        params: Dict[str, Any] = {
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
            "early_stopping_patience": self._early_stopping_patience,
            "sensor_type": self._sensor_profile.name,
            "iaq_standard": self._iaq_standard.name,
            "num_features": self._sensor_profile.total_features,
            "data_source": self._source.name,
            "feature_names": ",".join(self._sensor_profile.all_feature_names),
        }
        # Model-specific hyperparameters
        model_cfg = settings.get_model_config(self._model_type)
        if self._model_type == "bnn":
            params["bnn_prior_sigma"] = model_cfg.get("prior_sigma")
            params["bnn_kl_weight"] = model_cfg.get("kl_weight")
        elif self._model_type == "kan":
            params["kan_reg_weight"] = model_cfg.get("reg_weight")
        if identity.get("sensor_id"):
            params["sensor_id"] = identity["sensor_id"]
        if identity.get("firmware_version"):
            params["firmware_version"] = identity["firmware_version"]
        if self._data_fingerprint:
            params["data_fingerprint"] = self._data_fingerprint
        if self._schema_fp:
            params["schema_fingerprint"] = self._schema_fp
        git_commit = _get_git_commit()
        if git_commit != "unknown":
            params["git_commit"] = git_commit
        return params

    def _build_data_manifest(self) -> dict:
        """Assemble the full data provenance manifest from pipeline state."""
        import pandas as pd

        # Serialize stage results
        stages = []
        for sr in self._fsm.stage_results:
            stages.append(
                {
                    "stage": sr.state.value,
                    "duration_seconds": round(sr.duration_seconds, 3),
                    "rows_in": sr.rows_in,
                    "rows_out": sr.rows_out,
                    "columns": sr.columns,
                    "extra": sr.extra,
                }
            )

        # Serialize preprocessing issues
        issues = []
        for issue in self._report.issues:
            issues.append(
                {
                    "severity": issue.severity.value,
                    "stage": issue.stage,
                    "message": issue.message,
                    "rows_affected": issue.rows_affected,
                }
            )

        run_params = self.collect_run_params()

        return {
            "created_at": pd.Timestamp.now().isoformat(),
            "git_commit": run_params.get("git_commit", _get_git_commit()),
            "data_fingerprint": self._data_fingerprint,
            "data_source": self._source.metadata,
            "config": run_params,
            "feature_statistics": self._feature_stats,
            "target_statistics": self._target_stats,
            "stages": stages,
            "preprocessing_issues": issues,
            "metrics": {k: float(v) for k, v in self._metrics.items()}
            if self._metrics
            else {},
        }

    # ── Orchestrator ───────────────────────────────────────────────────

    def orchestrate(self) -> PipelineResult:
        """Run the full pipeline from IDLE to COMPLETED."""

        def _after_stage(state, result):
            # Log preprocessing report after SCALING stage
            if state == PipelineState.SCALING:
                self._report.log_summary()

        self._fsm.run(
            interrupted=lambda: self._training_interrupted,
            after_stage=_after_stage,
        )

        # If training was interrupted, log and return partial result
        if self._training_interrupted:
            logger.info(
                "Training interrupted after %d epochs — skipping evaluation and saving",
                len(self._training_history["train_losses"]),
            )
            return self._build_result(interrupted=True)

        return self._build_result()

    def _build_result(self, interrupted: bool = False) -> PipelineResult:
        return PipelineResult(
            metrics=self._metrics or {},
            training_history=self._training_history,
            model_dir=self._model_dir,
            stage_results=self._fsm.stage_results,
            preprocessing_report=self._report,
            version=self._version,
            merkle_root_hash=self._merkle_root.content_hash
            if self._merkle_root
            else "",
            artifact_paths=getattr(self, "_artifact_paths", []),
            interrupted=interrupted,
        )
