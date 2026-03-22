# training/train.py
import logging
from pathlib import Path
from typing import Optional

from training.data_sources import DataSource, SyntheticSource
from integrations.mlflow.tracker import ExperimentTracker, NullTracker
from training.pipeline import PipelineError, PipelineResult, TrainingPipeline

logger = logging.getLogger(__name__)


def train_single_model(
    model_type: str,
    epochs: int = 200,
    window_size: Optional[int] = None,
    num_records: int = None,
    data_source: DataSource = None,
    experiment_name: str = "iaq4j",
    resume: bool = False,
    tracker: Optional[ExperimentTracker] = None,
) -> PipelineResult:
    """Train a single model using the TrainingPipeline.

    Args:
        model_type: One of "mlp", "kan", "lstm", "cnn", "bnn".
        epochs: Number of training epochs.
        window_size: Sliding window size. If None, reads from model_config.yaml.
        num_records: Number of synthetic samples (ignored when data_source is provided).
        data_source: Data source to use. Defaults to SyntheticSource.
        experiment_name: Experiment name for tracking.
        resume: If True, resume from last checkpoint if available.
        tracker: Experiment tracker. Defaults to NullTracker.

    Returns:
        PipelineResult on success.

    Raises:
        PipelineError: if any pipeline stage fails.
    """
    if tracker is None:
        tracker = NullTracker()

    if window_size is None:
        from app.config import settings
        window_size = settings.get_model_config(model_type).get("window_size", 10)

    if data_source is None:
        num_samples = num_records if num_records else 1000
        data_source = SyntheticSource(num_samples=num_samples)

    tracker.start_run(model_type, experiment_name)
    try:
        def on_epoch(epoch: int, train_loss: float, val_loss: float, lr: float) -> None:
            tracker.log_epoch(epoch, train_loss, val_loss, lr)

        pipeline = TrainingPipeline(
            source=data_source,
            model_type=model_type,
            epochs=epochs,
            window_size=window_size,
            on_epoch=on_epoch,
            resume=resume,
        )

        result = pipeline.orchestrate()

        if result.interrupted:
            logger.info("Training interrupted — checkpoint saved. Run marked KILLED.")
            tracker.end_run(status="KILLED")
            return result

        # Fix run name now that we have the semver (version is set during SAVING).
        # result.version already includes the model type prefix, e.g. "mlp-1.2.0"
        tracker.log_tags({"mlflow.runName": result.version})

        # Full param set — sensor type, iaq standard, schema fingerprint, data fingerprint, etc.
        tracker.log_params(pipeline.collect_run_params())

        # Final evaluation metrics
        tracker.log_metrics({
            "best_val_loss": result.training_history.get("best_val_loss", 0),
            "mae": result.metrics.get("mae", 0),
            "rmse": result.metrics.get("rmse", 0),
            "r2": result.metrics.get("r2", 0),
        })

        # Provenance tags
        tracker.log_tags({
            "version": result.version,
            "merkle_root": result.merkle_root_hash,
        })

        # Artifacts: scalers and data manifest alongside the pytorch model
        model_dir = Path(result.model_dir)
        for name in ("feature_scaler.pkl", "target_scaler.pkl", "data_manifest.json", "data_cleanse_report.json"):
            path = model_dir / name
            tracker.log_artifact(str(path))

        tracker.log_model(pipeline.model)

        tracker.end_run()
        return result

    except PipelineError as e:
        if e.failure_info:
            tracker.log_tags({"failure_stage": e.failure_info.failed_state.value})
        tracker.end_run(status="FAILED")
        raise
    except Exception:
        tracker.end_run(status="FAILED")
        raise
