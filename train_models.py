"""
Train all IAQ models (MLP, KAN, BNN, CNN, LSTM) from collected BSEC data.
"""
import logging
from pathlib import Path

from app.config import settings
from training.data_sources import InfluxDBSource
from integrations.mlflow.tracker import MLflowTracker, NullTracker
from training.pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)


def log_progress(state, result):
    if result and result.extra:
        print(f"  [{state.value}] {result.extra}")


if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING MLP, CNN, BNN, LSTM, KAN MODELS FROM BSEC DATA")
    print("=" * 70)

    try:
        tracker = MLflowTracker()
    except ImportError:
        tracker = NullTracker()

    results = {}

    for model_type in ["mlp", "kan", "bnn", "cnn", "lstm"]:
        model_cfg = settings.get_model_config(model_type)
        window_size = model_cfg.get("window_size", 10)

        print(f"\n{'─' * 70}")
        print(f"Training {model_type.upper()} (window_size={window_size})...")
        print(f"{'─' * 70}")

        try:
            source = InfluxDBSource(hours_back=168 * 8, database="home_study_room_iaq")

            tracker.start_run(model_type)

            def on_epoch(epoch, train_loss, val_loss, lr):
                tracker.log_epoch(epoch, train_loss, val_loss, lr)

            pipeline = TrainingPipeline(
                source, model_type=model_type, epochs=200, window_size=window_size,
                on_epoch=on_epoch, resume=True,
            )
            pipeline.on_stage_complete(log_progress)

            result = pipeline.orchestrate()
            if result.interrupted:
                print(f"⏸️  {model_type.upper()} interrupted — checkpoint saved")
                tracker.end_run(status="KILLED")
                results[model_type] = None
            else:
                tracker.log_params(pipeline.collect_run_params())
                tracker.log_metrics({
                    "best_val_loss": result.training_history.get("best_val_loss", 0),
                    "mae": result.metrics.get("mae", 0),
                    "rmse": result.metrics.get("rmse", 0),
                    "r2": result.metrics.get("r2", 0),
                })
                tracker.log_tags({
                    "version": result.version,
                    "merkle_root": result.merkle_root_hash,
                })
                model_dir = Path(result.model_dir)
                for name in ("feature_scaler.pkl", "target_scaler.pkl", "data_manifest.json", "data_cleanse_report.json"):
                    tracker.log_artifact(str(model_dir / name))
                tracker.log_model(pipeline.model)
                tracker.end_run()
                results[model_type] = result.metrics
            source.close()
        except Exception as e:
            print(f"❌ {model_type.upper()} training failed: {e}")
            tracker.end_run(status="FAILED")
            results[model_type] = None

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModel Comparison:")
    for name, metrics in results.items():
        if metrics:
            print(f"  {name.upper()}: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")
        else:
            print(f"  {name.upper()}: FAILED")
    print("\nRestart your service to use the new models:")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
