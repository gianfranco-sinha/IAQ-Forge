"""
Train all models (MLP, CNN, KAN) from collected BSEC data.
"""
import logging
from pathlib import Path

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
    print("TRAINING ALL MODELS: MLP (Baseline), CNN, KAN")
    print("=" * 70)

    try:
        tracker = MLflowTracker()
    except ImportError:
        tracker = NullTracker()

    results = {}

    for model_type in ["mlp", "cnn", "kan"]:
        print(f"\n{'─' * 70}")
        print(f"Training {model_type.upper()}...")
        print(f"{'─' * 70}")

        try:
            source = InfluxDBSource(hours_back=168 * 8)

            tracker.start_run(model_type)

            def on_epoch(epoch, train_loss, val_loss, lr):
                tracker.log_epoch(epoch, train_loss, val_loss, lr)

            pipeline = TrainingPipeline(source, model_type=model_type, epochs=200,
                                        on_epoch=on_epoch, resume=True)
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
        label = f"{name.upper()} (Baseline)" if name == "mlp" else f"{name.upper()}"
        pad = max(0, 18 - len(label))
        print(f"  {label}:{' ' * pad}MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")
    print("\nRestart service:")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
