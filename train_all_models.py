"""
Train all models (MLP, CNN, KAN) from collected BSEC data.
"""
import logging

from training.data_sources import InfluxDBSource
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

    results = {}

    for model_type in ["mlp", "cnn", "kan"]:
        print(f"\n{'─' * 70}")
        print(f"Training {model_type.upper()}...")
        print(f"{'─' * 70}")

        source = InfluxDBSource(hours_back=168 * 8)
        pipeline = TrainingPipeline(source, model_type=model_type, epochs=200)
        pipeline.on_stage_complete(log_progress)

        result = pipeline.orchestrate()
        results[model_type] = result.metrics
        source.close()

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
