"""
Pressure ablation experiment — trains models WITH and WITHOUT pressure
to measure whether barometric pressure improves IAQ prediction accuracy.

Usage:
    python ablation_pressure.py [--epochs 50] [--models mlp,lstm,cnn,bnn]

Results saved to: trained_models/ablation/results.json
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.builtin_profiles import BME680Profile, BME680NoPressureProfile
from app.config import settings
from app.profiles import SensorProfile
from training.data_sources import InfluxDBSource
from training.pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ablation_pressure")


def run_condition(
    condition: str,
    profile: SensorProfile,
    model_types: List[str],
    epochs: int,
    source: InfluxDBSource,
) -> Dict[str, dict]:
    """Train all model types under one experimental condition.

    Returns dict mapping model_type -> metrics dict.
    """
    results = {}
    base_dir = Path("trained_models/ablation") / condition

    for model_type in model_types:
        model_cfg = settings.get_model_config(model_type)
        window_size = model_cfg.get("window_size", 10)
        output_dir = str(base_dir / model_type)

        print(f"\n{'─' * 60}")
        print(f"  [{condition}] {model_type.upper()} (window={window_size}, "
              f"features={len(profile.raw_features)}+{len(profile.engineered_feature_names)})")
        print(f"{'─' * 60}")

        try:
            pipeline = TrainingPipeline(
                source=source,
                model_type=model_type,
                epochs=epochs,
                window_size=window_size,
                output_dir=output_dir,
            )
            # Override the sensor profile for this condition
            pipeline._sensor_profile = profile

            def log_progress(state, result):
                if result and result.extra:
                    print(f"    [{state.value}] {result.extra}")

            pipeline.on_stage_complete(log_progress)
            result = pipeline.orchestrate()

            if result.interrupted:
                print(f"    INTERRUPTED — checkpoint saved")
                results[model_type] = None
            else:
                metrics = result.metrics
                print(f"    MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
                      f"R2={metrics['r2']:.4f}")
                results[model_type] = {
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "best_val_loss": result.training_history.get("best_val_loss"),
                    "version": result.version,
                    "num_features": len(profile.raw_features) + len(profile.engineered_feature_names),
                    "raw_features": profile.raw_features,
                }
        except Exception as e:
            print(f"    FAILED: {e}")
            logger.exception("Training failed for %s/%s", condition, model_type)
            results[model_type] = None

    return results


def print_comparison(with_results: dict, without_results: dict, model_types: List[str]):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("PRESSURE ABLATION RESULTS")
    print("=" * 80)

    header = f"{'Model':<8} │ {'WITH pressure':^30} │ {'WITHOUT pressure':^30} │ {'Delta':^8}"
    print(header)
    print(f"{'':─<8}─┼─{'':─<30}─┼─{'':─<30}─┼─{'':─<8}")

    for mt in model_types:
        w = with_results.get(mt)
        wo = without_results.get(mt)

        if w and wo:
            w_str = f"MAE={w['mae']:.4f}  R2={w['r2']:.4f}"
            wo_str = f"MAE={wo['mae']:.4f}  R2={wo['r2']:.4f}"
            delta_r2 = w["r2"] - wo["r2"]
            sign = "+" if delta_r2 >= 0 else ""
            delta_str = f"{sign}{delta_r2:.4f}"
        elif w:
            w_str = f"MAE={w['mae']:.4f}  R2={w['r2']:.4f}"
            wo_str = "FAILED"
            delta_str = "—"
        elif wo:
            w_str = "FAILED"
            wo_str = f"MAE={wo['mae']:.4f}  R2={wo['r2']:.4f}"
            delta_str = "—"
        else:
            w_str = "FAILED"
            wo_str = "FAILED"
            delta_str = "—"

        print(f"{mt.upper():<8} │ {w_str:^30} │ {wo_str:^30} │ {delta_str:^8}")

    print()
    print("Delta R2 > 0 means pressure HELPS. Delta R2 < 0 means pressure HURTS.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Pressure ablation experiment")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--models", type=str, default="mlp,cnn,bnn,lstm",
                        help="Comma-separated model types (default: mlp,cnn,bnn,lstm)")
    parser.add_argument("--hours-back", type=int, default=168 * 8,
                        help="InfluxDB lookback hours (default: 1344)")
    parser.add_argument("--database", type=str, default="home_study_room_iaq")
    args = parser.parse_args()

    model_types = [m.strip() for m in args.models.split(",")]

    print("=" * 80)
    print("PRESSURE ABLATION EXPERIMENT")
    print(f"  Models:     {', '.join(m.upper() for m in model_types)}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Database:   {args.database}")
    print(f"  Hours back: {args.hours_back}")
    print("=" * 80)

    # Use cached InfluxDB data for both conditions
    source = InfluxDBSource(
        hours_back=args.hours_back,
        database=args.database,
        cache=True,
    )

    profile_with = BME680Profile()
    profile_without = BME680NoPressureProfile()

    print(f"\nWith pressure:    {profile_with.raw_features}")
    print(f"Without pressure: {profile_without.raw_features}")

    start = time.time()

    # Condition 1: WITH pressure (baseline)
    print(f"\n{'=' * 80}")
    print("CONDITION 1: WITH PRESSURE (baseline)")
    print(f"{'=' * 80}")
    with_results = run_condition("with_pressure", profile_with, model_types, args.epochs, source)

    # Condition 2: WITHOUT pressure
    print(f"\n{'=' * 80}")
    print("CONDITION 2: WITHOUT PRESSURE")
    print(f"{'=' * 80}")
    without_results = run_condition("without_pressure", profile_without, model_types, args.epochs, source)

    source.close()
    elapsed = time.time() - start

    # Print comparison
    print_comparison(with_results, without_results, model_types)

    # Save results JSON
    output_dir = Path("trained_models/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "pressure_ablation",
        "timestamp": datetime.utcnow().isoformat(),
        "epochs": args.epochs,
        "database": args.database,
        "elapsed_seconds": round(elapsed, 1),
        "with_pressure": with_results,
        "without_pressure": without_results,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    print(f"Total time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
