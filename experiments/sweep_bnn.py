"""
BNN hyperparameter sweep — finds optimal KL weight, learning rate, and architecture.

Sweep dimensions:
  - kl_weight: [0.0001, 0.0003, 0.001]
  - learning_rate: [0.0003, 0.001]
  - hidden_dims: [[64, 32, 16], [128, 64, 32]]
  - grad_clip: [1.0, None]

Uses no-pressure profile (best from ablation). 30 epochs per trial (enough to
detect divergence). Best config retrained for 100 epochs.

Usage:
    python sweep_bnn.py [--sweep-epochs 30] [--final-epochs 100]
"""
import argparse
import itertools
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from app.builtin_profiles import BME680NoPressureProfile
from app.config import settings
from app.models import build_model
from training.data_sources import ParquetSource
from training.pipeline import TrainingPipeline
from training.utils import get_device, seed_everything, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sweep_bnn")

COMBINED_PARQUET = Path("cache/combined_3month_iaq.parquet")
INFLUX_CACHE = Path("cache/influxdb_a4b58597429641c7.parquet")


SWEEP_GRID = {
    "kl_weight": [0.00005, 0.0001, 0.0003, 0.001],
    "learning_rate": [0.0003, 0.0005, 0.001],
    "hidden_dims": [[64, 32, 16], [128, 64, 32]],
}


def run_trial(
    trial_id: int,
    params: dict,
    source: DataSource,
    profile: BME680NoPressureProfile,
    epochs: int,
) -> dict:
    """Run a single BNN trial with given hyperparameters."""
    output_dir = f"trained_models/sweep_bnn/trial_{trial_id:03d}"

    print(f"\n{'─' * 70}")
    print(f"  Trial {trial_id}: kl={params['kl_weight']}, lr={params['learning_rate']}, "
          f"hidden={params['hidden_dims']}")
    print(f"{'─' * 70}")

    try:
        pipeline = TrainingPipeline(
            source=source,
            model_type="bnn",
            epochs=epochs,
            window_size=10,
            learning_rate=params["learning_rate"],
            output_dir=output_dir,
        )
        pipeline._sensor_profile = profile

        def on_stage(state, result):
            if result and result.extra:
                print(f"    [{state.value}] {result.extra}")

        pipeline.on_stage_complete(on_stage)

        # Patch _do_training to inject custom hidden_dims and kl_weight
        from training.pipeline.types import StageResult, PipelineState
        import types as _types

        def custom_training(self_pipe):
            import time as _time
            t0 = _time.monotonic()

            self_pipe._device = get_device()
            seed_everything(self_pipe._random_state)

            # Build BNN with custom architecture
            from app.models import MODEL_REGISTRY
            ModelCls = MODEL_REGISTRY["bnn"]
            input_dim = self_pipe._window_size * self_pipe._sensor_profile.total_features

            model = ModelCls(
                input_dim=input_dim,
                hidden_dims=params["hidden_dims"],
                prior_sigma=1.0,
            )
            model._kl_weight = params["kl_weight"]
            self_pipe._model = model

            print(f"\n    Training BNN on {self_pipe._device} "
                  f"(kl={params['kl_weight']}, lr={params['learning_rate']}, "
                  f"hidden={params['hidden_dims']})...")

            tcfg = settings.get_training_config()
            tb_dir = None
            if tcfg.get("tensorboard_enabled", False):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                tb_dir = str(Path(tcfg.get("tensorboard_log_dir", "runs")) / f"bnn_sweep_{trial_id:03d}_{ts}")

            train_result = train_model(
                model=model,
                model_name="bnn",
                X_train=self_pipe._X_train,
                y_train=self_pipe._y_train,
                X_val=self_pipe._X_val,
                y_val=self_pipe._y_val,
                epochs=self_pipe._epochs,
                batch_size=self_pipe._batch_size,
                learning_rate=params["learning_rate"],
                lr_scheduler_patience=self_pipe._lr_scheduler_patience,
                lr_scheduler_factor=self_pipe._lr_scheduler_factor,
                device=self_pipe._device,
                on_epoch=self_pipe._on_epoch_callback,
                log_dir=tb_dir,
                checkpoint_freq=tcfg.get("checkpoint_freq", 20),
                model_dir=str(self_pipe._output_dir),
            )

            self_pipe._training_result = train_result
            self_pipe._tb_log_dir = tb_dir

            if train_result.get("interrupted"):
                return StageResult(
                    state=PipelineState.TRAINING,
                    duration_seconds=_time.monotonic() - t0,
                    rows_in=len(self_pipe._X_train),
                    rows_out=len(self_pipe._X_train),
                    extra={"interrupted": True},
                )

            best_val = train_result.get("best_val_loss", 0)
            print(f"    Best validation loss: {best_val:.6f}")

            return StageResult(
                state=PipelineState.TRAINING,
                duration_seconds=_time.monotonic() - t0,
                rows_in=len(self_pipe._X_train),
                rows_out=len(self_pipe._X_train),
                extra={
                    "device": str(self_pipe._device),
                    "best_val_loss": best_val,
                    "final_train_loss": train_result["train_loss"][-1] if train_result.get("train_loss") else None,
                },
            )

        pipeline._do_training = _types.MethodType(custom_training, pipeline)

        result = pipeline.orchestrate()

        if result.interrupted:
            return {"trial_id": trial_id, "params": params, "status": "interrupted"}

        metrics = result.metrics
        history = result.training_history
        val_losses = history.get("val_loss", [])

        # Detect divergence: did val loss explode?
        diverged = False
        if val_losses:
            best_val = min(val_losses)
            best_epoch = val_losses.index(best_val)
            final_val = val_losses[-1]
            diverged = final_val > best_val * 10  # 10x worse than best

        print(f"    MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}  "
              f"best_epoch={best_epoch}  diverged={diverged}")

        return {
            "trial_id": trial_id,
            "params": params,
            "status": "diverged" if diverged else "ok",
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "final_val_loss": final_val,
            "diverged": diverged,
            "version": result.version,
        }

    except Exception as e:
        print(f"    FAILED: {e}")
        logger.exception("Trial %d failed", trial_id)
        return {"trial_id": trial_id, "params": params, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="BNN hyperparameter sweep")
    parser.add_argument("--sweep-epochs", type=int, default=30)
    parser.add_argument("--final-epochs", type=int, default=100)
    args = parser.parse_args()

    if COMBINED_PARQUET.exists():
        data_path = COMBINED_PARQUET
    elif INFLUX_CACHE.exists():
        data_path = INFLUX_CACHE
    else:
        print("ERROR: No cached data found.")
        return

    source = ParquetSource(data_path)
    profile = BME680NoPressureProfile()

    # Generate all combinations
    keys = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    combos = list(itertools.product(*values))
    param_list = [dict(zip(keys, combo)) for combo in combos]

    print("=" * 70)
    print("BNN HYPERPARAMETER SWEEP")
    print(f"  Data:         {data_path}")
    print(f"  Sweep epochs: {args.sweep_epochs}")
    print(f"  Final epochs: {args.final_epochs}")
    print(f"  Trials:       {len(param_list)}")
    print(f"  Grid:")
    for k, v in SWEEP_GRID.items():
        print(f"    {k}: {v}")
    print("=" * 70)

    start = time.time()
    results = []

    for i, params in enumerate(param_list):
        trial_result = run_trial(i, params, source, profile, args.sweep_epochs)
        results.append(trial_result)

    # Rank by R² (only non-failed, non-diverged)
    valid = [r for r in results if r.get("status") == "ok"]
    valid.sort(key=lambda r: r["r2"], reverse=True)

    print(f"\n{'=' * 70}")
    print("SWEEP RESULTS (ranked by R²)")
    print(f"{'=' * 70}")
    print(f"{'#':<4} {'kl_weight':>10} {'lr':>8} {'hidden':>18} │ {'R²':>8} {'MAE':>8} {'Best':>6} {'Status':>10}")
    print(f"{'':─<4}─{'':─>10}─{'':─>8}─{'':─>18}─┼─{'':─>8}─{'':─>8}─{'':─>6}─{'':─>10}")

    for r in results:
        p = r["params"]
        hidden_str = str(p["hidden_dims"])
        if r["status"] in ("ok", "diverged"):
            r2_str = f"{r['r2']:.4f}"
            mae_str = f"{r['mae']:.2f}"
            best_str = f"e{r['best_epoch']}"
        else:
            r2_str = "—"
            mae_str = "—"
            best_str = "—"
        print(f"{r['trial_id']:<4} {p['kl_weight']:>10.5f} {p['learning_rate']:>8.4f} "
              f"{hidden_str:>18} │ {r2_str:>8} {mae_str:>8} {best_str:>6} {r['status']:>10}")

    # Retrain best config for more epochs
    if valid:
        best = valid[0]
        print(f"\n{'=' * 70}")
        print(f"BEST CONFIG: Trial {best['trial_id']}")
        print(f"  kl_weight={best['params']['kl_weight']}, lr={best['params']['learning_rate']}, "
              f"hidden={best['params']['hidden_dims']}")
        print(f"  Sweep: R²={best['r2']:.4f}, MAE={best['mae']:.2f}, best_epoch={best['best_epoch']}")
        print(f"\nRetraining for {args.final_epochs} epochs...")
        print(f"{'=' * 70}")

        final = run_trial(999, best["params"], source, profile, args.final_epochs)
        if final.get("r2") is not None:
            print(f"\nFINAL RESULT: R²={final['r2']:.4f}, MAE={final['mae']:.2f}")
    else:
        print("\nNo valid trials — all failed or diverged.")
        final = None

    elapsed = time.time() - start

    # Save
    output_dir = Path("trained_models/sweep_bnn")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "bnn_hyperparameter_sweep",
        "timestamp": datetime.utcnow().isoformat(),
        "sweep_epochs": args.sweep_epochs,
        "final_epochs": args.final_epochs,
        "grid": {k: [str(v) for v in vs] for k, vs in SWEEP_GRID.items()},
        "trials": results,
        "best_trial": valid[0] if valid else None,
        "final_retrain": final,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir / 'results.json'}")
    print(f"Total time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
