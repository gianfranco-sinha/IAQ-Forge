"""
Drift compensation ablation experiment.

Trains models on two conditions:
  1. Raw VOC resistance (no compensation)
  2. T/H-compensated VOC resistance (regression residuals)

Uses 3-month combined dataset (Nov 2025 – Feb 2026, ~2.5M rows).
Held-out evaluation window: last 2 weeks (Feb 12 – Feb 26 2026).

Usage:
    python ablation_drift_compensation.py [--epochs 50] [--models mlp,cnn,bnn,lstm]

Results saved to: trained_models/ablation_drift/results.json
"""
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.builtin_profiles import BME680Profile
from app.config import settings
from app.profiles import SensorProfile
from training.data_sources import ParquetSource
from training.pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ablation_drift")

COMBINED_PARQUET = Path("cache/combined_3month_iaq.parquet")
EVAL_CUTOFF = pd.Timestamp("2026-02-12", tz="UTC")


def fit_voc_compensation(df: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Fit OLS: log(voc_resistance) ~ 1 + temperature + rel_humidity.

    Returns:
        (beta, r2) — beta is [intercept, temp_coeff, hum_coeff], r2 is model fit.
    """
    mask = (
        df["voc_resistance"].notna()
        & df["temperature"].notna()
        & df["rel_humidity"].notna()
        & (df["voc_resistance"] > 0)
    )
    clean = df[mask]

    y = np.log(clean["voc_resistance"].values)
    X = np.column_stack([
        np.ones(len(clean)),
        clean["temperature"].values,
        clean["rel_humidity"].values,
    ])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - y_pred) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return beta, r2


def apply_voc_compensation(df: pd.DataFrame, beta: np.ndarray) -> pd.DataFrame:
    """Replace voc_resistance with T/H-compensated residuals (in resistance space).

    The compensated value = exp(residual) where residual = log(voc) - predicted_log(voc).
    This removes the T/H-driven component, isolating true VOC/IAQ signal.
    """
    df = df.copy()
    mask = (
        df["voc_resistance"].notna()
        & df["temperature"].notna()
        & df["rel_humidity"].notna()
        & (df["voc_resistance"] > 0)
    )

    log_voc = np.log(df.loc[mask, "voc_resistance"].values)
    X = np.column_stack([
        np.ones(mask.sum()),
        df.loc[mask, "temperature"].values,
        df.loc[mask, "rel_humidity"].values,
    ])
    predicted = X @ beta
    residuals = log_voc - predicted

    # Convert back to resistance-like scale: exp(residual) * exp(mean_predicted)
    # This preserves the magnitude while removing T/H effects
    mean_predicted = predicted.mean()
    df.loc[mask, "voc_resistance"] = np.exp(residuals + mean_predicted)

    return df


def run_condition(
    condition: str,
    source: DataSource,
    model_types: List[str],
    epochs: int,
) -> Dict[str, dict]:
    """Train all model types under one condition."""
    results = {}
    base_dir = Path("trained_models/ablation_drift") / condition

    for model_type in model_types:
        model_cfg = settings.get_model_config(model_type)
        window_size = model_cfg.get("window_size", 10)
        output_dir = str(base_dir / model_type)

        print(f"\n{'─' * 60}")
        print(f"  [{condition}] {model_type.upper()} (window={window_size})")
        print(f"{'─' * 60}")

        try:
            pipeline = TrainingPipeline(
                source=source,
                model_type=model_type,
                epochs=epochs,
                window_size=window_size,
                output_dir=output_dir,
            )

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
                }
        except Exception as e:
            print(f"    FAILED: {e}")
            logger.exception("Training failed for %s/%s", condition, model_type)
            results[model_type] = None

    return results


def evaluate_on_holdout(
    condition: str,
    holdout_df: pd.DataFrame,
    model_types: List[str],
) -> Dict[str, dict]:
    """Evaluate trained models on held-out data."""
    from app.models import build_model
    from training.utils import evaluate_model, create_sliding_windows, find_contiguous_segments
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import torch
    import pickle

    profile = BME680Profile()
    results = {}
    base_dir = Path("trained_models/ablation_drift") / condition

    for model_type in model_types:
        model_dir = base_dir / model_type
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            results[model_type] = None
            continue

        model_cfg = settings.get_model_config(model_type)
        window_size = model_cfg.get("window_size", 10)

        try:
            # Load scalers
            with open(model_dir / "feature_scaler.pkl", "rb") as f:
                feature_scaler = pickle.load(f)
            with open(model_dir / "target_scaler.pkl", "rb") as f:
                target_scaler = pickle.load(f)

            # Feature engineering on holdout
            raw = holdout_df[profile.raw_features].values
            timestamps = holdout_df.index.values if isinstance(holdout_df.index, pd.DatetimeIndex) else None
            baselines = profile.compute_baselines(raw, timestamps)
            features = profile.engineer_features(raw, baselines, timestamps=timestamps)
            targets = holdout_df["iaq"].values

            # Windowing
            segments, _gap_info = find_contiguous_segments(holdout_df.index)
            X_windows, y_windows = [], []
            for start_idx, end_idx in segments:
                if (end_idx - start_idx) < window_size:
                    continue
                seg_X = features[start_idx:end_idx]
                seg_y = targets[start_idx:end_idx]
                wx, wy = create_sliding_windows(seg_X, seg_y, window_size)
                X_windows.append(wx)
                y_windows.append(wy)

            if not X_windows:
                print(f"    [{condition}] {model_type.upper()}: no valid windows in holdout")
                results[model_type] = None
                continue

            X_all = np.concatenate(X_windows)
            y_all = np.concatenate(y_windows)

            # Scale
            n_samples, ws, nf = X_all.shape
            X_flat = X_all.reshape(-1, nf)
            X_scaled = feature_scaler.transform(X_flat).reshape(n_samples, ws, nf)
            y_scaled = target_scaler.transform(y_all.reshape(-1, 1)).flatten()

            # Flatten for model input
            X_input = X_scaled.reshape(n_samples, -1)

            # Load model
            num_features = profile.total_features
            model = build_model(model_type, window_size=window_size, num_features=num_features)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            # Predict
            X_tensor = torch.FloatTensor(X_input)
            with torch.no_grad():
                y_pred_scaled = model(X_tensor).numpy().flatten()

            # Inverse scale
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = y_all

            # Metrics
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            ss_res = ((y_true - y_pred) ** 2).sum()
            ss_tot = ((y_true - y_true.mean()) ** 2).sum()
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

            print(f"    [{condition}] {model_type.upper()} holdout: MAE={mae:.4f}  R2={r2:.4f}  (n={n_samples})")
            results[model_type] = {"mae": mae, "rmse": rmse, "r2": r2, "n_samples": n_samples}

        except Exception as e:
            print(f"    [{condition}] {model_type.upper()} holdout FAILED: {e}")
            logger.exception("Holdout eval failed for %s/%s", condition, model_type)
            results[model_type] = None

    return results


def print_comparison(
    raw_train: dict, comp_train: dict,
    raw_holdout: dict, comp_holdout: dict,
    model_types: List[str],
):
    """Print comparison tables."""
    print("\n" + "=" * 90)
    print("DRIFT COMPENSATION ABLATION — TRAINING METRICS")
    print("=" * 90)
    header = f"{'Model':<8} │ {'Raw VOC (val split)':^30} │ {'Compensated VOC (val split)':^30} │ {'Delta R2':^8}"
    print(header)
    print(f"{'':─<8}─┼─{'':─<30}─┼─{'':─<30}─┼─{'':─<8}")

    for mt in model_types:
        r = raw_train.get(mt)
        c = comp_train.get(mt)
        r_str = f"MAE={r['mae']:.4f}  R2={r['r2']:.4f}" if r else "FAILED"
        c_str = f"MAE={c['mae']:.4f}  R2={c['r2']:.4f}" if c else "FAILED"
        if r and c:
            d = c["r2"] - r["r2"]
            d_str = f"{'+' if d >= 0 else ''}{d:.4f}"
        else:
            d_str = "—"
        print(f"{mt.upper():<8} │ {r_str:^30} │ {c_str:^30} │ {d_str:^8}")

    print("\n" + "=" * 90)
    print("DRIFT COMPENSATION ABLATION — HOLDOUT METRICS (last 2 weeks)")
    print("=" * 90)
    header = f"{'Model':<8} │ {'Raw VOC (holdout)':^30} │ {'Compensated VOC (holdout)':^30} │ {'Delta R2':^8}"
    print(header)
    print(f"{'':─<8}─┼─{'':─<30}─┼─{'':─<30}─┼─{'':─<8}")

    for mt in model_types:
        r = raw_holdout.get(mt) if raw_holdout else None
        c = comp_holdout.get(mt) if comp_holdout else None
        r_str = f"MAE={r['mae']:.4f}  R2={r['r2']:.4f}" if r else "FAILED"
        c_str = f"MAE={c['mae']:.4f}  R2={c['r2']:.4f}" if c else "FAILED"
        if r and c:
            d = c["r2"] - r["r2"]
            d_str = f"{'+' if d >= 0 else ''}{d:.4f}"
        else:
            d_str = "—"
        print(f"{mt.upper():<8} │ {r_str:^30} │ {c_str:^30} │ {d_str:^8}")

    print()
    print("Delta R2 > 0 means compensation HELPS.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Drift compensation ablation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--models", type=str, default="mlp,cnn,bnn,lstm")
    parser.add_argument("--eval-cutoff", type=str, default="2026-02-12",
                        help="Holdout eval start date (default: 2026-02-12)")
    args = parser.parse_args()

    model_types = [m.strip() for m in args.models.split(",")]
    eval_cutoff = pd.Timestamp(args.eval_cutoff, tz="UTC")

    if not COMBINED_PARQUET.exists():
        print(f"ERROR: {COMBINED_PARQUET} not found. Run the data fetch first.")
        return

    # Load full dataset
    print("Loading combined 3-month dataset...")
    full_df = pd.read_parquet(COMBINED_PARQUET)
    print(f"  {len(full_df)} rows, {full_df.index[0]} to {full_df.index[-1]}")

    # Filter iaq_accuracy >= 2
    if "iaq_accuracy" in full_df.columns:
        before = len(full_df)
        full_df = full_df[full_df["iaq_accuracy"] >= 2]
        print(f"  After iaq_accuracy >= 2 filter: {len(full_df)} rows (dropped {before - len(full_df)})")

    # Split into train and holdout
    train_df = full_df[full_df.index < eval_cutoff].copy()
    holdout_df = full_df[full_df.index >= eval_cutoff].copy()
    print(f"  Train: {len(train_df)} rows (up to {eval_cutoff.date()})")
    print(f"  Holdout: {len(holdout_df)} rows (from {eval_cutoff.date()})")

    # Fit VOC compensation model on training data only
    print("\nFitting VOC T/H compensation model on training data...")
    beta, r2 = fit_voc_compensation(train_df)
    print(f"  log(voc) ~ {beta[1]:+.4f}*temp {beta[2]:+.4f}*hum (R2={r2:.3f})")
    print(f"  T/H explains {r2*100:.1f}% of VOC variance")

    # Create compensated training data
    train_compensated = apply_voc_compensation(train_df, beta)
    # Apply same compensation to holdout (using coefficients from training)
    holdout_compensated = apply_voc_compensation(holdout_df, beta)

    # Save compensated parquets for the pipeline to use
    comp_train_path = Path("cache/train_compensated.parquet")
    comp_holdout_path = Path("cache/holdout_compensated.parquet")
    raw_train_path = Path("cache/train_raw.parquet")
    raw_holdout_path = Path("cache/holdout_raw.parquet")

    train_df.to_parquet(raw_train_path)
    holdout_df.to_parquet(raw_holdout_path)
    train_compensated.to_parquet(comp_train_path)
    holdout_compensated.to_parquet(comp_holdout_path)

    print(f"\n{'=' * 90}")
    print(f"DRIFT COMPENSATION ABLATION EXPERIMENT")
    print(f"  Models:     {', '.join(m.upper() for m in model_types)}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Train:      {len(train_df)} rows (Nov 2025 – {eval_cutoff.date()})")
    print(f"  Holdout:    {len(holdout_df)} rows ({eval_cutoff.date()} – Feb 26)")
    print(f"  VOC comp:   R2={r2:.3f}")
    print(f"{'=' * 90}")

    start = time.time()

    # Condition 1: Raw VOC
    print(f"\n{'=' * 90}")
    print("CONDITION 1: RAW VOC RESISTANCE")
    print(f"{'=' * 90}")
    source_raw = ParquetSource(raw_train_path)
    raw_results = run_condition("raw_voc", source_raw, model_types, args.epochs)

    # Condition 2: Compensated VOC
    print(f"\n{'=' * 90}")
    print("CONDITION 2: T/H-COMPENSATED VOC RESISTANCE")
    print(f"{'=' * 90}")
    source_comp = ParquetSource(comp_train_path)
    comp_results = run_condition("compensated_voc", source_comp, model_types, args.epochs)

    # Holdout evaluation
    print(f"\n{'=' * 90}")
    print("HOLDOUT EVALUATION (last 2 weeks)")
    print(f"{'=' * 90}")
    raw_holdout_results = evaluate_on_holdout("raw_voc", holdout_df, model_types)
    comp_holdout_results = evaluate_on_holdout("compensated_voc", holdout_compensated, model_types)

    elapsed = time.time() - start

    # Print comparison
    print_comparison(raw_results, comp_results, raw_holdout_results, comp_holdout_results, model_types)

    # Save results
    output_dir = Path("trained_models/ablation_drift")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "drift_compensation_ablation",
        "timestamp": datetime.utcnow().isoformat(),
        "epochs": args.epochs,
        "eval_cutoff": str(eval_cutoff),
        "train_rows": len(train_df),
        "holdout_rows": len(holdout_df),
        "voc_compensation_r2": r2,
        "voc_compensation_beta": beta.tolist(),
        "elapsed_seconds": round(elapsed, 1),
        "training": {
            "raw_voc": raw_results,
            "compensated_voc": comp_results,
        },
        "holdout": {
            "raw_voc": raw_holdout_results,
            "compensated_voc": comp_holdout_results,
        },
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    print(f"Total time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
