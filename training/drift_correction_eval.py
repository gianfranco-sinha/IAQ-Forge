"""Evaluate drift correction impact on model accuracy.

Loads trained models, applies per-reading drift correction based on
sensor age (estimated from timestamps vs. drift summary date_start),
and compares metrics (MAE, RMSE, R2) across 5 models x 3 modes
(uncorrected, linear, VOC-compensated).
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.config import settings
import app.builtin_profiles  # noqa: F401
from app.profiles import get_iaq_standard, get_sensor_profile
from training.drift_correction import (
    apply_linear_correction,
    apply_voc_compensated_correction,
    compute_sensor_age_days,
    load_drift_summary,
)
from training.utils import (
    create_sliding_windows,
    find_contiguous_segments,
    get_device,
)

logger = logging.getLogger("training.drift_correction_eval")

MODEL_TYPES = ["mlp", "kan", "lstm", "cnn", "bnn"]
CORRECTION_MODES = ["uncorrected", "linear", "voc_compensated"]


def _load_predictor(model_type: str):
    """Load a trained IAQPredictor for the given model type."""
    from app.models import IAQPredictor

    model_dir = Path(settings.TRAINED_MODELS_BASE) / model_type
    if not model_dir.exists():
        return None

    window_size = settings.get_model_config(model_type).get("window_size", 10)
    predictor = IAQPredictor(
        model_type=model_type,
        window_size=window_size,
        model_path=str(model_dir),
    )
    if not predictor.load_model(str(model_dir)):
        return None
    return predictor


def _fetch_data(
    data_source: str,
    num_samples: int = 2000,
    database: Optional[str] = None,
    cache: bool = False,
) -> pd.DataFrame:
    """Fetch evaluation data from the specified source."""
    if data_source == "influxdb":
        from training.data_sources import InfluxDBSource

        source = InfluxDBSource(database=database, cache=cache)
        source.validate()
        df = source.fetch()
        source.close()
        return df
    else:
        from training.data_sources import SyntheticSource

        source = SyntheticSource(num_samples=num_samples)
        return source.fetch()


def _apply_correction_to_df(
    df: pd.DataFrame,
    raw_features: List[str],
    mode: str,
    sensor_start: pd.Timestamp,
    coefficients: dict,
) -> pd.DataFrame:
    """Apply vectorized drift correction using each reading's timestamp.

    Each row's sensor age = row_timestamp - sensor_start (from drift summary).
    """
    if mode == "uncorrected":
        return df

    import math

    corrected = df.copy()

    # Compute per-row age in days (vectorized)
    if isinstance(df.index, pd.DatetimeIndex):
        age_days = (df.index - sensor_start).total_seconds() / 86400.0
        age_days = np.maximum(age_days, 0.0)
    else:
        age_days = np.zeros(len(df))

    for feat in raw_features:
        if feat not in corrected.columns:
            continue

        if mode == "linear":
            coeff = coefficients.get(feat)
            if coeff is not None:
                corrected[feat] = corrected[feat] - coeff.trend_slope_per_day * age_days
        else:  # voc_compensated
            if feat == "voc_resistance":
                comp_coeff = coefficients.get("voc_resistance_compensated")
                if comp_coeff is not None:
                    corrected[feat] = corrected[feat] * np.exp(
                        -comp_coeff.trend_slope_per_day * age_days
                    )
            else:
                coeff = coefficients.get(feat)
                if coeff is not None:
                    corrected[feat] = (
                        corrected[feat] - coeff.trend_slope_per_day * age_days
                    )

    return corrected


def _process_and_evaluate(
    df: pd.DataFrame,
    model,
    model_type: str,
    profile,
    standard,
    window_size: int,
) -> Dict[str, float]:
    """Run feature engineering, windowing, scaling, and evaluation on a DataFrame."""
    raw = df[profile.raw_features].values
    baselines = profile.compute_baselines(raw)

    timestamps = df.index.values if isinstance(df.index, pd.DatetimeIndex) else None
    features = profile.engineer_features(raw, baselines, timestamps=timestamps)
    targets = df[standard.target_column].values

    # Windowing across contiguous segments
    segments, _ = find_contiguous_segments(df.index)
    valid_segments = [(s, e) for s, e in segments if (e - s) >= window_size]

    if not valid_segments:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    all_X, all_y = [], []
    for start, end in valid_segments:
        seg_features = features[start:end]
        seg_targets = targets[start:end]
        X_seg, y_seg = create_sliding_windows(seg_features, seg_targets, window_size)
        if len(X_seg) > 0:
            all_X.append(X_seg)
            all_y.append(y_seg)

    if not all_X:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    # Chronological split — use validation portion
    split_idx = int(len(X) * 0.8)
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    if len(X_val) < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    # Scale using freshly-fit scalers (same procedure as training pipeline)
    X_train = X[:split_idx]
    y_train = y[:split_idx]

    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    feature_scaler.fit(X_train)
    target_scaler.fit(y_train.reshape(-1, 1))

    X_val_scaled = feature_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

    device = get_device()
    return _batched_evaluate(model, X_val_scaled, y_val_scaled, target_scaler, device)


def _batched_evaluate(model, X_val, y_val, target_scaler, device, batch_size=4096):
    """Evaluate model in batches to avoid OOM on large datasets."""
    model.eval()
    model = model.to(device)

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            batch = torch.FloatTensor(X_val[i : i + batch_size]).to(device)
            preds = model(batch).cpu().numpy()
            all_preds.append(preds)

    predictions = np.concatenate(all_preds)
    y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(predictions).flatten()

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def run_evaluation(
    data_source: str = "synthetic",
    drift_summary_path: Optional[str] = None,
    output_path: Optional[str] = None,
    num_samples: int = 2000,
    database: Optional[str] = None,
    cache: bool = False,
) -> Dict:
    """Run drift correction evaluation across all models and modes.

    Sensor age is estimated per-reading from the drift summary's date_start
    (when the sensor began collecting data) and each reading's timestamp.

    Returns:
        Dict with results keyed by model_type -> mode -> metrics.
    """
    # Load drift summary (coefficients + sensor start date)
    summary = load_drift_summary(drift_summary_path)
    sensor_start = summary.date_start
    coefficients = summary.coefficients
    logger.info("Loaded drift coefficients for %d features", len(coefficients))

    # Fetch data
    print(f"Fetching {data_source} data...")
    df = _fetch_data(
        data_source, num_samples=num_samples, database=database, cache=cache
    )
    print(f"  {len(df)} samples loaded")

    profile = get_sensor_profile()
    standard = get_iaq_standard()

    # Report sensor age range from timestamps
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        age_start = compute_sensor_age_days(sensor_start, pd.Timestamp(df.index[0]))
        age_end = compute_sensor_age_days(sensor_start, pd.Timestamp(df.index[-1]))
        print(f"  Sensor start: {sensor_start.date()}")
        print(f"  Data range: {df.index[0].date()} to {df.index[-1].date()}")
        print(
            f"  Sensor age range: {age_start:.0f} to {age_end:.0f} days "
            f"({age_start / 365.25:.1f} to {age_end / 365.25:.1f} yr)"
        )
    else:
        age_start = 0.0
        age_end = 0.0
        print("  No DatetimeIndex — sensor age defaults to 0 (no correction applied)")

    results = {}

    for model_type in MODEL_TYPES:
        predictor = _load_predictor(model_type)
        if predictor is None:
            print(f"  {model_type.upper()}: not found, skipping")
            continue

        window_size = settings.get_model_config(model_type).get("window_size", 10)
        model_results = {}

        for mode in CORRECTION_MODES:
            corrected_df = _apply_correction_to_df(
                df, profile.raw_features, mode, sensor_start, coefficients
            )
            metrics = _process_and_evaluate(
                corrected_df,
                predictor.model,
                model_type,
                profile,
                standard,
                window_size,
            )
            model_results[mode] = metrics

        results[model_type] = model_results

    # Format and print table
    _print_results_table(results, sensor_start, df)

    # Save JSON
    out_path = output_path or "results/drift_correction_eval.json"
    _save_results(results, out_path, sensor_start, df, data_source)

    return results


def _print_results_table(results: dict, sensor_start: pd.Timestamp, df: pd.DataFrame):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 80}")
    print("Drift Correction Evaluation Results")
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        age_end = compute_sensor_age_days(sensor_start, pd.Timestamp(df.index[-1]))
        print(
            f"Sensor start: {sensor_start.date()}, "
            f"last reading: {df.index[-1].date()}, "
            f"max age: {age_end:.0f} days ({age_end / 365.25:.1f} yr)"
        )
    print(f"{'=' * 80}")

    header = f"{'Model':<6} {'Mode':<18} {'MAE':>8} {'RMSE':>8} {'R2':>10}"
    print(header)
    print("-" * len(header))

    for model_type in MODEL_TYPES:
        if model_type not in results:
            continue
        for mode in CORRECTION_MODES:
            m = results[model_type].get(mode, {})
            mae = (
                f"{m['mae']:.4f}" if not np.isnan(m.get("mae", float("nan"))) else "N/A"
            )
            rmse = (
                f"{m['rmse']:.4f}"
                if not np.isnan(m.get("rmse", float("nan")))
                else "N/A"
            )
            r2 = f"{m['r2']:.6f}" if not np.isnan(m.get("r2", float("nan"))) else "N/A"
            print(f"{model_type:<6} {mode:<18} {mae:>8} {rmse:>8} {r2:>10}")
        print()


def _save_results(results: dict, output_path: str, sensor_start, df, data_source):
    """Save evaluation results to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "sensor_start": str(sensor_start.date()),
        "data_source": data_source,
    }

    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        payload["data_start"] = str(df.index[0].date())
        payload["data_end"] = str(df.index[-1].date())
        age_end = compute_sensor_age_days(sensor_start, pd.Timestamp(df.index[-1]))
        payload["max_sensor_age_days"] = round(age_end, 1)

    payload["models"] = results

    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"Results saved to {out}")
