"""
Log VOC ablation — trains with log(voc_resistance) instead of raw resistance.

Conditions:
  1. Raw VOC, no pressure (baseline — best from pressure ablation)
  2. Log VOC, no pressure

Uses cached InfluxDB data or combined 3-month dataset.

Usage:
    python ablation_log_voc.py [--epochs 50] [--models mlp,cnn,bnn,lstm]
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

from app.builtin_profiles import BME680Profile, BME680NoPressureProfile
from app.config import settings
from app.profiles import SensorProfile
from training.data_sources import DataSource, ParquetSource
from training.pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ablation_log_voc")

# Prefer the 3-month combined dataset; fall back to InfluxDB cache
COMBINED_PARQUET = Path("cache/combined_3month_iaq.parquet")
INFLUX_CACHE = Path("cache/influxdb_a4b58597429641c7.parquet")


class BME680LogVOCProfile(BME680NoPressureProfile):
    """BME680 with log(voc_resistance) replacing raw voc_resistance.

    Baselines and envelope ratios are computed on raw voc_resistance,
    then the raw column is replaced with log(voc_resistance) in the
    output feature array.
    """

    @property
    def name(self) -> str:
        return "bme680_log_voc"

    @property
    def valid_ranges(self) -> Dict[str, Tuple[float, float]]:
        # Keep raw-space ranges — pipeline checks raw data before engineer_features
        return super().valid_ranges

    def engineer_features(
        self,
        raw: np.ndarray,
        baselines: Optional[Dict[str, float]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Same as parent but replaces voc_resistance with log(voc_resistance)."""
        from training.utils import calculate_absolute_humidity

        voc_idx = self.raw_features.index("voc_resistance")
        temp_idx = self.raw_features.index("temperature")
        hum_idx = self.raw_features.index("rel_humidity")

        abs_humidity = calculate_absolute_humidity(raw[:, temp_idx], raw[:, hum_idx])

        # Envelope baselines computed on RAW voc_resistance (not log)
        voc_col = raw[:, voc_idx]
        if timestamps is not None:
            voc_series = pd.Series(voc_col, index=pd.DatetimeIndex(timestamps))
            use_time = True
        else:
            voc_series = pd.Series(voc_col)
            use_time = False

        baseline_24h, baseline_7d = self._compute_envelope(voc_series, use_time)
        baseline_24h_vals = baseline_24h.values
        baseline_7d_vals = baseline_7d.values

        gas_ratio_24h = np.clip(voc_col / baseline_24h_vals, 1e-6, None)
        log_ratio_24h = np.log(gas_ratio_24h)
        gas_ratio_7d = np.clip(voc_col / baseline_7d_vals, 1e-6, None)
        log_ratio_7d = np.log(gas_ratio_7d)

        # Temporal cyclical features
        if timestamps is not None:
            ts_index = pd.DatetimeIndex(timestamps)
            hours = ts_index.hour.values.astype(float)
            dows = ts_index.dayofweek.values.astype(float)
        else:
            hours = np.zeros(len(raw))
            dows = np.zeros(len(raw))

        hour_sin, hour_cos = self._cyclical_encode(hours, 24.0)
        dow_sin, dow_cos = self._cyclical_encode(dows, 7.0)

        # Replace raw voc_resistance with log(voc_resistance)
        raw_modified = raw.copy()
        raw_modified[:, voc_idx] = np.log(np.clip(voc_col, 1, None))

        return np.column_stack([
            raw_modified,
            abs_humidity.reshape(-1, 1),
            baseline_24h_vals.reshape(-1, 1),
            gas_ratio_24h.reshape(-1, 1),
            log_ratio_24h.reshape(-1, 1),
            baseline_7d_vals.reshape(-1, 1),
            gas_ratio_7d.reshape(-1, 1),
            log_ratio_7d.reshape(-1, 1),
            hour_sin.reshape(-1, 1),
            hour_cos.reshape(-1, 1),
            dow_sin.reshape(-1, 1),
            dow_cos.reshape(-1, 1),
        ])

    def compute_baselines(
        self, raw: np.ndarray, timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute baselines on raw voc_resistance (before log transform)."""
        # Parent computes on raw values — this is correct
        return super().compute_baselines(raw, timestamps)

    def engineer_features_single(
        self,
        reading: Dict[str, float],
        baselines: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        """Single-reading version with log(voc_resistance)."""
        from training.utils import calculate_absolute_humidity

        voc_r = reading["voc_resistance"]
        log_voc = np.log(max(voc_r, 1))

        # Raw features with log voc
        raw_vals = []
        for f in self.raw_features:
            if f == "voc_resistance":
                raw_vals.append(log_voc)
            else:
                raw_vals.append(reading[f])

        bl_24h = (
            baselines.get("baseline_24h", baselines.get("voc_resistance", voc_r))
            if baselines else voc_r
        )
        bl_7d = (
            baselines.get("baseline_7d", baselines.get("voc_resistance", voc_r))
            if baselines else voc_r
        )

        gas_ratio_24h = max(voc_r / bl_24h, 1e-6)
        log_ratio_24h = np.log(gas_ratio_24h)
        gas_ratio_7d = max(voc_r / bl_7d, 1e-6)
        log_ratio_7d = np.log(gas_ratio_7d)

        abs_hum = calculate_absolute_humidity(
            np.array([reading["temperature"]]),
            np.array([reading["rel_humidity"]]),
        )[0]

        hour = float(timestamp.hour) if timestamp is not None else 0.0
        dow = float(timestamp.weekday()) if timestamp is not None else 0.0
        hour_sin, hour_cos = self._cyclical_encode(np.array([hour]), 24.0)
        dow_sin, dow_cos = self._cyclical_encode(np.array([dow]), 7.0)

        return np.array(raw_vals + [
            abs_hum,
            bl_24h, gas_ratio_24h, log_ratio_24h,
            bl_7d, gas_ratio_7d, log_ratio_7d,
            hour_sin[0], hour_cos[0],
            dow_sin[0], dow_cos[0],
        ])


def run_condition(
    condition: str,
    profile: SensorProfile,
    source: DataSource,
    model_types: List[str],
    epochs: int,
    early_stopping_patience: int = 0,
) -> Dict[str, dict]:
    results = {}
    base_dir = Path("trained_models/ablation_log_voc") / condition

    for model_type in model_types:
        model_cfg = settings.get_model_config(model_type)
        window_size = model_cfg.get("window_size", 10)
        output_dir = str(base_dir / model_type)

        print(f"\n{'─' * 60}")
        print(f"  [{condition}] {model_type.upper()} (window={window_size}, "
              f"features={profile.total_features})")
        print(f"{'─' * 60}")

        try:
            pipeline = TrainingPipeline(
                source=source,
                model_type=model_type,
                epochs=epochs,
                window_size=window_size,
                output_dir=output_dir,
                early_stopping_patience=early_stopping_patience,
            )
            pipeline._sensor_profile = profile

            def log_progress(state, result):
                if result and result.extra:
                    print(f"    [{state.value}] {result.extra}")

            pipeline.on_stage_complete(log_progress)
            result = pipeline.orchestrate()

            if result.interrupted:
                print(f"    INTERRUPTED")
                results[model_type] = None
            else:
                m = result.metrics
                print(f"    MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}")
                results[model_type] = {
                    "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
                    "best_val_loss": result.training_history.get("best_val_loss"),
                    "version": result.version,
                }
        except Exception as e:
            print(f"    FAILED: {e}")
            logger.exception("Failed: %s/%s", condition, model_type)
            results[model_type] = None

    return results


def print_comparison(raw_results: dict, log_results: dict, model_types: List[str]):
    print("\n" + "=" * 90)
    print("LOG VOC ABLATION RESULTS")
    print("=" * 90)
    header = f"{'Model':<8} │ {'Raw VOC (no pressure)':^30} │ {'Log VOC (no pressure)':^30} │ {'Delta R2':^8}"
    print(header)
    print(f"{'':─<8}─┼─{'':─<30}─┼─{'':─<30}─┼─{'':─<8}")

    for mt in model_types:
        r = raw_results.get(mt)
        l = log_results.get(mt)
        r_str = f"MAE={r['mae']:.4f}  R2={r['r2']:.4f}" if r else "FAILED"
        l_str = f"MAE={l['mae']:.4f}  R2={l['r2']:.4f}" if l else "FAILED"
        if r and l:
            d = l["r2"] - r["r2"]
            d_str = f"{'+' if d >= 0 else ''}{d:.4f}"
        else:
            d_str = "—"
        print(f"{mt.upper():<8} │ {r_str:^30} │ {l_str:^30} │ {d_str:^8}")

    print()
    print("Delta R2 > 0 means log transform HELPS.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Log VOC ablation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--models", type=str, default="mlp,cnn,bnn,lstm")
    parser.add_argument("--early-stopping", type=int, default=0, metavar="N",
                        help="Stop after N epochs with no improvement (0=off)")
    args = parser.parse_args()

    model_types = [m.strip() for m in args.models.split(",")]

    # Pick data source
    if COMBINED_PARQUET.exists():
        data_path = COMBINED_PARQUET
    elif INFLUX_CACHE.exists():
        data_path = INFLUX_CACHE
    else:
        print("ERROR: No cached data found. Run data fetch first.")
        return

    source = ParquetSource(data_path)

    profile_raw = BME680NoPressureProfile()
    profile_log = BME680LogVOCProfile()

    print("=" * 90)
    print("LOG VOC ABLATION EXPERIMENT")
    print(f"  Data:       {data_path}")
    print(f"  Models:     {', '.join(m.upper() for m in model_types)}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Condition 1: raw VOC, no pressure ({profile_raw.total_features} features)")
    print(f"  Condition 2: log(VOC), no pressure ({profile_log.total_features} features)")
    print("=" * 90)

    start = time.time()

    # Condition 1: Raw VOC, no pressure
    print(f"\n{'=' * 90}")
    print("CONDITION 1: RAW VOC (no pressure)")
    print(f"{'=' * 90}")
    raw_results = run_condition("raw_voc", profile_raw, source, model_types, args.epochs, args.early_stopping)

    # Condition 2: Log VOC, no pressure
    print(f"\n{'=' * 90}")
    print("CONDITION 2: LOG VOC (no pressure)")
    print(f"{'=' * 90}")
    log_results = run_condition("log_voc", profile_log, source, model_types, args.epochs, args.early_stopping)

    elapsed = time.time() - start

    print_comparison(raw_results, log_results, model_types)

    # Save results
    output_dir = Path("trained_models/ablation_log_voc")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "log_voc_ablation",
        "timestamp": datetime.utcnow().isoformat(),
        "epochs": args.epochs,
        "data_source": str(data_path),
        "elapsed_seconds": round(elapsed, 1),
        "raw_voc": raw_results,
        "log_voc": log_results,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir / 'results.json'}")
    print(f"Total time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
