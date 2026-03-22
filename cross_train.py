"""Cross-training: self-supervised pre-training + supervised fine-tuning.

Phase 1: Pre-train a masked autoencoder on ALL available sensor data
         (no IAQ labels needed — uses raw BME680 readings only).
Phase 2: Transfer encoder weights to MLP, fine-tune on labeled BSEC data.
Phase 3 (optional): Pseudo-label high-confidence old data, retrain on combined.

Usage:
    python cross_train.py --pretrain-epochs 100 --finetune-epochs 200
    python cross_train.py --skip-pretrain --encoder-path cache/encoder.pt
    python cross_train.py --pseudo-label --confidence 0.9

Requires:
    - Unlabeled data: cache/combined_3month_iaq.parquet or InfluxDB
    - Labeled data: InfluxDB with iaq_accuracy >= 2 filter
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from app.builtin_profiles import BME680Profile
from app.config import settings
from app.models import MLPRegressor, build_model
from training.data_sources import ParquetSource
from training.pipeline import TrainingPipeline
from training.pretrain import MaskedAutoencoder, pretrain_encoder, transfer_weights
from training.utils import (
    create_sliding_windows,
    find_contiguous_segments,
    get_device,
    seed_everything,
    train_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("cross_train")

COMBINED_PARQUET = Path("cache/combined_3month_iaq.parquet")
INFLUX_CACHE = Path("cache/influxdb_a4b58597429641c7.parquet")
ENCODER_CACHE = Path("cache/pretrained_encoder.pt")


def load_unlabeled_windows(
    data_path: Path,
    profile: BME680Profile,
    window_size: int = 10,
) -> np.ndarray:
    """Load raw sensor data, apply warm-up filtering, engineer features, create windows.

    Returns flattened windows (n_windows, window_size * total_features).
    No IAQ target needed.
    """
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    # Filter by valid ranges
    for feat, (lo, hi) in profile.valid_ranges.items():
        if feat in df.columns:
            before = len(df)
            df = df[(df[feat] >= lo) & (df[feat] <= hi)]
            dropped = before - len(df)
            if dropped > 0:
                logger.info("  %s range filter: dropped %d rows", feat, dropped)

    df = df.dropna()
    logger.info("After cleaning: %d rows", len(df))

    # Find contiguous segments and trim warm-up
    segments, gap_info = find_contiguous_segments(df.index)
    warmup_samples = int(profile.warmup_seconds / (profile.expected_interval_seconds or 3.0))

    trimmed_segments = []
    for s, e in segments:
        if (e - s) > warmup_samples:
            trimmed_segments.append((s + warmup_samples, e))

    if not trimmed_segments:
        trimmed_segments = segments  # fallback

    logger.info(
        "%d segments, %d after warm-up trim (%ds), %d gaps",
        len(segments), len(trimmed_segments),
        int(profile.warmup_seconds), gap_info.get("gaps_found", 0),
    )

    # Extract raw features
    raw_cols = [c for c in profile.raw_features if c in df.columns]
    raw_data = df[raw_cols].values

    # Engineer features (includes envelope baselines, temporal, abs humidity)
    timestamps = df.index.values if isinstance(df.index, pd.DatetimeIndex) else None
    features = profile.engineer_features(raw_data, timestamps=timestamps)

    # Create windows from all valid segments
    all_windows = []
    for s, e in trimmed_segments:
        seg_len = e - s
        if seg_len >= window_size:
            seg_features = features[s:e]
            # Manual sliding window (no target needed)
            for i in range(seg_len - window_size + 1):
                window = seg_features[i:i + window_size].flatten()
                all_windows.append(window)

    X = np.array(all_windows)
    logger.info("Created %d unlabeled windows (window=%d, dim=%d)", len(X), window_size, X.shape[1])
    return X


def run_pretrain(
    X: np.ndarray,
    hidden_dims: List[int],
    epochs: int,
    mask_ratio: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 15,
) -> Dict:
    """Run Phase 1: self-supervised pre-training."""
    print("=" * 70)
    print("PHASE 1: SELF-SUPERVISED PRE-TRAINING")
    print(f"  Windows:    {len(X):,}")
    print(f"  Input dim:  {X.shape[1]}")
    print(f"  Hidden:     {hidden_dims}")
    print(f"  Mask ratio: {mask_ratio:.0%}")
    print(f"  Epochs:     {epochs}")
    print("=" * 70)

    result = pretrain_encoder(
        X=X,
        input_dim=X.shape[1],
        hidden_dims=hidden_dims,
        mask_ratio=mask_ratio,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
    )

    # Cache encoder weights
    ENCODER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, ENCODER_CACHE)
    print(f"  Encoder saved to {ENCODER_CACHE}")

    return result


def run_finetune(
    pretrain_result: Dict,
    data_path: Path,
    epochs: int,
    freeze_encoder: bool = True,
    model_type: str = "mlp",
) -> Dict:
    """Run Phase 2: supervised fine-tuning with transferred weights."""
    print(f"\n{'=' * 70}")
    print("PHASE 2: SUPERVISED FINE-TUNING")
    print(f"  Data:           {data_path}")
    print(f"  Model:          {model_type}")
    print(f"  Epochs:         {epochs}")
    print(f"  Freeze encoder: {freeze_encoder}")
    print("=" * 70)

    output_dir = f"trained_models/cross_train/{model_type}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    source = ParquetSource(data_path)
    pipeline = TrainingPipeline(
        source=source,
        model_type=model_type,
        epochs=epochs,
        output_dir=output_dir,
    )

    def on_stage(state, result):
        if result and result.extra:
            print(f"    [{state.value}] {result.extra}")

    pipeline.on_stage_complete(on_stage)

    # Hook: after model is built but before training, transfer weights
    import types

    original_do_training = pipeline._do_training

    def patched_do_training(self_pipe):
        from training.pipeline.types import StageResult, PipelineState

        # Run original training setup (builds model, sets device)
        result = original_do_training()

        # At this point self._model exists but train_model() already ran.
        # We need a different approach — intercept model build instead.
        return result

    # Better approach: let pipeline build + train, but inject weights after build_model
    original_build = pipeline._do_training.__func__ if hasattr(pipeline._do_training, '__func__') else None

    # Simplest approach: run pipeline stages up to SCALING, then do our own training
    from training.pipeline.types import PipelineState

    # Run the pipeline in custom mode
    pipeline._do_source_access()
    pipeline._do_ingestion()
    pipeline._do_feature_engineering()
    pipeline._do_windowing()
    pipeline._do_splitting()
    pipeline._do_scaling()

    # Now we have pipeline._X_train, _y_train, _X_val, _y_val
    # Build model and transfer weights
    from app.profiles import get_sensor_profile
    profile = get_sensor_profile()

    model_cfg = settings.get_model_config(model_type)
    window_size = model_cfg.get("window_size", 10)

    model = build_model(model_type, window_size=window_size, num_features=profile.total_features)

    n_transferred = transfer_weights(model, pretrain_result, freeze_encoder=freeze_encoder)

    if n_transferred == 0:
        print("  WARNING: No weights transferred — training from scratch")

    # Train with transferred weights
    device = get_device()
    seed_everything(42)

    tcfg = settings.get_training_config()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = str(Path(tcfg.get("tensorboard_log_dir", "runs")) / f"cross_train_{model_type}_{ts}")

    history = train_model(
        model=model,
        model_name=f"{model_type.upper()} (cross-trained)",
        X_train=pipeline._X_train,
        y_train=pipeline._y_train,
        X_val=pipeline._X_val,
        y_val=pipeline._y_val,
        epochs=epochs,
        batch_size=pipeline._batch_size,
        learning_rate=pipeline._learning_rate,
        lr_scheduler_patience=pipeline._lr_scheduler_patience,
        lr_scheduler_factor=pipeline._lr_scheduler_factor,
        device=device,
        log_dir=tb_dir,
        checkpoint_freq=tcfg.get("checkpoint_freq", 20),
        model_dir=output_dir,
    )

    # Evaluate
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        X_val_t = torch.FloatTensor(pipeline._X_val).to(device)
        preds = model(X_val_t).cpu().numpy().flatten()

    # Inverse transform
    if pipeline._target_scaler is not None:
        preds = pipeline._target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        y_true = pipeline._target_scaler.inverse_transform(
            pipeline._y_val.reshape(-1, 1)
        ).flatten()
    else:
        y_true = pipeline._y_val

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)

    print(f"\n  RESULTS: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # Save model
    from training.utils import save_trained_model
    save_trained_model(
        model=model,
        model_type=model_type,
        model_dir=output_dir,
        feature_scaler=pipeline._feature_scaler,
        target_scaler=pipeline._target_scaler,
        window_size=window_size,
        training_history={
            "train_losses": history.get("train_losses", []),
            "val_losses": history.get("val_losses", []),
        },
        metrics={"mae": mae, "rmse": rmse, "r2": r2},
        sensor_type=profile.name,
        iaq_standard="bsec",
        baselines=pipeline._baselines if hasattr(pipeline, "_baselines") else {},
    )
    print(f"  Model saved to {output_dir}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "best_val_loss": history.get("best_val_loss"),
        "epochs_trained": len(history.get("train_losses", [])),
        "weights_transferred": n_transferred,
        "freeze_encoder": freeze_encoder,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-training: pre-train + fine-tune")
    parser.add_argument("--pretrain-epochs", type=int, default=100,
                        help="Pre-training epochs (Phase 1)")
    parser.add_argument("--finetune-epochs", type=int, default=200,
                        help="Fine-tuning epochs (Phase 2)")
    parser.add_argument("--mask-ratio", type=float, default=0.2,
                        help="Fraction of features to mask (0.15-0.30)")
    parser.add_argument("--freeze", action="store_true", default=False,
                        help="Freeze encoder during fine-tuning (train head only)")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip Phase 1, load encoder from cache")
    parser.add_argument("--encoder-path", type=str, default=str(ENCODER_CACHE),
                        help="Path to cached encoder weights")
    parser.add_argument("--model", type=str, default="mlp",
                        help="Model type for fine-tuning")
    parser.add_argument("--hidden-dims", type=str, default="64,32,16",
                        help="Hidden layer dimensions (comma-separated)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/mps/cuda). Default: auto-detect.")
    args = parser.parse_args()

    # Override device detection if specified
    if args.device:
        import training.utils as _tu
        import training.pretrain as _tp
        _forced_device = args.device
        _tu.get_device = lambda: _forced_device
        _tp.get_device = lambda: _forced_device

    hidden_dims = [int(d) for d in args.hidden_dims.split(",")]

    # Find data
    if COMBINED_PARQUET.exists():
        data_path = COMBINED_PARQUET
    elif INFLUX_CACHE.exists():
        data_path = INFLUX_CACHE
    else:
        print("ERROR: No cached data found. Run data fetch first.")
        return

    profile = BME680Profile()
    model_cfg = settings.get_model_config(args.model)
    window_size = model_cfg.get("window_size", 10)

    start = time.time()

    # ── Phase 1: Pre-train ──
    if args.skip_pretrain:
        encoder_path = Path(args.encoder_path)
        if not encoder_path.exists():
            print(f"ERROR: Encoder not found at {encoder_path}")
            return
        print(f"Loading pre-trained encoder from {encoder_path}")
        pretrain_result = torch.load(encoder_path, map_location="cpu")
    else:
        X_unlabeled = load_unlabeled_windows(data_path, profile, window_size)
        pretrain_result = run_pretrain(
            X=X_unlabeled,
            hidden_dims=hidden_dims,
            epochs=args.pretrain_epochs,
            mask_ratio=args.mask_ratio,
        )

    # ── Phase 2: Fine-tune ──
    finetune_result = run_finetune(
        pretrain_result=pretrain_result,
        data_path=data_path,
        epochs=args.finetune_epochs,
        freeze_encoder=args.freeze,
        model_type=args.model,
    )

    elapsed = time.time() - start

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("CROSS-TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Pre-training:  {pretrain_result.get('epochs_trained', '?')} epochs, "
          f"val_loss={pretrain_result.get('best_val_loss', 0):.6f}")
    print(f"  Fine-tuning:   {finetune_result['epochs_trained']} epochs")
    print(f"  Transferred:   {finetune_result['weights_transferred']:,} parameters")
    print(f"  Frozen:        {finetune_result['freeze_encoder']}")
    print(f"  MAE:           {finetune_result['mae']:.4f}")
    print(f"  RMSE:          {finetune_result['rmse']:.4f}")
    print(f"  R²:            {finetune_result['r2']:.4f}")
    print(f"  Time:          {elapsed / 60:.1f} minutes")

    # Save results
    output_dir = Path("trained_models/cross_train")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "cross_training",
        "timestamp": datetime.utcnow().isoformat(),
        "data_source": str(data_path),
        "pretrain": {
            "epochs": pretrain_result.get("epochs_trained"),
            "best_val_loss": pretrain_result.get("best_val_loss"),
            "hidden_dims": pretrain_result.get("hidden_dims"),
            "mask_ratio": args.mask_ratio,
        },
        "finetune": finetune_result,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
