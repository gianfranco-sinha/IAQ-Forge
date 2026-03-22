from pathlib import Path
import hashlib
import json
import logging
import math
import os
import pickle
import random
import signal
import subprocess
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("training.utils")


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducible training.

    Seeds Python, NumPy, and PyTorch RNGs. Enables deterministic algorithms
    with warn_only=True since MPS lacks deterministic implementations for some ops.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_device():
    """Detect best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_sliding_windows(features, targets, window_size=10):
    """Create sliding window sequences from feature array and target array."""
    windows_X = []
    windows_y = []

    for i in range(len(features) - window_size + 1):
        windows_X.append(features[i : i + window_size].flatten())
        windows_y.append(targets[i + window_size - 1])

    return np.array(windows_X), np.array(windows_y)


def find_contiguous_segments(index, max_gap_factor=2.0):
    """Find contiguous segments in a DatetimeIndex by detecting time gaps.

    A gap is defined as a delta exceeding max_gap_factor * median_delta.

    Returns:
        segments: list of (start, end) index pairs into the original array
        gap_info: dict with gap detection metadata
    """
    import pandas as pd

    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return [(0, len(index))], {"gaps_found": 0, "segments": 1, "skipped": True}

    deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
    median_delta = np.median(deltas)
    threshold = median_delta * max_gap_factor

    gap_mask = deltas > threshold
    gap_indices = np.where(gap_mask)[0] + 1  # index of first reading after gap

    # Build segment boundaries
    boundaries = [0] + gap_indices.tolist() + [len(index)]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    gap_info = {
        "gaps_found": int(gap_mask.sum()),
        "segments": len(segments),
        "median_interval_seconds": float(median_delta),
        "gap_threshold_seconds": float(threshold),
    }

    if gap_mask.any():
        gap_sizes = deltas[gap_mask]
        gap_info["largest_gap_seconds"] = float(gap_sizes.max())

    return segments, gap_info


def calculate_absolute_humidity(temperature, rel_humidity):
    """Calculate absolute humidity (g/m^3) from temperature (C) and relative humidity (%)."""
    a, b = 17.27, 237.7
    alpha = ((a * temperature) / (b + temperature)) + np.log(rel_humidity / 100.0)
    return (6.112 * np.exp(alpha) * 2.1674) / (273.15 + temperature)


def load_checkpoint(checkpoint_path: str) -> Optional[dict]:
    """Load a training checkpoint if it exists. Returns None if not found."""
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        logger.info("Loaded checkpoint from %s (epoch %d)", path, checkpoint.get("epoch", 0))
        return checkpoint
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s", path, e)
        return None


def _save_checkpoint(
    path: str,
    epoch: int,
    model,
    optimizer,
    scheduler,
    train_losses: list,
    val_losses: list,
    best_val_loss: float,
) -> None:
    """Save a training checkpoint to disk."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        },
        path,
    )
    logger.info("Saved checkpoint at epoch %d to %s", epoch, path)


def train_model(
    model, X_train, y_train, X_val, y_val, model_name,
    epochs=200, device=None, batch_size=32, learning_rate=0.001,
    lr_scheduler_patience=10, lr_scheduler_factor=0.5,
    log_dir=None, histogram_freq=50,
    seed=None,
    on_epoch=None,
    checkpoint_path: Optional[str] = None,
    checkpoint_freq: int = 20,
    model_dir: Optional[str] = None,
):
    """Train a model with DataLoader, LR scheduler, and validation tracking.

    Args:
        log_dir: If set, write TensorBoard events (scalars, LR, weight histograms).
        histogram_freq: Log weight histograms every N epochs (0 to disable).
        seed: If set, seed all RNGs and use a deterministic DataLoader shuffle.
        on_epoch: Optional callback invoked after each epoch with signature
            (epoch: int, train_loss: float, val_loss: float, lr: float) -> None.
        checkpoint_path: If set and file exists, resume training from this checkpoint.
        checkpoint_freq: Save checkpoint every N epochs (0 to disable).
        model_dir: Directory to save checkpoint.pt into. Required for checkpointing.

    Returns:
        Dict with best_val_loss, train_losses, val_losses, and interrupted flag.
    """
    if seed is not None:
        seed_everything(seed)

    if device is None:
        device = get_device()

    # ── TensorBoard setup ─────────────────────────────────────────────
    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)

    print(f"\nTraining {model_name} on {device}...")
    model = model.to(device)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1)
    )

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    start_epoch = 0
    interrupted = False

    # ── Resume from checkpoint ────────────────────────────────────────
    if checkpoint_path:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            train_losses = ckpt["train_losses"]
            val_losses = ckpt["val_losses"]
            best_val_loss = ckpt["best_val_loss"]
            print(f"  Resuming from epoch {start_epoch} (best_val_loss={best_val_loss:.6f})")

    # ── Resolve checkpoint save path ──────────────────────────────────
    ckpt_save_path = None
    if model_dir and checkpoint_freq > 0:
        ckpt_save_path = str(Path(model_dir) / "checkpoint.pt")

    # ── Signal handling for graceful interrupt ────────────────────────
    _interrupted = False

    def _interrupt_handler(signum, frame):
        nonlocal _interrupted
        _interrupted = True

    old_sigint = signal.signal(signal.SIGINT, _interrupt_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _interrupt_handler)

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                if hasattr(model, 'kl_loss'):
                    kl_weight = getattr(model, '_kl_weight', 1.0)
                    loss = loss + kl_weight * model.kl_loss() / len(train_loader.dataset)
                if hasattr(model, 'regularization_loss'):
                    reg_weight = getattr(model, '_reg_weight', 1.0)
                    loss = loss + reg_weight * model.regularization_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            if math.isnan(train_loss) or math.isinf(train_loss):
                from app.exceptions import TrainingDivergedError
                raise TrainingDivergedError(
                    f"Training diverged at epoch {epoch + 1}: loss={train_loss}",
                    suggestion="Reduce learning rate or check data for extreme values",
                )
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                    if hasattr(model, 'kl_loss'):
                        kl_weight = getattr(model, '_kl_weight', 1.0)
                        loss = loss + kl_weight * model.kl_loss() / len(val_loader.dataset)
                    if hasattr(model, 'regularization_loss'):
                        reg_weight = getattr(model, '_reg_weight', 1.0)
                        loss = loss + reg_weight * model.regularization_loss()
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch + 1}/{epochs}], Train: {train_loss:.6f}, Val: {val_loss:.6f}")

            # ── Epoch callback ────────────────────────────────────────
            if on_epoch is not None:
                on_epoch(epoch, train_loss, val_loss, optimizer.param_groups[0]["lr"])

            # ── TensorBoard logging ───────────────────────────────────
            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

                if histogram_freq > 0 and (epoch + 1) % histogram_freq == 0:
                    for name, param in model.named_parameters():
                        try:
                            writer.add_histogram(f"Weights/{name}", param, epoch)
                        except ValueError:
                            pass
                        if param.grad is not None:
                            try:
                                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
                            except ValueError:
                                pass

            # ── Periodic checkpoint ───────────────────────────────────
            if ckpt_save_path and (epoch + 1) % checkpoint_freq == 0:
                _save_checkpoint(
                    ckpt_save_path, epoch + 1, model, optimizer, scheduler,
                    train_losses, val_losses, best_val_loss,
                )

            # ── Graceful interrupt ────────────────────────────────────
            if _interrupted:
                print(f"\n  Interrupted at epoch {epoch + 1} — saving checkpoint")
                if ckpt_save_path:
                    _save_checkpoint(
                        ckpt_save_path, epoch + 1, model, optimizer, scheduler,
                        train_losses, val_losses, best_val_loss,
                    )
                interrupted = True
                break
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    if writer is not None:
        writer.flush()
        writer.close()

    # ── Cleanup checkpoint on successful completion ───────────────────
    if not interrupted and ckpt_save_path:
        ckpt_path = Path(ckpt_save_path)
        if ckpt_path.exists():
            ckpt_path.unlink()
            logger.info("Training complete — removed checkpoint %s", ckpt_save_path)

    print(f"  Best validation loss: {best_val_loss:.6f}")
    return {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "interrupted": interrupted,
    }


def evaluate_model(model, X_val, y_val, target_scaler, device=None):
    """Evaluate model and return MAE, RMSE, R2 in original IAQ scale."""
    if device is None:
        device = get_device()

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()

    y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(predictions).flatten()

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def save_training_history(model_type, epochs, train_losses, val_losses, metrics, output_dir):
    """Save epoch-level training history and final metrics to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "model_type": model_type,
        "trained_date": pd.Timestamp.now().isoformat(),
        "epochs": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


def _compute_data_fingerprint(df: pd.DataFrame) -> str:
    """SHA256 of cleaned DataFrame for deterministic data identification.

    Columns sorted alphabetically + CSV serialization for determinism.
    Same data → same hash.
    """
    sorted_df = df[sorted(df.columns)]
    csv_bytes = sorted_df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _get_git_commit() -> str:
    """Return short git hash of HEAD, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _compute_feature_statistics(features: np.ndarray, feature_names: list) -> dict:
    """Per-feature mean/std/min/max dict."""
    stats = {}
    for i, name in enumerate(feature_names):
        col = features[:, i] if features.ndim > 1 else features
        stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return stats


def save_data_manifest(manifest: dict, model_dir) -> None:
    """Write data_manifest.json to the model directory."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def compute_schema_fingerprint(
    sensor_type: str,
    iaq_standard: str,
    window_size: int,
    num_features: int,
    model_type: str,
    feature_names: List[str] = None,
) -> str:
    """Hash the fields that define a model's input/output contract.

    If this fingerprint changes between versions, it signals a MAJOR (breaking)
    version bump — the model's input shape or contract has changed.

    When feature_names is provided, it is included in the hash so that a
    reordering of features (same count but different order) triggers a
    fingerprint change.  When None, the hash is identical to the legacy
    computation for backward compatibility.
    """
    schema = {
        "sensor_type": sensor_type,
        "iaq_standard": iaq_standard,
        "window_size": window_size,
        "num_features": num_features,
        "model_type": model_type,
    }
    if feature_names is not None:
        schema["feature_names"] = feature_names
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True).encode()
    ).hexdigest()[:12]


def _parse_semver(version_str: str, model_type: str):
    """Parse a semver version string like 'mlp-1.2.0' into (major, minor, patch).

    Returns None if the string is a legacy format (e.g. 'mlp-v6') or unparseable.
    """
    import re
    prefix = f"{model_type}-"
    if not version_str.startswith(prefix):
        return None
    remainder = version_str[len(prefix):]
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", remainder)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def compute_semver(
    model_type: str,
    schema_fingerprint: str,
    data_fingerprint: str,
    manifest_runs: list,
    metrics: dict = None,
) -> str:
    """Determine the next semantic version for a model type.

    Compares against the previous *active* version of the same model_type:
    - schema_fingerprint changed → MAJOR bump (reset minor/patch)
    - data_fingerprint changed OR metrics changed → MINOR bump (reset patch)
    - only metadata changed → PATCH bump

    In practice every training run changes metrics, so the common path is MINOR.
    PATCH is reserved for metadata-only updates (no retraining).

    Legacy versions (e.g. 'mlp-v6') are ignored — first semver starts at 1.0.0.
    """
    # Find previous active semver run for this model type
    prev_run = None
    for run in manifest_runs:
        if (
            run.get("model_type") == model_type
            and run.get("is_active")
            and _parse_semver(run.get("version", ""), model_type) is not None
        ):
            prev_run = run

    if prev_run is None:
        return "1.0.0"

    prev_ver = _parse_semver(prev_run["version"], model_type)
    major, minor, patch = prev_ver

    prev_schema_fp = prev_run.get("schema_fingerprint", "")
    prev_data_fp = prev_run.get("data_fingerprint", "")

    if prev_schema_fp and prev_schema_fp != schema_fingerprint:
        return f"{major + 1}.0.0"

    if prev_data_fp != data_fingerprint:
        return f"{major}.{minor + 1}.0"

    # Check if metrics changed (different weights → different metrics = retrained)
    if metrics and prev_run.get("metrics"):
        prev_metrics = prev_run["metrics"]
        for key in ("mae", "rmse", "r2"):
            if key in metrics and key in prev_metrics:
                if abs(float(metrics[key]) - float(prev_metrics[key])) > 1e-9:
                    return f"{major}.{minor + 1}.0"

    return f"{major}.{minor}.{patch + 1}"


def update_central_manifest(
    model_type: str,
    run_entry: dict,
    schema_fingerprint: str = None,
) -> str:
    """Read/create MANIFEST.json, compute semver, append run, return version string.

    Previous runs of the same model_type are marked is_active: false.
    Legacy versions (mlp-v6) are preserved as-is — never modified or deleted.
    """
    from app.config import settings

    manifest_path = Path(settings.TRAINED_MODELS_BASE) / "MANIFEST.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            central = json.load(f)
    else:
        central = {"runs": []}

    # Compute semver before deactivating previous runs
    semver = compute_semver(
        model_type,
        schema_fingerprint or "",
        run_entry.get("data_fingerprint", ""),
        central["runs"],
        metrics=run_entry.get("metrics"),
    )
    new_version = f"{model_type}-{semver}"

    # Advisory: warn if sensor_id changed between consecutive runs
    for run in reversed(central["runs"]):
        if run.get("model_type") == model_type and run.get("is_active"):
            prev_sensor = run.get("sensor_id")
            curr_sensor = run_entry.get("sensor_id")
            if prev_sensor and curr_sensor and prev_sensor != curr_sensor:
                logger.warning("Sensor changed: %s → %s", prev_sensor, curr_sensor)
            break

    # Deactivate previous runs of same model type
    for run in central["runs"]:
        if run.get("model_type") == model_type:
            run["is_active"] = False

    run_entry["version"] = new_version
    run_entry["model_type"] = model_type
    run_entry["is_active"] = True
    if schema_fingerprint:
        run_entry["schema_fingerprint"] = schema_fingerprint
    central["runs"].append(run_entry)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(central, f, indent=2, default=str)

    return new_version


def save_trained_model(
    model, feature_scaler, target_scaler, model_type,
    window_size, model_dir, metrics,
    baselines=None, sensor_type=None, iaq_standard=None,
    sensor_id=None, firmware_version=None,
    training_history=None,
    data_manifest=None,
    feature_names: List[str] = None,
):
    """Save a fully trained model with scalers, config, and checkpoint."""
    os.makedirs(model_dir, exist_ok=True)

    with open(f"{model_dir}/feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)

    with open(f"{model_dir}/target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    if baselines is None:
        baselines = {}

    config = {
        "sensor_type": sensor_type or "bme680",
        "iaq_standard": iaq_standard or "bsec",
        "baselines": {k: float(v) for k, v in baselines.items()},
        "trained_date": pd.Timestamp.now().isoformat(),
        "window_size": window_size,
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "notes": f"Trained with {model_type.upper()} ({sensor_type or 'bme680'}/{iaq_standard or 'bsec'})",
    }
    if feature_names is not None:
        config["feature_names"] = feature_names
    if sensor_id:
        config["sensor_id"] = sensor_id
    if firmware_version:
        config["firmware_version"] = firmware_version

    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Determine num_features from the active sensor profile
    try:
        from app.profiles import get_sensor_profile
        num_features = get_sensor_profile().total_features
    except Exception:
        num_features = 15

    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "model_type": model_type,
        "window_size": window_size,
        "input_dim": window_size * num_features,
    }

    if hasattr(model, 'prior_sigma'):
        checkpoint["prior_sigma"] = model.prior_sigma

    torch.save(checkpoint, f"{model_dir}/model.pt")

    if training_history is not None:
        save_training_history(
            model_type=model_type,
            epochs=len(training_history["train_losses"]),
            train_losses=training_history["train_losses"],
            val_losses=training_history["val_losses"],
            metrics=metrics,
            output_dir=model_dir,
        )

    if data_manifest is not None:
        save_data_manifest(data_manifest, model_dir)

    print(f"\n  Saved {model_type.upper()}: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")


def patch_config_with_version(model_dir, version: str, schema_fingerprint: str) -> None:
    """Patch config.json in model_dir with version and schema_fingerprint after save."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        config = json.load(f)
    config["version"] = version
    config["schema_fingerprint"] = schema_fingerprint
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def patch_config_with_merkle_hash(model_dir, merkle_root_hash: str) -> None:
    """Patch config.json in model_dir with merkle_root_hash after save."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        config = json.load(f)
    config["merkle_root_hash"] = merkle_root_hash
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def patch_manifest_merkle(model_type: str, version: str, merkle_root_hash: str) -> None:
    """Patch the active run in central MANIFEST.json with merkle_root_hash."""
    from app.config import settings

    manifest_path = Path(settings.TRAINED_MODELS_BASE) / "MANIFEST.json"
    if not manifest_path.exists():
        return
    with open(manifest_path) as f:
        central = json.load(f)
    for run in central.get("runs", []):
        if (
            run.get("model_type") == model_type
            and run.get("is_active")
            and run.get("version") == version
        ):
            run["merkle_root_hash"] = merkle_root_hash
    with open(manifest_path, "w") as f:
        json.dump(central, f, indent=2, default=str)
