from pathlib import Path
import hashlib
import json
import os
import pickle
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def train_model(
    model, X_train, y_train, X_val, y_val, model_name,
    epochs=200, device=None, batch_size=32, learning_rate=0.001,
    lr_scheduler_patience=10, lr_scheduler_factor=0.5,
    log_dir=None, histogram_freq=50,
):
    """Train a model with DataLoader, LR scheduler, and validation tracking.

    Args:
        log_dir: If set, write TensorBoard events (scalars, LR, weight histograms).
        histogram_freq: Log weight histograms every N epochs (0 to disable).
    """
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
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
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
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
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}], Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        # ── TensorBoard logging ───────────────────────────────────────
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

            if histogram_freq > 0 and (epoch + 1) % histogram_freq == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f"Weights/{name}", param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    if writer is not None:
        writer.flush()
        writer.close()

    print(f"  Best validation loss: {best_val_loss:.6f}")
    return {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
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


def update_central_manifest(model_type: str, run_entry: dict) -> str:
    """Read/create MANIFEST.json, increment version, append run, return version string.

    Previous runs of the same model_type are marked is_active: false.
    """
    from app.config import settings

    manifest_path = Path(settings.TRAINED_MODELS_BASE) / "MANIFEST.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            central = json.load(f)
    else:
        central = {"runs": []}

    # Find max version for this model type
    max_version = 0
    for run in central["runs"]:
        if run.get("model_type") == model_type:
            run["is_active"] = False
            v = run.get("version", "")
            if v.startswith(f"{model_type}-v"):
                try:
                    max_version = max(max_version, int(v.split("-v")[1]))
                except ValueError:
                    pass

    new_version = f"{model_type}-v{max_version + 1}"
    run_entry["version"] = new_version
    run_entry["model_type"] = model_type
    run_entry["is_active"] = True
    central["runs"].append(run_entry)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(central, f, indent=2, default=str)

    return new_version


def save_trained_model(
    model, feature_scaler, target_scaler, model_type,
    window_size, model_dir, metrics,
    baselines=None, sensor_type=None, iaq_standard=None,
    baseline_gas_resistance=None,  # legacy compat
    training_history=None,
    data_manifest=None,
):
    """Save a fully trained model with scalers, config, and checkpoint."""
    os.makedirs(model_dir, exist_ok=True)

    with open(f"{model_dir}/feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)

    with open(f"{model_dir}/target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    # Resolve baselines — support both new dict and legacy scalar
    if baselines is None:
        baselines = {}
    if baseline_gas_resistance is not None and "voc_resistance" not in baselines:
        baselines["voc_resistance"] = float(baseline_gas_resistance)

    config = {
        "sensor_type": sensor_type or "bme680",
        "iaq_standard": iaq_standard or "bsec",
        "baselines": {k: float(v) for k, v in baselines.items()},
        "baseline_gas_resistance": baselines.get("voc_resistance", 0.0),
        "trained_date": pd.Timestamp.now().isoformat(),
        "window_size": window_size,
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "notes": f"Trained with {model_type.upper()} ({sensor_type or 'bme680'}/{iaq_standard or 'bsec'})",
    }

    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Determine num_features from the active sensor profile
    try:
        from app.profiles import get_sensor_profile
        num_features = get_sensor_profile().total_features
    except Exception:
        num_features = 6

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
