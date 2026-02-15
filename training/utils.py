from pathlib import Path
import json
import os
import pickle

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


def calculate_absolute_humidity(temperature, rel_humidity):
    """Calculate absolute humidity (g/m^3) from temperature (C) and relative humidity (%)."""
    a, b = 17.27, 237.7
    alpha = ((a * temperature) / (b + temperature)) + np.log(rel_humidity / 100.0)
    return (6.112 * np.exp(alpha) * 2.1674) / (273.15 + temperature)


def train_model(
    model, X_train, y_train, X_val, y_val, model_name,
    epochs=200, device=None, batch_size=32, learning_rate=0.001,
    lr_scheduler_patience=10, lr_scheduler_factor=0.5,
):
    """Train a model with DataLoader, LR scheduler, and validation tracking."""
    if device is None:
        device = get_device()

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
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}], Train: {train_loss:.6f}, Val: {val_loss:.6f}")

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


def save_trained_model(
    model, feature_scaler, target_scaler, model_type,
    window_size, baseline_gas_resistance, model_dir, metrics,
    training_history=None,
):
    """Save a fully trained model with scalers, config, and checkpoint."""
    os.makedirs(model_dir, exist_ok=True)

    with open(f"{model_dir}/feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)

    with open(f"{model_dir}/target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    config = {
        "baseline_gas_resistance": float(baseline_gas_resistance),
        "trained_date": pd.Timestamp.now().isoformat(),
        "window_size": window_size,
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "notes": f"Trained on real BSEC data with {model_type.upper()}",
    }

    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "model_type": model_type,
        "window_size": window_size,
        "input_dim": window_size * 6,
    }

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

    print(f"\n  Saved {model_type.upper()}: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")
