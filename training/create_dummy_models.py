"""
Create dummy trained models for testing the service without real training.
WARNING: These will produce random predictions!
"""

import torch
import pickle
import json
import os
from app.models import MLPRegressor, CNNRegressor, KANRegressor, LSTMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


def create_dummy_model(model_type="mlp", model_dir="trained_models/mlp"):
    """Create a dummy model for testing."""

    os.makedirs(model_dir, exist_ok=True)

    window_size = 10
    num_features = 6
    input_dim = window_size * num_features  # 60

    # Create model
    if model_type == "mlp":
        model = MLPRegressor(input_dim, hidden_dims=[64, 32, 16])
        print(f"Creating MLP (Baseline) model...")
    elif model_type == "cnn":
        model = CNNRegressor(window_size=window_size, num_features=num_features)
        print(f"Creating CNN model...")
    elif model_type == "lstm":
        model = LSTMRegressor(window_size=window_size, num_features=num_features)
        print(f"Creating LSTM model...")
    else:  # kan
        model = KANRegressor(input_dim, hidden_dims=[32, 16])
        print(f"Creating KAN model...")

    # Create dummy scalers
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit with dummy data
    dummy_features = np.random.randn(100, input_dim)
    dummy_targets = np.random.rand(100, 1) * 500

    feature_scaler.fit(dummy_features)
    target_scaler.fit(dummy_targets)

    # Save scalers
    with open(f"{model_dir}/feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)

    with open(f"{model_dir}/target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    # Save config
    config = {
        "baseline_resistance": 100000,
        "trained_date": "2026-01-30",
        "note": f"DUMMY {model_type.upper()} MODEL FOR TESTING ONLY",
    }

    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_type": model_type,
        "window_size": window_size,
        "input_dim": input_dim,
    }

    if model_type == "mlp":
        checkpoint["hidden_dims"] = [64, 32, 16]
    elif model_type == "cnn":
        checkpoint["num_filters"] = [64, 128, 256]
        checkpoint["kernel_sizes"] = [3, 3, 3]
    else:  # kan
        checkpoint["hidden_dims"] = [32, 16]

    torch.save(checkpoint, f"{model_dir}/model.pt")

    print(f"✓ Created dummy {model_type.upper()} model in {model_dir}")


if __name__ == "__main__":
    print("Creating dummy models for testing...")
    print("=" * 60)
    print()

    create_dummy_model("mlp", "../trained_models/mlp")
    create_dummy_model("cnn", "../trained_models/cnn")
    create_dummy_model("kan", "../trained_models/kan")
    create_dummy_model("lstm", "../trained_models/lstm")
    print()
    print("=" * 60)
    print("✓ All dummy models created successfully!")
    print()
    print("Models created:")
    print("  • MLP (Baseline) - Simple feedforward neural network")
    print("  • CNN - Convolutional neural network for temporal patterns")
    print("  • LSTM - Long Short Term Memory")
    print("  • KAN - Kolmogorov-Arnold Network")
    print()
    print("⚠️  WARNING: These models produce RANDOM predictions!")
    print("⚠️  Train real models with actual BSEC data before production!")
