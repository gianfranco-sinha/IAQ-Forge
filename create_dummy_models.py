"""
Create dummy models for testing the service without real training.
WARNING: These will produce random predictions!
"""

import torch
import pickle
import json
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from app.models import MLPRegressor, KANRegressor


def create_dummy_model(model_type='mlp', model_dir='trained_models/mlp'):
    """Create a dummy model for testing."""

    os.makedirs(model_dir, exist_ok=True)

    window_size = 10
    num_features = 6  # temp, humidity, pressure, resistance, gas_ratio, abs_humidity
    input_dim = window_size * num_features  # 60

    # Create model
    if model_type == 'mlp':
        model = MLPRegressor(input_dim, hidden_dims=[64, 32, 16])
    else:  # kan
        model = KANRegressor(input_dim, hidden_dims=[32, 16])

    # Create dummy scalers
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit with dummy data
    import numpy as np
    dummy_features = np.random.randn(100, input_dim)
    dummy_targets = np.random.rand(100, 1) * 500  # IAQ 0-500

    feature_scaler.fit(dummy_features)
    target_scaler.fit(dummy_targets)

    # Save scalers
    with open(f'{model_dir}/feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)

    with open(f'{model_dir}/target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)

    # Save config
    config = {
        'baseline_resistance': 100000,
        'trained_date': '2026-01-29',
        'note': 'DUMMY MODEL FOR TESTING ONLY'
    }

    with open(f'{model_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save model
    checkpoint = {
        'state_dict': model.state_dict(),
        'model_type': model_type,
        'window_size': window_size,
        'input_dim': input_dim,
        'hidden_dims': [64, 32, 16] if model_type == 'mlp' else [32, 16]
    }

    torch.save(checkpoint, f'{model_dir}/model.pt')

    print(f"✓ Created dummy {model_type.upper()} model in {model_dir}")


if __name__ == "__main__":
    print("Creating dummy models for testing...")
    create_dummy_model('mlp', 'trained_models/mlp')
    create_dummy_model('kan', 'trained_models/kan')
    print("\n✓ Dummy models created!")
    print("⚠️  WARNING: These models produce RANDOM predictions!")
    print("⚠️  Train real models with actual data before production use!")