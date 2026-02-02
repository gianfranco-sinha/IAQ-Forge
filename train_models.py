"""
Train MLP and KAN models from collected BSEC data
"""

from influxdb import DataFrameClient
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json
import os

from app.models import MLPRegressor, KANRegressor

# InfluxDB settings - MODIFY THESE
INFLUX_HOST = "87.106.102.14"
INFLUX_PORT = 8086
INFLUX_DATABASE = "home_study_room_iaq"  # Change to your database name
INFLUX_USERNAME = ""  # Add if needed
INFLUX_PASSWORD = ""  # Add if needed

# Training settings
MEASUREMENT = "bme688"  # Change to your measurement name
HOURS_BACK = 168 * 8  # 8 weeks (adjust based on your data)
WINDOW_SIZE = 10
MIN_IAQ_ACCURACY = 2  # Only use well-calibrated data


def fetch_training_data():
    """Fetch training data from InfluxDB."""
    print(f"Connecting to InfluxDB at {INFLUX_HOST}:{INFLUX_PORT}...")

    client = DataFrameClient(
        host=INFLUX_HOST,
        port=INFLUX_PORT,
        username=INFLUX_USERNAME,
        password=INFLUX_PASSWORD,
        database=INFLUX_DATABASE
    )

    # Query for all available data or limit by time
    query = f'''
    SELECT temperature, rel_humidity, pressure, gas_resistance, iaq, iaq_accuracy
    FROM {MEASUREMENT}
    WHERE time > now() - {HOURS_BACK}h
    '''

    print(f"Fetching data from last {HOURS_BACK / 168:.1f} weeks...")
    result = client.query(query)

    if MEASUREMENT not in result:
        raise ValueError(f"No data found in measurement '{MEASUREMENT}'")

    df = result[MEASUREMENT]
    print(f"Fetched {len(df)} raw data points")

    # Filter for calibrated data
    df = df[df['iaq_accuracy'] >= MIN_IAQ_ACCURACY]
    df = df.dropna()

    print(f"After filtering (iaq_accuracy >= {MIN_IAQ_ACCURACY}): {len(df)} samples")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"IAQ range: {df['iaq'].min():.1f} - {df['iaq'].max():.1f}")
    print(f"Temperature range: {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")

    return df, client


def create_sliding_windows(features, targets, window_size=10):
    """Create sliding window sequences."""
    windows_X = []
    windows_y = []

    for i in range(len(features) - window_size + 1):
        window_X = features[i:i + window_size].flatten()
        window_y = targets[i + window_size - 1]
        windows_X.append(window_X)
        windows_y.append(window_y)

    return np.array(windows_X), np.array(windows_y)


def calculate_absolute_humidity(temperature, rel_humidity):
    """Calculate absolute humidity from T and RH."""
    a, b = 17.27, 237.7
    alpha = ((a * temperature) / (b + temperature)) + np.log(rel_humidity / 100.0)
    return (6.112 * np.exp(alpha) * 2.1674) / (273.15 + temperature)


def prepare_features(df, window_size=10):
    """Prepare features with engineering."""
    print("\nEngineering features...")

    # Raw features
    features = df[['temperature', 'rel_humidity', 'pressure', 'gas_resistance']].values

    # Feature engineering
    baseline_gas_resistance = np.median(features[:, 3])
    print(f"Baseline gas gas_resistance: {baseline_gas_resistance:.0f} Ω")

    gas_ratio = features[:, 3] / baseline_gas_resistance
    abs_humidity = calculate_absolute_humidity(features[:, 0], features[:, 1])

    # Combine all features
    features_enhanced = np.column_stack([
        features,
        gas_ratio.reshape(-1, 1),
        abs_humidity.reshape(-1, 1)
    ])

    targets = df['iaq'].values

    # Create sliding windows
    print(f"Creating sliding windows (size={window_size})...")
    X, y = create_sliding_windows(features_enhanced, targets, window_size)

    print(f"Created {len(X)} windows with {X.shape[1]} features each")

    return X, y, baseline_gas_resistance


def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=200):
    """Train a model."""
    print(f"\nTraining {model_name}...")

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).reshape(-1, 1)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            print(f'  Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    print(f"  Best validation loss: {best_val_loss:.6f}")
    return best_val_loss


def evaluate_model(model, X_val, y_val, target_scaler):
    """Evaluate model performance."""
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_val)).numpy()

    y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(predictions).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def save_model(model, feature_scaler, target_scaler, model_type,
               window_size, baseline_gas_resistance, model_dir, metrics):
    """Save model and associated files."""
    os.makedirs(model_dir, exist_ok=True)

    # Save scalers
    with open(f'{model_dir}/feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)

    with open(f'{model_dir}/target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)

    # Save config
    config = {
        'baseline_gas_resistance': float(baseline_gas_resistance),
        'trained_date': pd.Timestamp.now().isoformat(),
        'window_size': window_size,
        'mae': float(metrics['mae']),
        'rmse': float(metrics['rmse']),
        'r2': float(metrics['r2']),
        'notes': f'Trained on real BSEC data with {model_type.upper()} architecture'
    }

    with open(f'{model_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save model
    checkpoint = {
        'state_dict': model.state_dict(),
        'model_type': model_type,
        'window_size': window_size,
        'input_dim': 60,  # window_size * 6 features
        'hidden_dims': [64, 32, 16] if model_type == 'mlp' else [32, 16]
    }

    torch.save(checkpoint, f'{model_dir}/model.pt')

    print(f"\n✓ Saved {model_type.upper()} model to {model_dir}/")
    print(f"  - MAE: {metrics['mae']:.2f} IAQ points")
    print(f"  - RMSE: {metrics['rmse']:.2f} IAQ points")
    print(f"  - R²: {metrics['r2']:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING MLP AND KAN MODELS FROM BSEC DATA")
    print("=" * 70)

    # Fetch data
    print("\n[1/5] Fetching training data from InfluxDB...")
    df, client = fetch_training_data()

    if len(df) < 1000:
        print("\n⚠️  WARNING: Less than 1000 samples!")
        print("   Models may not perform well. Consider collecting more data.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            exit()

    # Prepare features
    print("\n[2/5] Preparing features...")
    X, y, baseline_gas_resistance = prepare_features(df, WINDOW_SIZE)

    # Normalize
    print("\n[3/5] Normalizing data...")
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Train MLP
    print("\n[4/5] Training models...")
    print("-" * 70)
    mlp = MLPRegressor(input_dim=60, hidden_dims=[64, 32, 16])
    train_model(mlp, X_train, y_train, X_val, y_val, "MLP", epochs=200)
    mlp_metrics = evaluate_model(mlp, X_val, y_val, target_scaler)

    # Train KAN
    print("-" * 70)
    kan = KANRegressor(input_dim=60, hidden_dims=[32, 16])
    train_model(kan, X_train, y_train, X_val, y_val, "KAN", epochs=200)
    kan_metrics = evaluate_model(kan, X_val, y_val, target_scaler)

    # Save models
    print("\n[5/5] Saving models...")
    print("-" * 70)
    save_model(mlp, feature_scaler, target_scaler, 'mlp', WINDOW_SIZE,
               baseline_gas_resistance, 'trained_models/mlp', mlp_metrics)
    save_model(kan, feature_scaler, target_scaler, 'kan', WINDOW_SIZE,
               baseline_gas_resistance, 'trained_models/kan', kan_metrics)

    # Close connection
    client.close()

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModel Comparison:")
    print(f"  MLP: MAE={mlp_metrics['mae']:.2f}, R²={mlp_metrics['r2']:.4f}")
    print(f"  KAN: MAE={kan_metrics['mae']:.2f}, R²={kan_metrics['r2']:.4f}")
    print("\nRestart your service to use the new models:")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")