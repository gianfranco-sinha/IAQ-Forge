# ============================================================================
# File: app/models.py
# ============================================================================
import torch
import torch.nn as nn
from efficient_kan import KAN
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import json


class MLPRegressor(nn.Module):
    """MLP for IAQ prediction."""

    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout=0.2):
        super(MLPRegressor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class KANRegressor(nn.Module):
    """KAN for IAQ prediction."""

    def __init__(self, input_dim, hidden_dims=[32, 16]):
        super(KANRegressor, self).__init__()
        layers = [input_dim] + hidden_dims + [1]
        self.kan = KAN(layers)

    def forward(self, x):
        return self.kan(x)


class IAQPredictor:
    """Wrapper for IAQ prediction with feature engineering."""

    def __init__(self, model_type='mlp', window_size=10):
        self.model_type = model_type
        self.window_size = window_size
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.buffer = []
        self.config = {}

    def _calculate_absolute_humidity(self, temperature, relative_humidity):
        """Calculate absolute humidity."""
        a, b = 17.27, 237.7
        alpha = ((a * temperature) / (b + temperature)) + np.log(rel_humidity / 100.0)
        return (6.112 * np.exp(alpha) * 2.1674) / (273.15 + temperature)

    def _engineer_features(self, temp, rel_humidity, pressure, gas_resistance):
        """Engineer features from raw sensor data."""
        # Gas resistance ratio (requires baseline from config)
        baseline_gas_resistance = self.config.get('baseline_gas_resistance', 100000)
        gas_ratio = gas_resistance / baseline_gas_resistance

        # Absolute humidity
        abs_humidity = self._calculate_absolute_humidity(temp, rel_humidity)

        return np.array([temp, rel_humidity, pressure, gas_resistance, gas_ratio, abs_humidity])

    def add_to_buffer(self, sample):
        """Add sample to sliding window buffer."""
        self.buffer.append(sample)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def predict(self, temperature, rel_humidity, pressure, gas_resistance):
        """
        Predict IAQ from sensor readings.

        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity (%)
            pressure: Pressure in hPa
            resistance: Gas resistance in Ohms

        Returns:
            dict with iaq prediction and metadata
        """
        # Engineer features
        features = self._engineer_features(temperature, rel_humidity, pressure, gas_resistance)

        # Add to buffer
        self.add_to_buffer(features)

        # Check if buffer is full
        if len(self.buffer) < self.window_size:
            return {
                'iaq': None,
                'status': 'buffering',
                'buffer_size': len(self.buffer),
                'required': self.window_size,
                'message': f'Collecting initial data ({len(self.buffer)}/{self.window_size})'
            }

        # Create window and normalize
        window = np.array(self.buffer).flatten().reshape(1, -1)
        normalized = self.feature_scaler.transform(window)

        # Predict
        self.model.eval()
        with torch.no_grad():
            prediction_scaled = self.model(torch.FloatTensor(normalized)).numpy()

        # Inverse transform to IAQ scale (0-500)
        iaq = self.target_scaler.inverse_transform(prediction_scaled)[0][0]

        # Clip to valid range
        iaq = np.clip(iaq, 0, 500)

        # Determine air quality category
        if iaq <= 50:
            category = "Excellent"
        elif iaq <= 100:
            category = "Good"
        elif iaq <= 150:
            category = "Lightly polluted"
        elif iaq <= 200:
            category = "Moderately polluted"
        elif iaq <= 250:
            category = "Heavily polluted"
        elif iaq <= 350:
            category = "Severely polluted"
        else:
            category = "Extremely polluted"

        return {
            'iaq': float(iaq),
            'category': category,
            'status': 'ready',
            'model_type': self.model_type,
            'raw_inputs': {
                'temperature': float(temperature),
                'rel_humidity': float(rel_humidity),
                'pressure': float(pressure),
                'gas_resistance': float(gas_resistance)
            }
        }

    def load_model(self, model_dir):
        """Load trained model, scalers, and config."""
        # Load config
        with open(f'{model_dir}/config.json', 'r') as f:
            self.config = json.load(f)

        # Load scalers
        with open(f'{model_dir}/feature_scaler.pkl', 'rb') as f:
            self.feature_scaler = pickle.load(f)

        with open(f'{model_dir}/target_scaler.pkl', 'rb') as f:
            self.target_scaler = pickle.load(f)

        # Load model
        checkpoint = torch.load(f'{model_dir}/model.pt', map_location='cpu')

        if self.model_type == 'mlp':
            self.model = MLPRegressor(
                checkpoint['input_dim'],
                checkpoint['hidden_dims']
            )
        else:
            self.model = KANRegressor(
                checkpoint['input_dim'],
                checkpoint['hidden_dims']
            )

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def save_model(self, model_dir):
        """Save model, scalers, and config."""
        import os
        os.makedirs(model_dir, exist_ok=True)

        # Save config
        with open(f'{model_dir}/config.json', 'w') as f:
            json.dump(self.config, f)

        # Save scalers
        with open(f'{model_dir}/feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.feature_scaler, f)

        with open(f'{model_dir}/target_scaler.pkl', 'wb') as f:
            pickle.dump(self.target_scaler, f)

        # Save model
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'window_size': self.window_size,
        }

        if self.model_type == 'mlp':
            checkpoint['input_dim'] = self.model.network[0].in_features
            checkpoint['hidden_dims'] = [64, 32, 16]  # Should extract from model
        else:
            checkpoint['input_dim'] = 60  # window_size * 6 features
            checkpoint['hidden_dims'] = [32, 16]

        torch.save(checkpoint, f'{model_dir}/model.pt')