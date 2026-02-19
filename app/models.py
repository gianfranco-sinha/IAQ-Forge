# ============================================================================
# File: app/models.py - Complete Version with All Models
# ============================================================================
import logging
import torch
import torch.nn as nn
import numpy as np
from app.kan import KAN

logger = logging.getLogger(__name__)


class BayesianLinear(nn.Module):
    """Linear layer with learned weight distributions for Bayes by Backprop.

    Each forward pass samples weights from N(mu, softplus(rho)), giving
    intrinsic epistemic uncertainty without dropout.
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

        # Kaiming init for mu
        nn.init.kaiming_normal_(self.weight_mu, nonlinearity="relu")
        nn.init.zeros_(self.bias_mu)

        # KL divergence accumulated during forward pass
        self._kl: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))  # softplus
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # Reparameterization trick
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)

        # Analytical KL divergence vs N(0, prior_sigma)
        self._kl = self._kl_divergence(
            self.weight_mu, weight_sigma, self.prior_sigma
        ) + self._kl_divergence(
            self.bias_mu, bias_sigma, self.prior_sigma
        )

        return nn.functional.linear(x, weight, bias)

    @staticmethod
    def _kl_divergence(mu, sigma, prior_sigma):
        """KL(N(mu, sigma^2) || N(0, prior_sigma^2))."""
        prior_var = prior_sigma ** 2
        return 0.5 * torch.sum(
            sigma ** 2 / prior_var + mu ** 2 / prior_var - 1.0 - 2.0 * torch.log(sigma / prior_sigma)
        )

    @property
    def kl_divergence(self) -> torch.Tensor:
        return self._kl


class BNNRegressor(nn.Module):
    """Bayesian Neural Network for IAQ prediction.

    Same architecture as MLPRegressor but with BayesianLinear layers instead
    of nn.Linear.  No Dropout — uncertainty comes from weight sampling.
    """

    def __init__(self, input_dim, hidden_dims=None, prior_sigma=1.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.prior_sigma = prior_sigma
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_sigma))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self._hidden = nn.Sequential(*layers)
        self._head = BayesianLinear(prev_dim, 1, prior_sigma)

    def forward(self, x):
        return self._head(self._hidden(x))

    def kl_loss(self) -> torch.Tensor:
        """Sum KL divergence from all BayesianLinear layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, BayesianLinear):
                total = total + m.kl_divergence
        return total


class MLPRegressor(nn.Module):
    """MLP for IAQ prediction (Baseline Model)."""

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


class CNNRegressor(nn.Module):
    """CNN for IAQ prediction from temporal sequences."""

    def __init__(
        self,
        window_size=10,
        num_features=6,
        num_filters=[64, 128, 256],
        kernel_sizes=[3, 3, 3],
        dropout=0.3,
    ):
        super(CNNRegressor, self).__init__()

        self.window_size = window_size
        self.num_features = num_features

        # Build convolutional layers
        conv_layers = []
        in_channels = num_features

        for num_filter, kernel_size in zip(num_filters, kernel_sizes):
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filter,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            conv_layers.append(nn.BatchNorm1d(num_filter))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = num_filter

        self.conv = nn.Sequential(*conv_layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(num_filters[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # Reshape from (batch, window_size * num_features) to (batch, num_features, window_size)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.window_size, self.num_features)
        x = x.permute(0, 2, 1)  # (batch, num_features, window_size)

        # Convolutional layers
        x = self.conv(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Fully connected
        x = self.fc(x)

        return x


class LSTMRegressor(nn.Module):
    """LSTM for IAQ prediction from temporal sequences."""

    def __init__(
        self,
        window_size=10,
        num_features=6,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    ):
        super(LSTMRegressor, self).__init__()

        self.window_size = window_size
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Fully connected layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # Reshape from (batch, window_size * num_features) to (batch, window_size, num_features)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.window_size, self.num_features)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take the output from the last time step
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            last_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            last_hidden = hidden[-1, :, :]

        # Fully connected layers
        output = self.fc(last_hidden)

        return output


class KANRegressor(nn.Module):
    """KAN for IAQ prediction."""

    def __init__(self, input_dim, hidden_dims=[32, 16]):
        super(KANRegressor, self).__init__()
        layers = [input_dim] + hidden_dims + [1]
        self.kan = KAN(layers)

    def forward(self, x):
        return self.kan(x)


MODEL_REGISTRY = {
    "mlp": MLPRegressor,
    "lstm": LSTMRegressor,
    "cnn": CNNRegressor,
    "kan": KANRegressor,
    "bnn": BNNRegressor,
}


def build_model(model_type: str, window_size: int = 10, num_features: int = 6) -> nn.Module:
    """Build a model instance from the registry using YAML config.

    Args:
        model_type: One of "mlp", "kan", "lstm", "cnn".
        window_size: Sliding window size (used to compute input_dim for MLP/KAN).
        num_features: Number of features per timestep.

    Returns:
        An instantiated nn.Module ready for training.
    """
    from app.config import settings

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of {list(MODEL_REGISTRY)}")

    ModelCls = MODEL_REGISTRY[model_type]
    cfg = settings.get_model_config(model_type)

    if model_type == "mlp":
        return ModelCls(
            input_dim=window_size * num_features,
            hidden_dims=cfg.get("hidden_dims", [64, 32, 16]),
            dropout=cfg.get("dropout", 0.2),
        )
    elif model_type == "kan":
        return ModelCls(
            input_dim=window_size * num_features,
            hidden_dims=cfg.get("hidden_dims", [32, 16]),
        )
    elif model_type == "lstm":
        return ModelCls(
            window_size=cfg.get("window_size", window_size),
            num_features=cfg.get("num_features", num_features),
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
            bidirectional=cfg.get("bidirectional", True),
        )
    elif model_type == "cnn":
        return ModelCls(
            window_size=cfg.get("window_size", window_size),
            num_features=cfg.get("num_features", num_features),
            num_filters=cfg.get("num_filters", [64, 128, 256]),
            kernel_sizes=cfg.get("kernel_sizes", [3, 3, 3]),
            dropout=cfg.get("dropout", 0.3),
        )
    elif model_type == "bnn":
        return ModelCls(
            input_dim=window_size * num_features,
            hidden_dims=cfg.get("hidden_dims", [64, 32, 16]),
            prior_sigma=cfg.get("prior_sigma", 1.0),
        )

    return ModelCls(**cfg)


class IAQPredictor:
    """Main predictor class that handles model loading and inference with sliding window."""

    def __init__(
        self, model_type: str = "mlp", window_size: int = 10, model_path: str = None
    ):
        self.model_type: str = model_type
        self.window_size: int = window_size
        self.model_path: str = model_path
        self.model = None  # Type: torch.nn.Module
        self.feature_scaler = None  # StandardScaler for engineered features
        self.target_scaler = None  # MinMaxScaler for target
        self.config = None  # Type: dict
        self.device: str = "cpu"
        self._baselines: dict = {}

        # Sliding window buffer for all models (all use windowed input)
        self.buffer: list = []

        self._model_registry = MODEL_REGISTRY

        from app.profiles import get_iaq_standard, get_sensor_profile
        self.sensor_profile = get_sensor_profile()
        self.iaq_standard = get_iaq_standard()

    def load_model(self, model_path: str) -> bool:
        """Load model from saved artifacts."""
        import json
        import joblib
        from pathlib import Path

        model_dir = Path(model_path)

        try:
            # Load config
            config_path = model_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path) as f:
                self.config = json.load(f)

            # Get model class and create instance
            if self.model_type not in self._model_registry:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            ModelClass = self._model_registry[self.model_type]

            # Load model weights first to get parameters
            weights_path = model_dir / "model.pt"
            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights not found: {weights_path}")

            model_data = torch.load(weights_path, map_location=self.device)

            # Extract model parameters from saved data or config
            model_params = self.config.get("model_params", {})

            if isinstance(model_data, dict):
                if self.model_type in ["mlp", "kan", "bnn"]:
                    # MLP, KAN, and BNN specific parameters
                    if "input_dim" in model_data:
                        model_params["input_dim"] = model_data["input_dim"]
                    if "hidden_dims" in model_data:
                        model_params["hidden_dims"] = model_data["hidden_dims"]
                    if "dropout" in model_data:
                        model_params["dropout"] = model_data["dropout"]
                    if "prior_sigma" in model_data:
                        model_params["prior_sigma"] = model_data["prior_sigma"]
                elif self.model_type in ["lstm", "cnn"]:
                    # LSTM and CNN specific parameters
                    if "window_size" in model_data:
                        model_params["window_size"] = model_data["window_size"]
                    if "num_features" in model_data:
                        model_params["num_features"] = model_data["num_features"]

            num_features = self.sensor_profile.total_features

            # Add required parameters based on model type
            if self.model_type in ["mlp", "kan", "bnn"]:
                if "input_dim" not in model_params:
                    model_params["input_dim"] = self.window_size * num_features
            elif self.model_type in ["lstm", "cnn"]:
                model_params.setdefault("window_size", self.window_size)
                model_params.setdefault("num_features", num_features)

            self.model = ModelClass(**model_params)

            # Load model weights
            if isinstance(model_data, dict) and "state_dict" in model_data:
                self.model.load_state_dict(model_data["state_dict"])
            else:
                self.model.load_state_dict(model_data)
            self.model.to(self.device)
            self.model.eval()

            # Load scalers
            feature_scaler_path = model_dir / "feature_scaler.pkl"
            target_scaler_path = model_dir / "target_scaler.pkl"
            if feature_scaler_path.exists():
                self.feature_scaler = joblib.load(feature_scaler_path)
            if target_scaler_path.exists():
                self.target_scaler = joblib.load(target_scaler_path)

            # Load baselines from config (new format or legacy)
            self._baselines = self.config.get("baselines", {})
            if not self._baselines and self.config.get("baseline_gas_resistance"):
                self._baselines = {
                    "voc_resistance": self.config["baseline_gas_resistance"]
                }

            # Read window_size from config if saved
            if "window_size" in self.config:
                self.window_size = self.config["window_size"]

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _forward_pass_mc(self, features_tensor, n_samples=10):
        """Run n_samples stochastic forward passes.

        Operates on an already-prepared tensor — does NOT touch the buffer.
        Returns array of shape (n_samples,) with IAQ-scale predictions.

        - BNN: each forward pass samples different weights (intrinsic stochasticity).
        - Models with Dropout: MC dropout (enable dropout in eval).
        - Deterministic models (e.g. KAN): single prediction repeated.
        """
        is_bnn = isinstance(self.model, BNNRegressor)
        has_dropout = any(
            isinstance(m, nn.Dropout) for m in self.model.modules()
        )

        if not has_dropout and not is_bnn:
            with torch.no_grad():
                pred = self.model(features_tensor)
                scaled = pred.cpu().numpy()
            if self.target_scaler is not None:
                val = float(self.target_scaler.inverse_transform(scaled)[0, 0])
            else:
                val = float(scaled[0, 0])
            return np.full(n_samples, val)

        # For BNN: forward passes are stochastic even in eval mode.
        # For MC dropout: enable only Dropout layers (keep BatchNorm in eval).
        if has_dropout:
            for m in self.model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(features_tensor)
                scaled = pred.cpu().numpy()
            if self.target_scaler is not None:
                val = float(self.target_scaler.inverse_transform(scaled)[0, 0])
            else:
                val = float(scaled[0, 0])
            predictions.append(val)

        self.model.eval()
        return np.array(predictions)

    def predict(self, readings: dict = None, n_mc_samples: int = 1,
                **kwargs) -> dict:
        """Predict IAQ from sensor readings.

        Accepts either a readings dict or keyword arguments for backward
        compatibility (temperature, rel_humidity, pressure, voc_resistance).

        Args:
            readings: sensor readings dict.
            n_mc_samples: number of MC dropout forward passes (1 = deterministic).

        Flow: engineer features → buffer → flatten → scale → forward → inverse-scale → clamp → categorize
        """
        if readings is None:
            readings = kwargs

        if self.model is None:
            return {"iaq": None, "status": "error", "message": "Model not loaded"}

        try:
            # Step 1: profile-driven feature engineering
            features = self.sensor_profile.engineer_features_single(
                readings, self._baselines
            )

            # Step 2: buffer (all models use windowed input)
            self.buffer.append(features)
            if len(self.buffer) > self.window_size:
                self.buffer.pop(0)

            if len(self.buffer) < self.window_size:
                return {
                    "iaq": None,
                    "status": "buffering",
                    "buffer_size": len(self.buffer),
                    "required": self.window_size,
                    "message": f"Collecting data... {len(self.buffer)}/{self.window_size}",
                }

            # Step 3: flatten window → scale
            window_flat = np.array(self.buffer).flatten().reshape(1, -1)

            if self.feature_scaler is not None:
                window_flat = self.feature_scaler.transform(window_flat)

            features_tensor = torch.FloatTensor(window_flat).to(self.device)

            # Step 4: forward pass (with optional MC dropout)
            uncertainty_info = None
            if n_mc_samples > 1:
                mc_predictions = self._forward_pass_mc(
                    features_tensor, n_mc_samples
                )
                iaq_value = float(np.mean(mc_predictions))
                is_bnn = isinstance(self.model, BNNRegressor)
                has_dropout = any(
                    isinstance(m, nn.Dropout) for m in self.model.modules()
                )
                if is_bnn:
                    method = "weight_sampling"
                elif has_dropout:
                    method = "mc_dropout"
                else:
                    method = "deterministic"
                uncertainty_info = {
                    "std": float(np.std(mc_predictions)),
                    "ci_lower": float(np.percentile(mc_predictions, 2.5)),
                    "ci_upper": float(np.percentile(mc_predictions, 97.5)),
                    "method": method,
                }
            else:
                with torch.no_grad():
                    prediction = self.model(features_tensor)
                    scaled_value = prediction.cpu().numpy()
                if self.target_scaler is not None:
                    iaq_value = float(
                        self.target_scaler.inverse_transform(scaled_value)[0, 0]
                    )
                else:
                    iaq_value = float(scaled_value[0, 0])

            # Step 5: standard-driven clamp and categorize
            iaq_value = self.iaq_standard.clamp(iaq_value)
            category = self.iaq_standard.categorize(iaq_value)

            # Build engineered features dict for observation
            eng_names = self.sensor_profile.engineered_feature_names
            eng_values = features[-len(eng_names):] if eng_names else []
            eng_dict = dict(zip(eng_names, [float(v) for v in eng_values]))

            # Build predicted block
            predicted = {
                "mean": iaq_value,
                "category": category,
                "iaq_standard": self.iaq_standard.name,
            }
            if uncertainty_info is not None:
                predicted["uncertainty"] = uncertainty_info

            result = {
                # Backward-compatible top-level fields
                "iaq": iaq_value,
                "category": category,
                "status": "ready",
                "model_type": self.model_type,
                "raw_inputs": readings,
                "buffer_size": len(self.buffer),
                "required": self.window_size,
                # Structured inference fields
                "observation": {
                    "sensor_type": self.sensor_profile.name,
                    "readings": readings,
                    "engineered_features": eng_dict or None,
                },
                "predicted": predicted,
                "inference": {
                    "model_type": self.model_type,
                    "window_size": self.window_size,
                    "buffer_size": len(self.buffer),
                    "uncertainty_method": (
                        uncertainty_info["method"] if uncertainty_info else None
                    ),
                    "mc_samples": n_mc_samples if n_mc_samples > 1 else None,
                },
            }

            return result

        except Exception as e:
            return {
                "iaq": None,
                "status": "error",
                "message": f"Prediction failed: {str(e)}",
            }

    def reset_buffer(self) -> None:
        """Reset the sliding window buffer."""
        self.buffer = []
