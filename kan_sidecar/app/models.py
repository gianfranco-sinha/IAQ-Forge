"""KAN model definition for the sidecar container."""

import torch.nn as nn
from app.kan import KAN


class KANRegressor(nn.Module):
    """KAN for IAQ prediction."""

    def __init__(self, input_dim, hidden_dims=None, grid_size=5, spline_order=3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]
        layers = [input_dim] + hidden_dims + [1]
        self.kan = KAN(layers, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        return self.kan(x)

    def regularization_loss(self):
        return self.kan.regularization_loss()
