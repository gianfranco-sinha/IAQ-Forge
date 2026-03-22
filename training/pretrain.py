"""Self-supervised pre-training for IAQ models.

Trains a masked autoencoder on unlabeled sensor data (no IAQ target needed),
then transfers the learned feature representations to a supervised prediction
model.  This allows leveraging years of raw sensor data that lack BSEC IAQ
labels to improve accuracy on the smaller labeled dataset.

Architecture:
    Encoder (shared with prediction model) → Bottleneck → Decoder (discarded)

Pre-training task:
    Randomly mask 15-30% of input features per window, reconstruct them.
    The encoder learns sensor dynamics, drift patterns, and cross-feature
    correlations without needing IAQ labels.

Usage:
    from training.pretrain import MaskedAutoencoder, pretrain_encoder, transfer_weights

    # Phase 1: pre-train on unlabeled data
    encoder_weights = pretrain_encoder(X_unlabeled, ...)

    # Phase 2: transfer to prediction model
    model = build_model("mlp", ...)
    transfer_weights(model, encoder_weights)

    # Phase 3: fine-tune on labeled data (normal train_model call)
    train_model(model, X_train, y_train, ...)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("training.pretrain")


class MaskedAutoencoder(nn.Module):
    """Masked autoencoder for self-supervised sensor feature learning.

    Shares the encoder architecture with MLPRegressor so weights can be
    transferred directly.  The decoder is symmetric and discarded after
    pre-training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        mask_ratio: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.input_dim = input_dim
        self.mask_ratio = mask_ratio
        self.hidden_dims = hidden_dims

        # Encoder — mirrors MLPRegressor.network (minus the output layer)
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder — symmetric (reverse hidden dims)
        decoder_layers = []
        decoder_dims = list(reversed(hidden_dims))
        prev_dim = hidden_dims[-1]  # bottleneck dim
        for hidden_dim in decoder_dims[1:]:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        # Final reconstruction layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def _apply_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly zero out features and return (masked_input, mask)."""
        mask = torch.bernoulli(
            torch.full_like(x, 1.0 - self.mask_ratio)
        )
        masked_x = x * mask
        return masked_x, mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: mask → encode → decode.

        Returns (reconstruction, original, mask) for loss computation.
        """
        masked_x, mask = self._apply_mask(x)
        encoded = self.encoder(masked_x)
        decoded = self.decoder(encoded)
        return decoded, x, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without masking (for inference / weight transfer)."""
        return self.encoder(x)


def pretrain_encoder(
    X: np.ndarray,
    input_dim: int,
    hidden_dims: List[int] = None,
    dropout: float = 0.2,
    mask_ratio: float = 0.2,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = None,
    val_split: float = 0.1,
    patience: int = 10,
    on_epoch=None,
) -> Dict[str, object]:
    """Pre-train a masked autoencoder on unlabeled sensor windows.

    Args:
        X: Flattened sensor windows, shape (n_windows, input_dim).
            No target labels needed.
        input_dim: Dimension of flattened input.
        hidden_dims: Encoder hidden layer sizes.
        mask_ratio: Fraction of input features to mask per sample.
        epochs: Maximum pre-training epochs.
        patience: Early stopping patience on validation reconstruction loss.
        on_epoch: Optional callback(epoch, train_loss, val_loss).

    Returns:
        Dict with:
            encoder_state_dict: weights for the encoder layers
            hidden_dims: architecture used (needed for transfer)
            train_losses: list of per-epoch training losses
            val_losses: list of per-epoch validation losses
            best_val_loss: best validation reconstruction loss
            epochs_trained: actual number of epochs (may be < epochs if early stopped)
    """
    from training.utils import get_device, seed_everything

    if device is None:
        device = get_device()

    if hidden_dims is None:
        hidden_dims = [64, 32, 16]

    # Split into train/val
    n = len(X)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val
    indices = np.random.permutation(n)
    X_train = X[indices[:n_train]]
    X_val = X[indices[n_train:]]

    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = MaskedAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        mask_ratio=mask_ratio,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_encoder_state = None
    patience_counter = 0

    logger.info(
        "Pre-training masked autoencoder: %d train / %d val windows, "
        "input_dim=%d, hidden=%s, mask_ratio=%.0f%%",
        n_train, n_val, input_dim, hidden_dims, mask_ratio * 100,
    )
    print(f"\nPre-training autoencoder on {device} "
          f"({n_train} train / {n_val} val windows)...")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            reconstructed, original, mask = model(batch_x)

            # Reconstruction loss only on masked positions
            loss = _masked_mse_loss(reconstructed, original, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                reconstructed, original, mask = model(batch_x)
                loss = _masked_mse_loss(reconstructed, original, mask)
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if on_epoch:
            on_epoch(epoch, train_loss, val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch + 1:3d}/{epochs}: "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_encoder_state = {
                k: v.cpu().clone()
                for k, v in model.encoder.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    epochs_trained = len(train_losses)
    logger.info(
        "Pre-training complete: %d epochs, best_val_loss=%.6f",
        epochs_trained, best_val_loss,
    )
    print(f"  Pre-training done: {epochs_trained} epochs, "
          f"best_val_loss={best_val_loss:.6f}")

    return {
        "encoder_state_dict": best_encoder_state,
        "hidden_dims": hidden_dims,
        "input_dim": input_dim,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs_trained,
    }


def transfer_weights(
    model: nn.Module,
    pretrain_result: Dict[str, object],
    freeze_encoder: bool = False,
) -> int:
    """Transfer pre-trained encoder weights into a prediction model.

    Maps autoencoder encoder layers → MLPRegressor.network layers.
    The encoder's (Linear, BatchNorm, ReLU, Dropout) blocks map 1:1
    to the MLP's (Linear, BatchNorm, ReLU, Dropout) blocks.

    Args:
        model: Target model (MLPRegressor or similar).
        pretrain_result: Output of pretrain_encoder().
        freeze_encoder: If True, freeze transferred layers (train head only).

    Returns:
        Number of parameters transferred.
    """
    encoder_state = pretrain_result["encoder_state_dict"]

    # Find the Sequential containing the model's layers
    if hasattr(model, "network"):
        # MLPRegressor: network is Sequential of [Linear, BN, ReLU, Dropout, ..., Linear(out)]
        target = model.network
    elif hasattr(model, "_hidden"):
        # BNNRegressor: _hidden is Sequential (but BayesianLinear — skip, incompatible)
        logger.warning("BNN uses BayesianLinear — pre-trained weights not directly compatible")
        return 0
    else:
        logger.warning("Model %s has no recognized encoder structure", type(model).__name__)
        return 0

    # The encoder state keys are like "0.weight", "0.bias", "1.weight", etc.
    # These map positionally to the target Sequential's modules
    transferred = 0
    target_state = target.state_dict()
    for key, value in encoder_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value
            transferred += value.numel()
        else:
            logger.debug("Skipping encoder key %s (shape mismatch or missing)", key)

    target.load_state_dict(target_state)

    if freeze_encoder:
        # Freeze all encoder layers, leave the final output layer trainable
        hidden_dims = pretrain_result["hidden_dims"]
        # Each hidden block = 4 modules: Linear, BN, ReLU, Dropout
        n_encoder_modules = len(hidden_dims) * 4
        for i, child in enumerate(target.children()):
            if i < n_encoder_modules:
                for param in child.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        logger.info(
            "Froze encoder: %d params frozen, %d trainable (head only)",
            frozen, trainable,
        )
        print(f"  Encoder frozen: {frozen} params frozen, {trainable} trainable")

    logger.info("Transferred %d parameters from pre-trained encoder", transferred)
    print(f"  Transferred {transferred:,} pre-trained parameters")
    return transferred


def _masked_mse_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss computed only on masked (zeroed-out) positions.

    This forces the encoder to learn cross-feature correlations — it can
    only reconstruct masked values from the unmasked context.
    """
    # mask=1 means feature was visible, mask=0 means it was masked
    inverted_mask = 1.0 - mask
    n_masked = inverted_mask.sum().clamp(min=1.0)
    loss = ((reconstructed - original) ** 2 * inverted_mask).sum() / n_masked
    return loss
