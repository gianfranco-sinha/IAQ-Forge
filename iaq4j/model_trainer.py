#!/usr/bin/env python3
"""
Model Trainer CLI Module
Handles training of specific models from the registry.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.train import train_single_model
from training.data_sources import DataSource
from app.config import settings


class ModelTrainer:
    """Handles training of specific models from the registry."""

    def __init__(self):
        db_config = settings.get_database_config()
        self.influx_database = db_config.get("database")

    def train_model(
        self,
        model_type: str,
        epochs: int = 200,
        window_size: int = 10,
        num_records: int = None,
        data_source: DataSource = None,
    ):
        """Train a specific model type."""
        if model_type not in ["mlp", "kan", "lstm", "cnn", "bnn"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        source_label = data_source.name if data_source else "synthetic"
        print(f"Starting training for {model_type.upper()} model...")
        print(f"Configuration:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Window Size: {window_size}")
        print(f"  - Data Source: {source_label}")
        print(f"  - Data Records: {num_records if num_records else 'All available'}")

        result = train_single_model(
            model_type=model_type,
            epochs=epochs,
            window_size=window_size,
            num_records=num_records,
            data_source=data_source,
        )

        if result:
            model_path = f"trained_models/{model_type}"
            print(f"\n✅ Training completed successfully!")
            if result.version:
                print(f"   Version: {result.version}")
            if result.metrics:
                print(f"   MAE={result.metrics['mae']:.2f}, "
                      f"RMSE={result.metrics['rmse']:.2f}, "
                      f"R2={result.metrics['r2']:.4f}")
            print(f"   Model saved to: {model_path}/")
            print(f"   Timestamp: {datetime.now().isoformat()}")
        else:
            raise RuntimeError(f"Training failed for {model_type.upper()} model")

    def train_all_models(
        self, epochs: int = 200, window_size: int = 10, num_records: int = None
    ):
        """Train all models in registry."""
        models_to_train = ["mlp", "kan", "lstm", "cnn", "bnn"]

        for model_type in models_to_train:
            try:
                self.train_model(
                    model_type,
                    epochs=epochs,
                    window_size=window_size,
                    num_records=num_records,
                )
            except RuntimeError as e:
                print(f"❌ {e}")
