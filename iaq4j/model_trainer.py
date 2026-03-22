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
from integrations.mlflow.tracker import ExperimentTracker, MLflowTracker, NullTracker
from app.config import settings


class ModelTrainer:
    """Handles training of specific models from the registry."""

    def __init__(self):
        db_config = settings.get_database_config()
        self.influx_database = db_config.get("database")
        try:
            self._tracker: ExperimentTracker = MLflowTracker()
        except ImportError:
            self._tracker = NullTracker()

    def train_model(
        self,
        model_type: str,
        epochs: int = 200,
        window_size: int = None,
        num_records: int = None,
        data_source: DataSource = None,
        resume: bool = False,
        early_stopping_patience: int = None,
    ):
        """Train a specific model type."""
        if model_type not in ["mlp", "kan", "lstm", "cnn", "bnn"]:
            from app.exceptions import ConfigurationError
            raise ConfigurationError(f"Unsupported model type: {model_type}")

        effective_window = window_size or settings.get_model_config(model_type).get("window_size", 10)
        source_label = data_source.name if data_source else "synthetic"
        print(f"Starting training for {model_type.upper()} model...")
        print(f"Configuration:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Window Size: {effective_window}")
        print(f"  - Data Source: {source_label}")
        print(f"  - Data Records: {num_records if num_records else 'All available'}")
        if resume:
            print(f"  - Resume: from checkpoint if available")

        result = train_single_model(
            model_type=model_type,
            epochs=epochs,
            window_size=window_size,
            num_records=num_records,
            data_source=data_source,
            resume=resume,
            tracker=self._tracker,
            early_stopping_patience=early_stopping_patience,
        )

        if result and result.interrupted:
            print(f"\n⏸️  Training interrupted — checkpoint saved. Resume with --resume")
            return result

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
            # Print data cleanse report
            report = result.preprocessing_report
            if report.issues:
                severity_prefix = {"error": "E", "warning": "W", "info": "I"}
                print(f"\n   Data Cleanse Report ({len(report.issues)} issue(s)):")
                for issue in report.issues:
                    prefix = severity_prefix.get(issue.severity.value, "?")
                    row_suffix = f" ({issue.rows_affected} rows)" if issue.rows_affected else ""
                    print(f"     [{prefix}] {issue.message}{row_suffix}")
        else:
            from app.exceptions import TrainingDivergedError
            raise TrainingDivergedError(f"Training failed for {model_type.upper()} model")

    def train_all_models(
        self, epochs: int = 200, window_size: int = None, num_records: int = None,
        resume: bool = False, early_stopping_patience: int = None,
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
                    resume=resume,
                    early_stopping_patience=early_stopping_patience,
                )
            except RuntimeError as e:
                print(f"❌ {e}")
