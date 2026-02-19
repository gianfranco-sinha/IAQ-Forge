#!/usr/bin/env python3
"""
iaq4j - CLI Training Module

Usage:
    python -m iaq4j train --model mlp
    python -m iaq4j train --model kan --epochs 100
    python -m iaq4j train --model all

Supported models: mlp, kan, lstm, cnn, all
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iaq4j.model_trainer import ModelTrainer


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="iaq4j CLI - Model Training and Data Management"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a specific model")
    train_parser.add_argument(
        "--model",
        choices=["mlp", "kan", "lstm", "cnn", "bnn", "all"],
        required=True,
        help='Model type to train (or "all" for all models)',
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)",
    )
    train_parser.add_argument(
        "--window-size", type=int, default=10, help="Sliding window size (default: 10)"
    )
    train_parser.add_argument(
        "--data-records",
        type=int,
        help="Number of records to fetch from database (optional)",
    )
    train_parser.add_argument(
        "--data-source",
        choices=["synthetic", "influxdb"],
        default="synthetic",
        help="Data source for training (default: synthetic)",
    )

    # List models command
    list_parser = subparsers.add_parser(
        "list", help="List available models in registry"
    )

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "train":
        trainer = ModelTrainer()

        if args.model == "all":
            models_to_train = ["mlp", "kan", "lstm", "cnn", "bnn"]
        else:
            models_to_train = [args.model]

        for model_type in models_to_train:
            print(f"\n{'=' * 60}")
            print(f"Training {model_type.upper()} model...")
            print(f"{'=' * 60}")

            # Build data source
            data_source = None
            if args.data_source == "influxdb":
                from training.data_sources import InfluxDBSource
                data_source = InfluxDBSource()

            try:
                trainer.train_model(
                    model_type=model_type,
                    epochs=args.epochs,
                    window_size=args.window_size,
                    num_records=args.data_records,
                    data_source=data_source,
                )
                print(f"✅ {model_type.upper()} training completed successfully")
            except Exception as e:
                print(f"❌ {model_type.upper()} training failed: {e}")
                continue

    elif args.command == "list":
        from app.models import MODEL_REGISTRY

        print("Available models in registry:")
        for model_type in MODEL_REGISTRY:
            print(f"  - {model_type}")


if __name__ == "__main__":
    main()
