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
        choices=["synthetic", "influxdb", "csv"],
        default="synthetic",
        help="Data source for training (default: synthetic)",
    )
    train_parser.add_argument(
        "--csv-path",
        type=str,
        help="Path to CSV file (required when --data-source csv)",
    )

    # List models command
    list_parser = subparsers.add_parser(
        "list", help="List available models in registry"
    )

    # Map fields command
    map_parser = subparsers.add_parser(
        "map-fields", help="Map CSV column headers to iaq4j features"
    )
    map_parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to CSV file whose columns to map",
    )
    map_parser.add_argument(
        "--sample-rows",
        type=int,
        default=10,
        help="Number of sample rows for range validation (default: 10)",
    )
    map_parser.add_argument(
        "--threshold",
        type=int,
        default=70,
        help="Fuzzy match score threshold 0-100 (default: 70)",
    )
    map_parser.add_argument(
        "--save",
        action="store_true",
        help="Save mapping to model_config.yaml under sensor.field_mapping",
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
            elif args.data_source == "csv":
                if not args.csv_path:
                    print("❌ --csv-path is required when --data-source is csv")
                    sys.exit(1)
                from training.data_sources import CSVDataSource
                data_source = CSVDataSource(args.csv_path)

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

    elif args.command == "map-fields":
        import app.builtin_profiles  # noqa: F401
        from app.field_mapper import FieldMapper
        from app.profiles import get_sensor_profile

        profile = get_sensor_profile()
        mapper = FieldMapper(profile, fuzzy_threshold=args.threshold)

        headers, sample_values = FieldMapper.sample_csv(args.source, n_rows=args.sample_rows)
        result = mapper.map_fields(headers, sample_values=sample_values)

        print(f"\nField mapping for: {args.source}")
        print(f"Sensor profile: {profile.name}\n")
        print(mapper.format_report(result))

        if args.save and result.matches:
            import yaml

            config_path = project_root / "model_config.yaml"
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            mapping = {m.source_field: m.target_feature for m in result.matches}
            cfg.setdefault("sensor", {})["field_mapping"] = mapping

            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

            print(f"\nMapping saved to {config_path} under sensor.field_mapping")


if __name__ == "__main__":
    main()
