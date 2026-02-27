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
        choices=["synthetic", "influxdb", "csv", "labelstudio"],
        default="synthetic",
        help="Data source for training (default: synthetic)",
    )
    train_parser.add_argument(
        "--ls-project-id",
        type=int,
        help="Label Studio project ID (required when --data-source labelstudio)",
    )
    train_parser.add_argument(
        "--ls-url",
        type=str,
        help="Label Studio server URL (overrides label_studio.url in model_config.yaml)",
    )
    train_parser.add_argument(
        "--database",
        type=str,
        help="InfluxDB database name (for influxdb source)",
    )
    train_parser.add_argument(
        "--hours-back",
        type=int,
        help="Hours of data to fetch from InfluxDB (default: 1344 ≈ 56 days)",
    )
    train_parser.add_argument(
        "--max-records",
        type=int,
        help="Maximum number of records to fetch from InfluxDB",
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

    # Version command
    version_parser = subparsers.add_parser(
        "version", help="Show active model versions (semver)"
    )

    # Verify command
    verify_parser = subparsers.add_parser(
        "verify", help="Verify Merkle tree provenance of trained models"
    )
    verify_parser.add_argument(
        "--model",
        choices=["mlp", "kan", "lstm", "cnn", "bnn", "all"],
        default="all",
        help="Model type to verify (default: all)",
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
    map_parser.add_argument(
        "--backend",
        choices=["fuzzy", "ollama"],
        default="fuzzy",
        help="Mapping backend: fuzzy (Tier 1+2) or ollama (adds Tier 3 LLM) (default: fuzzy)",
    )
    map_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip interactive confirmation (accept mapping automatically)",
    )

    return parser


def _interactive_confirm(result, profile):
    """Prompt user to accept, reject, or edit proposed field mappings.

    Returns:
        List of confirmed FieldMatch objects, or None if user aborts.
    """
    try:
        choice = input("\nAccept this mapping? [Y/n/e(dit)] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if choice in ("", "y", "yes"):
        return result.matches

    if choice in ("n", "no"):
        return None

    if choice not in ("e", "edit"):
        print(f"Unknown option: {choice}")
        return None

    # Edit mode: walk through each match
    available_features = list(profile.feature_quantities.keys())
    confirmed = []
    for m in result.matches:
        try:
            resp = (
                input(
                    f"  {m.source_field} -> {m.target_feature} ({m.confidence:.0%} {m.method})"
                    f"  [a(ccept)/o(verride)/s(kip)] "
                )
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            return None

        if resp in ("", "a", "accept"):
            confirmed.append(m)
        elif resp in ("s", "skip"):
            continue
        elif resp in ("o", "override"):
            print(f"    Available features: {', '.join(available_features)}")
            try:
                new_target = input("    Enter target feature name: ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if new_target in available_features:
                from app.field_mapper import FieldMatch

                confirmed.append(
                    FieldMatch(
                        source_field=m.source_field,
                        target_quantity=profile.feature_quantities[new_target],
                        target_feature=new_target,
                        confidence=1.0,
                        method="manual",
                    )
                )
            else:
                print(f"    Unknown feature '{new_target}', skipping.")
        else:
            print(f"    Unknown option '{resp}', accepting.")
            confirmed.append(m)

    return confirmed


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

                data_source = InfluxDBSource(
                    database=args.database,
                    max_records=args.max_records,
                    hours_back=args.hours_back,
                )
            elif args.data_source == "csv":
                if not args.csv_path:
                    print("❌ --csv-path is required when --data-source is csv")
                    sys.exit(1)
                from training.data_sources import CSVDataSource

                data_source = CSVDataSource(args.csv_path)
            elif args.data_source == "labelstudio":
                from training.data_sources import LabelStudioDataSource

                data_source = LabelStudioDataSource(
                    project_id=args.ls_project_id,
                    url=args.ls_url,
                )

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

    elif args.command == "version":
        import json

        from app.config import settings

        manifest_path = Path(settings.TRAINED_MODELS_BASE) / "MANIFEST.json"
        if not manifest_path.exists():
            print("No MANIFEST.json found. Train a model first.")
            return

        with open(manifest_path) as f:
            central = json.load(f)

        active_runs = [r for r in central.get("runs", []) if r.get("is_active")]

        if not active_runs:
            print("No active model versions found.")
            return

        print(
            f"\n{'Model':<8} {'Version':<16} {'Schema FP':<14} {'MAE':>8} {'RMSE':>8} {'R2':>8}  {'Trained'}"
        )
        print("-" * 88)

        for run in sorted(active_runs, key=lambda r: r.get("model_type", "")):
            model_type = run.get("model_type", "?")
            version = run.get("version", "?")
            schema_fp = run.get("schema_fingerprint", "—")
            metrics = run.get("metrics", {})
            mae = f"{metrics['mae']:.2f}" if "mae" in metrics else "—"
            rmse = f"{metrics['rmse']:.2f}" if "rmse" in metrics else "—"
            r2 = f"{metrics['r2']:.4f}" if "r2" in metrics else "—"
            trained = run.get("timestamp", "—")[:19]

            print(
                f"{model_type:<8} {version:<16} {schema_fp:<14} {mae:>8} {rmse:>8} {r2:>8}  {trained}"
            )

        print()

    elif args.command == "verify":
        from app.config import settings
        from training.merkle import verify_merkle_tree

        model_types = (
            ["mlp", "kan", "lstm", "cnn", "bnn"]
            if args.model == "all"
            else [args.model]
        )
        any_checked = False

        for model_type in model_types:
            model_dir = Path(settings.TRAINED_MODELS_BASE) / model_type
            if not model_dir.exists():
                continue

            any_checked = True
            result = verify_merkle_tree(model_dir)

            if result["valid"]:
                print(
                    f"  [PASS] {model_type:<6} merkle_root={result['root_hash'][:16]}..."
                )
            else:
                print(f"  [FAIL] {model_type:<6}")
                for mismatch in result["mismatches"]:
                    print(f"         - {mismatch}")

        if not any_checked:
            print("No trained models found. Train a model first.")

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

        headers, sample_values = FieldMapper.sample_csv(
            args.source, n_rows=args.sample_rows
        )
        result = mapper.map_fields(
            headers,
            sample_values=sample_values,
            backend=args.backend,
        )

        print(f"\nField mapping for: {args.source}")
        print(f"Sensor profile: {profile.name}")
        if args.backend == "ollama":
            print("Backend: ollama (Tier 1+2+3)")
        print()
        print(mapper.format_report(result))

        if not result.matches:
            print("\nNo matches found.")
            return

        # Interactive confirmation (skip with --yes)
        if not args.yes:
            confirmed_matches = _interactive_confirm(result, profile)
            if confirmed_matches is None:
                print("\nMapping aborted.")
                return
            result.matches = confirmed_matches

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
