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
        "--window-size",
        type=int,
        default=None,
        help="Sliding window size (default: reads per-model value from model_config.yaml)",
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
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    train_parser.add_argument(
        "--unit-overrides",
        type=str,
        default=None,
        help='JSON mapping of feature names to source units, e.g. \'{"voc_resistance": "kΩ"}\'',
    )
    train_parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable caching of InfluxDB data for faster subsequent runs",
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

    # Drift analysis command
    drift_parser = subparsers.add_parser(
        "drift-analysis",
        help="Analyze sensor drift and seasonal variation from historical data",
    )
    drift_parser.add_argument(
        "--hours-back",
        type=int,
        default=8760,
        help="Hours of historical data to fetch (default: 8760 = 1 year)",
    )
    drift_parser.add_argument(
        "--database",
        type=str,
        help="InfluxDB database name (default: from config)",
    )
    drift_parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for CSV/JSON output (optional)",
    )
    drift_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PNG drift charts",
    )
    drift_parser.add_argument(
        "--gap-threshold",
        type=float,
        default=1.0,
        help="Gap detection threshold in hours (default: 1.0)",
    )

    # Drift correction evaluation command
    drift_eval_parser = subparsers.add_parser(
        "drift-correction-eval",
        help="Evaluate drift correction impact on model accuracy",
    )
    drift_eval_parser.add_argument(
        "--data-source",
        choices=["synthetic", "influxdb"],
        default="synthetic",
        help="Data source for evaluation (default: synthetic)",
    )
    drift_eval_parser.add_argument(
        "--database",
        type=str,
        help="InfluxDB database name (default: from config)",
    )
    drift_eval_parser.add_argument(
        "--drift-summary",
        type=str,
        help="Path to drift_summary.json (default: results/drift_3yr/drift_summary.json)",
    )
    drift_eval_parser.add_argument(
        "--output",
        type=str,
        help="Output JSON path (default: results/drift_correction_eval.json)",
    )
    drift_eval_parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of synthetic samples (default: 2000)",
    )
    drift_eval_parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable caching of InfluxDB data for faster subsequent runs",
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

    # Cache command
    cache_parser = subparsers.add_parser("cache", help="Manage cached InfluxDB data")
    cache_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cached data",
    )
    cache_parser.add_argument(
        "--list",
        action="store_true",
        help="List cached entries with metadata",
    )
    cache_parser.add_argument(
        "--dir",
        type=str,
        default="cache",
        help="Cache directory (default: cache)",
    )

    # Export to Label Studio command
    export_ls_parser = subparsers.add_parser(
        "export-to-ls",
        help="Export InfluxDB data to Label Studio with before/after cleanse views",
    )
    export_ls_parser.add_argument(
        "--project-id",
        type=int,
        help="Label Studio project ID (falls back to integrations.yaml)",
    )
    export_ls_parser.add_argument(
        "--hours-back",
        type=int,
        default=1344,
        help="Hours of InfluxDB data to fetch (default: 1344 = 8 weeks)",
    )
    export_ls_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Tasks per Label Studio API call (default: 50)",
    )
    export_ls_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show task stats without posting to Label Studio",
    )
    export_ls_parser.add_argument(
        "--include-dropped",
        action="store_true",
        help="Include rows the pipeline dropped during cleansing",
    )
    export_ls_parser.add_argument(
        "--database",
        type=str,
        help="InfluxDB database name (default: from config)",
    )
    export_ls_parser.add_argument(
        "--csv",
        type=str,
        metavar="PATH",
        help="Export to CSV file instead of Label Studio (e.g. --csv export.csv)",
    )

    # Label Studio command
    ls_parser = subparsers.add_parser(
        "label-studio",
        help="Start Label Studio with iaq4j integration",
    )
    ls_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Label Studio port (default: 8080)",
    )
    ls_parser.add_argument(
        "--csv-port",
        type=int,
        default=9999,
        help="CSV file server port (default: 9999)",
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

            # Parse unit overrides
            unit_overrides = None
            if args.unit_overrides:
                import json as _json

                try:
                    unit_overrides = _json.loads(args.unit_overrides)
                except _json.JSONDecodeError as e:
                    print(f"Invalid --unit-overrides JSON: {e}")
                    sys.exit(1)

            # Build data source
            data_source = None
            if args.data_source == "influxdb":
                from training.data_sources import InfluxDBSource

                data_source = InfluxDBSource(
                    database=args.database,
                    max_records=args.max_records,
                    hours_back=args.hours_back,
                    cache=args.cache,
                    unit_overrides=unit_overrides,
                )
            elif args.data_source == "csv":
                if not args.csv_path:
                    print("❌ --csv-path is required when --data-source is csv")
                    sys.exit(1)
                from training.data_sources import CSVDataSource

                data_source = CSVDataSource(
                    args.csv_path,
                    unit_overrides=unit_overrides,
                )
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
                    resume=args.resume,
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

    elif args.command == "drift-analysis":
        import app.builtin_profiles  # noqa: F401
        from app.profiles import get_iaq_standard, get_sensor_profile
        from training.data_sources import InfluxDBSource
        from training.drift_analysis import (
            compute_moving_averages,
            format_console_report,
            plot_drift_charts,
            save_drift_output,
        )

        profile = get_sensor_profile()
        standard = get_iaq_standard()

        # Features to analyze: raw sensor features + IAQ target
        features = list(profile.raw_features) + [standard.target_column]

        source = InfluxDBSource(
            hours_back=args.hours_back,
            database=args.database,
        )

        print(f"Connecting to {source.name}...")
        try:
            source.validate()
        except Exception as e:
            print(f"Failed to connect to InfluxDB: {e}")
            sys.exit(1)

        print(f"Fetching {args.hours_back}h of data...")
        try:
            df = source.fetch()
        except Exception as e:
            print(f"Failed to fetch data: {e}")
            sys.exit(1)
        finally:
            source.close()

        print(f"Analyzing {len(df):,} samples...")
        report = compute_moving_averages(
            df, features, gap_threshold_hours=args.gap_threshold
        )
        print(format_console_report(report))

        if args.output_dir:
            save_drift_output(report, args.output_dir)
            print(f"Output saved to {args.output_dir}/")

        if args.plot:
            output_dir = args.output_dir or "drift_output"
            plot_drift_charts(report, output_dir)
            print(f"Charts saved to {output_dir}/")

    elif args.command == "drift-correction-eval":
        from training.drift_correction_eval import run_evaluation

        run_evaluation(
            data_source=args.data_source,
            drift_summary_path=args.drift_summary,
            output_path=args.output,
            num_samples=args.num_samples,
            database=args.database,
            cache=args.cache,
        )

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

    elif args.command == "export-to-ls":
        import app.builtin_profiles  # noqa: F401
        from app.profiles import get_iaq_standard, get_sensor_profile
        from integrations.label_studio.exporter import cleanse_dataframe
        from integrations.label_studio.launcher import (
            generate_timeseries_csvs,
            setup_project,
            import_tasks,
            clear_project,
            _start_csv_server,
            CSV_PORT,
        )
        from training.data_sources import InfluxDBSource

        profile = get_sensor_profile()
        standard = get_iaq_standard()

        # Fetch raw data from InfluxDB
        source = InfluxDBSource(
            hours_back=args.hours_back,
            database=args.database,
        )

        print(f"Connecting to {source.name}...")
        try:
            source.validate()
        except Exception as e:
            print(f"Failed to connect to InfluxDB: {e}")
            sys.exit(1)

        print(f"Fetching {args.hours_back}h of data...")
        try:
            raw_df = source.fetch()
        except Exception as e:
            print(f"Failed to fetch data: {e}")
            sys.exit(1)
        finally:
            source.close()

        print(f"Fetched {len(raw_df):,} raw rows")

        # Get gap threshold from config
        from integrations.config import get_integration_config as _get_int_cfg
        _ls_cfg = _get_int_cfg("label_studio")
        gap_threshold = _ls_cfg.get("gap_threshold_seconds", 30.0)

        # Cleanse
        clean_df, drop_reasons, ingest_report = cleanse_dataframe(
            raw_df, profile, standard.target_column,
            gap_threshold_seconds=gap_threshold,
        )
        dropped = ingest_report["dropped_rows"]
        print(f"After cleansing: {len(clean_df):,} kept, {dropped:,} dropped")
        print(f"Discontinuities: {ingest_report['discontinuities']} "
              f"(gaps > {gap_threshold}s)")
        if ingest_report["gaps"]:
            for g in ingest_report["gaps"][:10]:
                print(f"  gap {g['gap_seconds']:.0f}s: {g['before']} -> {g['after']}")
            if len(ingest_report["gaps"]) > 10:
                print(f"  ... and {len(ingest_report['gaps']) - 10} more")

        if args.csv:
            # Export to flat CSV
            import pandas as _pd

            rows = []
            target = standard.target_column
            raw_features = list(profile.raw_features)
            data_columns = raw_features + ([target] if target in raw_df.columns else [])
            kept_indices = clean_df.index

            for idx in kept_indices:
                row = {"cleanse_status": "kept"}
                if isinstance(idx, _pd.Timestamp):
                    row["timestamp"] = idx.isoformat()
                for col in data_columns:
                    row[f"raw_{col}"] = raw_df.loc[idx].get(col)
                    row[f"clean_{col}"] = clean_df.loc[idx].get(col)
                rows.append(row)

            if args.include_dropped:
                dropped_indices = raw_df.index.difference(kept_indices)
                for idx in dropped_indices:
                    row = {"cleanse_status": "dropped"}
                    if isinstance(idx, _pd.Timestamp):
                        row["timestamp"] = idx.isoformat()
                    for col in data_columns:
                        row[f"raw_{col}"] = raw_df.loc[idx].get(col)
                    if drop_reasons is not None and idx in drop_reasons.index:
                        row["cleanse_reason"] = str(drop_reasons.loc[idx])
                    rows.append(row)

            out_df = _pd.DataFrame(rows)
            out_df.to_csv(args.csv, index=False)
            print(f"Exported {len(out_df):,} rows to {args.csv}")
        else:
            # Generate time series CSVs split at discontinuities
            out_dir, tasks = generate_timeseries_csvs(
                raw_df, clean_df, profile, standard,
                drop_reasons=drop_reasons,
                include_dropped=args.include_dropped,
                gap_threshold_seconds=gap_threshold,
            )
            print(f"Generated {len(tasks)} segment CSVs in {out_dir}/")

            if args.dry_run:
                print(f"\n[DRY RUN] Would import {len(tasks)} tasks into Label Studio")
                return

            # Get API key
            import os
            from integrations.config import get_integration_config

            int_cfg = get_integration_config("label_studio")
            api_key = os.environ.get("LABEL_STUDIO_API_KEY") or int_cfg.get("api_key")
            ls_url = int_cfg.get("url", "http://localhost:8080")

            if not api_key:
                print("Set LABEL_STUDIO_API_KEY env var or api_key in integrations.yaml")
                sys.exit(1)

            # Start CSV server so Label Studio can fetch the files
            csv_server = _start_csv_server(out_dir)
            print(f"CSV server started on http://localhost:{CSV_PORT}/")

            try:
                # Setup project with correct labeling config
                project_id = args.project_id
                if project_id:
                    # Clear existing tasks
                    print(f"Clearing existing tasks from project {project_id}...")
                    clear_project(ls_url, api_key, project_id)
                else:
                    project_id = setup_project(ls_url, api_key)

                print(f"Importing {len(tasks)} tasks into project {project_id}...")
                import_tasks(ls_url, api_key, project_id, tasks,
                             batch_size=args.batch_size)
                print(f"\nDone! Open http://localhost:8080/projects/{project_id}")
                print("Keep this terminal open (CSV server must stay running).")
                print("Press Ctrl+C to stop.")

                # Keep CSV server alive
                import signal
                signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                csv_server.shutdown()

    elif args.command == "label-studio":
        from integrations.label_studio.launcher import launch
        launch(ls_port=args.port, csv_port=args.csv_port)

    elif args.command == "cache":
        cache_dir = Path(args.dir)
        if args.clear:
            if cache_dir.exists():
                import shutil

                shutil.rmtree(cache_dir)
                print(f"Cache cleared: {cache_dir}")
            else:
                print(f"Cache directory does not exist: {cache_dir}")
        elif args.list:
            if not cache_dir.exists():
                print(f"No cache entries found: {cache_dir}")
                return
            print(f"Cache directory: {cache_dir}")
            print("-" * 50)
            for f in sorted(cache_dir.glob("*.parquet")):
                stat = f.stat()
                size_kb = stat.st_size / 1024
                mtime = pd.Timestamp(stat.st_mtime, unit="s")
                print(f"  {f.name}  {size_kb:.1f} KB  {mtime}")
            print("-" * 50)
            print(f"Total: {len(list(cache_dir.glob('*.parquet')))} entries")
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
