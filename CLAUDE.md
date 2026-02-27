# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

iaq4j — ML platform for indoor air quality prediction. **Scope: any indoor air quality sensor, any indoor IAQ standard, ML-driven prediction.** Trains and serves MLP, KAN, LSTM, CNN, and BNN models. Python 3.9.x (strict — KAN incompatible with 3.10+), FastAPI, PyTorch, InfluxDB. Default sensor: BME680. Default standard: BSEC IAQ.

## Commands

```bash
# Run dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Train models (CLI — uses synthetic data by default)
python -m iaq4j train --model mlp --epochs 200
python -m iaq4j train --model all --epochs 50
python -m iaq4j train --model mlp --data-source influxdb  # real data
python -m iaq4j train --model mlp --data-source csv --csv-path data.csv  # CSV file
python -m iaq4j list
python -m iaq4j version                             # show active semver + metrics
python -m iaq4j verify [--model mlp]                # verify Merkle tree provenance

# Train from real InfluxDB data (standalone scripts)
python train_models.py          # MLP + KAN
python train_all_models.py      # MLP + CNN + KAN

# Create untrained dummy models for dev/testing
python training/create_dummy_models.py

# Semantic field mapping (map CSV columns to internal feature names)
python -m iaq4j map-fields --source data.csv [--backend fuzzy|ollama] [--save] [-y]

# Integration test (sends simulated sensor readings to running server)
python test_client.py

# TensorBoard (enabled by default during training)
tensorboard --logdir runs/

# MLflow experiment tracking UI
mlflow ui --port 5000

# Linting / type checking
ruff check app/ training/
mypy app/
black app/ training/

# No pytest suite exists yet
```

## Architecture

**Two training paths** — this is the most important thing to understand:
1. `python -m iaq4j` CLI → `iaq4j/__main__.py` → `iaq4j/model_trainer.py` → `training/train.py:train_single_model()` → `training/pipeline.py:TrainingPipeline` — uses `SyntheticSource` by default or `InfluxDBSource`/`CSVDataSource`/`LabelStudioDataSource` via `--data-source`
2. `train_models.py` / `train_all_models.py` (standalone scripts) → `training/utils.train_model()` — fetches **real data from InfluxDB**

Both paths save artifacts to `trained_models/{model_type}/` (model.pt, config.json, scaler .pkl files, MANIFEST.json).

**Training pipeline** (`training/pipeline.py`): FSM with stages: SOURCE_ACCESS → INGESTION → FEATURE_ENGINEERING → WINDOWING → SPLITTING → SCALING → TRAINING → EVALUATION → SAVING. Each stage emits a `StageResult`. Data cleaning issues tracked in `PreprocessingReport`. Chronological train/val split (no shuffling) to prevent temporal data leakage. Data provenance captured per run: SHA256 data fingerprint, git commit, feature statistics, config snapshot.

**Data sources** (`training/data_sources.py`): `DataSource` ABC with four implementations: `SyntheticSource` (default — generates physically plausible random readings), `InfluxDBSource` (reads from `iaq_readings` measurement), `CSVDataSource` (flat CSV files), `LabelStudioDataSource` (exports annotated projects from Label Studio — Stage 1 only: config validation and connection, full pipeline ingestion pending).

**FastAPI service** (`app/main.py`): loads all 5 model types at startup into global `predictors` and `inference_engines` dicts. Active model switchable via `/model/select`. Predictions optionally written to InfluxDB. OpenAPI schema downgraded from 3.1 to 3.0.3 for Swagger UI compatibility.

**Inference engine** (`app/inference.py`): `InferenceEngine` wraps `IAQPredictor`. Flow: `SensorReading` → `InferenceEngine` → `IAQPredictor.predict()` → profile feature engineering → scaler transform → torch forward pass → standard clamp/categorize → `IAQResponse`. Uncertainty via MC dropout (MLP/LSTM/CNN), weight sampling (BNN), or history-based fallback (KAN). Supports Bayesian conjugate updates via `prior_variables` (external environmental signals that shift the posterior) and per-engine prediction history for running statistics and sensor drift analysis.

**Sensor & standard abstractions** (`app/profiles.py`): `SensorProfile` ABC defines raw features, valid ranges, and feature engineering. `IAQStandard` ABC defines target scale and category breakpoints. Built-in implementations in `app/builtin_profiles.py` (BME680 + BSEC). Selected via `sensor.type` and `iaq_standard.type` in `model_config.yaml`. Registries populated via `import app.builtin_profiles` side effect.

**Physical quantity registry** (`quantities.yaml` + `app/quantities.py`): Central YAML table of all known physical quantities (temperature, humidity, VOC resistance, CO2, etc.) with canonical units, valid ranges, and alternate unit conversion expressions. `app/quantities.py` loads it lazily, exposes `get_quantity()`, `convert_to_canonical()`, and `list_quantities()`. Unit conversion expressions use a safe AST-based evaluator (no `eval()`). `SensorProfile.feature_quantities` maps profile feature names to quantity names in this table.

**Config**: `model_config.yaml` is source of truth for model architecture, sensor profile, IAQ standard, and training hyperparameters. `database_config.yaml` for InfluxDB. Both loaded by `app/config.py:Settings` singleton (`settings`). YAML overrides hardcoded defaults.

## Key Technical Details

- **Features are profile-driven**: `SensorProfile.raw_features` + `SensorProfile.engineered_feature_names` determine `num_features`. BME680 default: 4 raw + 2 engineered = 6.
- **Feature engineering is code**: each `SensorProfile` subclass owns its `engineer_features()` method. No config DSL.
- **Sliding window**: all models buffer `window_size` readings before first prediction. `IAQPredictor.buffer` manages this.
- **Input dimensions**: MLP/KAN/BNN flatten to `window_size × num_features`; LSTM/CNN keep temporal shape `(batch, window_size, num_features)`.
- **Scaling**: StandardScaler for features, MinMaxScaler(0,1) for targets. Scalers saved as .pkl alongside models.
- **KAN**: `efficient-kan` vendored in `app/kan.py` to remove the external dependency (which broke on Python >3.9 on Apple Silicon).
- **BNN**: Bayesian Neural Network with variational weight layers. Produces `kl_loss` for ELBO training. Config: `prior_sigma`, `kl_weight`.
- **InfluxDB**: Dual support for 1.x (`influxdb` client) and 2.x (`influxdb-client`). Disabled by default.
- **GPU**: Auto-detects MPS/CUDA/CPU via `training/utils.get_device()`.
- **`SensorReading`**: accepts both `readings: Dict[str, float]` (generic) and legacy BME680 keyword fields. `_build_readings()` model_validator merges legacy fields into `readings` and applies any `sensor.field_mapping` from config.
- **`iaq_actual`**: optional ground-truth field on `SensorReading`, persisted to InfluxDB alongside predictions for evaluation.
- **Model artifacts**: `model.pt` (weights), `config.json` (architecture + sensor_type + iaq_standard + baselines + semver + schema_fingerprint), `feature_scaler.pkl`, `target_scaler.pkl`, `MANIFEST.json` (data lineage with version, fingerprint, metrics), `data_manifest.json` (Merkle tree provenance).
- **Artifact semver**: format `{model_type}-{MAJOR}.{MINOR}.{PATCH}`. MAJOR on schema change, MINOR on retrain/metrics change, PATCH on metadata-only. `schema_fingerprint` is SHA256[:12] of (sensor_type, iaq_standard, window_size, num_features, model_type).
- **Merkle tree**: 6-level chain (Sensor → RawData → CleansedData → PreprocessedData → SplitData → TrainedModel) in `training/merkle.py`. Root hash stored in `config.json`, `MANIFEST.json`, `data_manifest.json`. Verify with `python -m iaq4j verify`.

## Conventions

- Import order: stdlib → third-party → local (match existing file style)
- Config access: always use `from app.config import settings` — never pass DB/model params as function args
- Model types are lowercase strings: `mlp`, `kan`, `lstm`, `cnn`, `bnn`
- Model artifacts at `trained_models/{model_type}/`
- Absolute imports: `from app.models import IAQPredictor`
- All models inherit `torch.nn.Module`
- Profile registration: `import app.builtin_profiles  # noqa: F401` — required wherever profiles are needed (main.py, pipeline.py)
- Backward compat: old model artifacts with `baseline_gas_resistance` key still load; `SensorReading` accepts both old 4-field and new `readings` format

## Deployment

**Production server**: `pi@87.106.102.14` → `/home/pi/iaq4j/`

```bash
# Full deploy (rsync + deps + systemd restart + nginx + verify)
bash deploy/deploy.sh

# Docker (alternative)
docker compose up -d

# View production logs
ssh pi@87.106.102.14 'journalctl -u iaq4j -f'
```

- **Systemd service** (`deploy/iaq4j.service`): runs uvicorn on `127.0.0.1:8001` as user `pi`
- **Nginx** (`deploy/nginx-iaq4j.conf`): reverse proxies `/iaq4j/` → `localhost:8001`
- **Docker** (`docker-compose.yml`): CPU-only PyTorch, port 8001, mounts `trained_models/` and config YAMLs, joins `iotstack_default` network
- **`ROOT_PATH`** env var: set to `/iaq4j` in production for nginx subpath routing; defaults to `""` in dev. Must match nginx `location` path.
- **InfluxDB writes**: `influx_manager.write_prediction()` writes to `iaq_predictions` measurement, tagged by model type. Global singleton in `app/database.py`.

## Roadmap

See `docs/roadmap.md` for pending features. Current status:
- ~~Physical quantity registry (P0)~~ — **DONE** (`quantities.yaml` + `app/quantities.py`)
- ~~Semantic field mapping (P1)~~ — **DONE** (CLI `map-fields` + REST `/sensors/register`)
- ~~Artifact semver (P1)~~ — **DONE**
- ~~Sensor Registration API (P1)~~ — **DONE**
- **MLflow Integration (P1)** — Basic integration live in `training/train.py`: experiment set, run started per training job, 3 params + 4 metrics + model artifact logged, `FAILED` status on exception. Remaining: log full `collect_run_params()` dict, wire `on_epoch` callback to `mlflow.log_metrics(..., step=epoch)`, log scaler artifacts + `data_manifest.json`, set tags (version, schema_fingerprint, merkle_root), update standalone scripts. Open: deployment mode (local `mlruns/` vs tracking server), layer vs replace MANIFEST.json, model registry.
- **LLM Readiness (P1)** — Phase 1: config cache invalidation, InfluxDB reads, training exception propagation, standardized error model. Phase 2: REST endpoints for profiles, quantities, models, history, config, training trigger, audit. Phase 3 = LLM Agent.
- **LLM Agent (P2)** — depends on LLM Readiness Phase 1+2. Tool registry maps 1:1 to Phase 2 endpoints.
- **LabelStudio data source** — Stage 1 done (config + validate + CLI flag). Stage 2: full pipeline ingestion (INGESTION → FEATURE_ENGINEERING stages).
