# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

iaq4j — ML platform for indoor air quality prediction. **Scope: any indoor air quality sensor, any indoor IAQ standard, ML-driven prediction.** Trains and serves MLP, KAN, LSTM, CNN, and BNN models. Python 3.9+, FastAPI, PyTorch, InfluxDB. Default sensor: BME680. Default standard: BSEC IAQ.

## Commands

```bash
# Run dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Train models (CLI — uses synthetic data by default)
python -m iaq4j train --model mlp --epochs 200
python -m iaq4j train --model all --epochs 50
python -m iaq4j train --model mlp --data-source influxdb  # real data
python -m iaq4j list

# Train from real InfluxDB data (standalone scripts)
python train_models.py          # MLP + KAN
python train_all_models.py      # MLP + CNN + KAN

# Create untrained dummy models for dev/testing
python training/create_dummy_models.py

# Integration test (sends simulated sensor readings to running server)
python test_client.py

# TensorBoard (enabled by default during training)
tensorboard --logdir runs/

# No pytest suite exists yet
```

## Architecture

**Two training paths** — this is the most important thing to understand:
1. `python -m iaq4j` CLI → `iaq4j/__main__.py` → `iaq4j/model_trainer.py` → `training/train.py:train_single_model()` → `training/pipeline.py:TrainingPipeline` — uses `SyntheticSource` by default or `InfluxDBSource` via `--data-source influxdb`
2. `train_models.py` / `train_all_models.py` (standalone scripts) → `training/utils.train_model()` — fetches **real data from InfluxDB**

Both paths save artifacts to `trained_models/{model_type}/` (model.pt, config.json, scaler .pkl files, MANIFEST.json).

**Training pipeline** (`training/pipeline.py`): FSM with stages: SOURCE_ACCESS → INGESTION → FEATURE_ENGINEERING → WINDOWING → SPLITTING → SCALING → TRAINING → EVALUATION → SAVING. Each stage emits a `StageResult`. Data cleaning issues tracked in `PreprocessingReport`. Chronological train/val split (no shuffling) to prevent temporal data leakage. Data provenance captured per run: SHA256 data fingerprint, git commit, feature statistics, config snapshot.

**FastAPI service** (`app/main.py`): loads all 5 model types at startup into global `predictors` and `inference_engines` dicts. Active model switchable via `/model/select`. Predictions optionally written to InfluxDB. OpenAPI schema downgraded from 3.1 to 3.0.3 for Swagger UI compatibility.

**Inference flow**: `SensorReading` → `InferenceEngine` → `IAQPredictor.predict()` → profile feature engineering → scaler transform → torch forward pass → standard clamp/categorize → `IAQResponse`. Uncertainty via MC dropout (MLP/LSTM/CNN), weight sampling (BNN), or history-based fallback (KAN).

**Sensor & standard abstractions** (`app/profiles.py`): `SensorProfile` ABC defines raw features, valid ranges, and feature engineering. `IAQStandard` ABC defines target scale and category breakpoints. Built-in implementations in `app/builtin_profiles.py` (BME680 + BSEC). Selected via `sensor.type` and `iaq_standard.type` in `model_config.yaml`. Registries populated via `import app.builtin_profiles` side effect.

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
- **`SensorReading`**: accepts both `readings: Dict[str, float]` (generic) and legacy BME680 keyword fields. `_build_readings()` model_validator merges legacy fields into `readings`.
- **`iaq_actual`**: optional ground-truth field on `SensorReading`, persisted to InfluxDB alongside predictions for evaluation.
- **Model artifacts**: `model.pt` (weights), `config.json` (architecture + sensor_type + iaq_standard + baselines), `feature_scaler.pkl`, `target_scaler.pkl`, `MANIFEST.json` (data lineage with version, fingerprint, metrics).

## Conventions

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

See `docs/roadmap.md` for pending features: physical quantity registry (P0), semantic field mapping with LLM (P1), artifact semver (P1), sensor registration API (P1).
