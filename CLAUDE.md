# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IAQ-Forge — ML platform for air quality prediction from BME680 sensor data. Trains and serves MLP, KAN, LSTM, and CNN models that reproduce BSEC IAQ indices. Python 3.9+, FastAPI, PyTorch, InfluxDB.

## Commands

```bash
# Run dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Train models (CLI — uses dummy data via load_dataset())
python -m iaqforge train --model mlp --epochs 200
python -m iaqforge train --model all --epochs 50
python -m iaqforge list

# Train from real InfluxDB data (standalone scripts)
python train_models.py          # MLP + KAN
python train_all_models.py      # MLP + CNN + KAN

# Create untrained dummy models for dev/testing
python create_dummy_models.py

# Integration test (sends simulated sensor readings to running server)
python test_client.py

# No pytest suite exists yet
```

## Architecture

**Two training paths** — this is the most important thing to understand:
1. `python -m iaqforge` CLI → `iaqforge/__main__.py` → `iaqforge/model_trainer.py` → `training/train.py:train_single_model()` — uses **dummy data** via `training/utils.load_dataset()`
2. `train_models.py` / `train_all_models.py` (standalone scripts) → `training/utils.fetch_training_data()` — fetches **real data from InfluxDB**, trains with `training/utils.train_model()`

Both paths save artifacts to `trained_models/{model_type}/` (model.pt, config.json, optional scaler.pkl files).

**FastAPI service** (`app/main.py`): loads all 4 models at startup into global `predictors` and `inference_engines` dicts. Active model switchable via `/model/select`. Predictions optionally written to InfluxDB.

**Inference flow**: `SensorReading` → `InferenceEngine` → `IAQPredictor.predict()` → scaler transform → torch forward pass → clamp [0,500] → categorize → `IAQResponse`

**Config**: `model_config.yaml` is source of truth for model architecture. `database_config.yaml` for InfluxDB. Both loaded by `app/config.py:Settings` singleton (`settings`). YAML overrides hardcoded defaults.

## Key Technical Details

- **6 features**: 4 raw (temp, humidity, pressure, gas_resistance) + 2 engineered (gas_ratio, abs_humidity). Computed in `training/utils.prepare_features()`.
- **Sliding window**: LSTM/CNN need `window_size` readings buffered before first prediction. `IAQPredictor.buffer` manages this; returns status `buffering` until full.
- **Input dimensions**: MLP/KAN flatten to `window_size × num_features`; LSTM/CNN keep temporal shape `(batch, window_size, num_features)`.
- **Scaling**: StandardScaler for features, MinMaxScaler(0,1) for targets. Scalers saved as .pkl alongside models.
- **KAN**: `efficient-kan` vendored in `app/kan.py` — external package breaks on Python >3.9 on Apple Silicon.
- **InfluxDB**: Dual support for 1.x (`influxdb` client) and 2.x (`influxdb-client`). Disabled by default.
- **GPU**: Auto-detects MPS/CUDA/CPU via `training/utils.get_device()`.

## Conventions

- Config access: always use `from app.config import settings` — never pass DB/model params as function args
- Model types are lowercase strings: `mlp`, `kan`, `lstm`, `cnn`
- Model artifacts at `trained_models/{model_type}/`
- Absolute imports: `from app.models import IAQPredictor`
- All models inherit `torch.nn.Module`
