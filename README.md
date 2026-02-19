# iaq4j

An open-source, sensor-agnostic ML platform for indoor air quality prediction. Supports any indoor IAQ sensor and any IAQ standard through pluggable sensor profiles and standard definitions. Ships with BME680 + BSEC IAQ as the default.

## Quick Start

```bash
# Setup environment
./setup.sh

# Train models (synthetic data)
python -m iaq4j train --model all --epochs 50

# Train from InfluxDB
python -m iaq4j train --model mlp --data-source influxdb

# Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Test API
python test_client.py
```

## Features

- **Sensor Agnostic**: Pluggable `SensorProfile` abstraction — bring any indoor IAQ sensor
- **Standard Agnostic**: Pluggable `IAQStandard` abstraction — target any IAQ index scale
- **Multi-Model Support**: MLP, KAN, LSTM, CNN architectures
- **CLI Training**: `python -m iaq4j train` with synthetic or InfluxDB data sources
- **FastAPI Service**: RESTful API with auto-docs, model switching, prediction comparison
- **Real-time Inference**: Sliding window buffering for temporal models
- **YAML Configuration**: Model architecture, sensor profile, and IAQ standard in `model_config.yaml`

## Architecture

### Sensor & Standard Abstractions

iaq4j decouples sensor hardware from ML models through two abstractions in `app/profiles.py`:

- **`SensorProfile`** — defines raw features, valid ranges, feature engineering, and baseline computation. Each sensor type implements its own `engineer_features()` method.
- **`IAQStandard`** — defines the target scale, clamping range, and category breakpoints.

Built-in implementations (`app/builtin_profiles.py`):
| Profile | Sensor | Raw Features | Engineered Features |
|---------|--------|--------------|---------------------|
| `BME680Profile` | BME680 | temperature, rel_humidity, pressure, voc_resistance | voc_ratio, abs_humidity |

| Standard | Scale | Categories |
|----------|-------|------------|
| `BSECStandard` | 0-500 | Excellent, Good, Moderate, Poor, Very Poor |

Select via `model_config.yaml`:
```yaml
sensor:
  type: bme680
iaq_standard:
  type: bsec
```

### Models

| Model | Type | Best For | Features |
|-------|------|----------|----------|
| MLP | Baseline | Quick predictions, low resources | Dense layers, batch norm |
| KAN | Advanced | Non-linear patterns | Kolmogorov-Arnold networks |
| LSTM | Temporal | Sequential data | Bidirectional, sliding window |
| CNN | Spatiotemporal | Local patterns | Conv1d, adaptive pooling |

## Technology Stack

- **Backend**: Python 3.9+, FastAPI, PyTorch
- **ML Models**: MLP, KAN, LSTM, CNN
- **Data**: InfluxDB integration (1.x and 2.x)
- **Configuration**: YAML-based (`model_config.yaml`, `database_config.yaml`)

## API Endpoints

- `GET /health` - Service status
- `GET /models` - Available models
- `POST /predict` - Single prediction
- `POST /predict/compare` - Multi-model comparison
- `POST /model/select` - Switch active model
- `GET /statistics` - Prediction statistics

Visit `/docs` for interactive API documentation.

## Configuration

### Model & Sensor Configuration (`model_config.yaml`)
- Sensor profile and IAQ standard selection
- Neural network architecture parameters
- Training hyperparameters
- Global and model-specific settings

### Database Configuration (`database_config.yaml`)
- InfluxDB connection parameters
- Authentication credentials
- Operational settings and logging options
