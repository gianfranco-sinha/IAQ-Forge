# IAQ-Forge

Commercial ML platform for BSEC IAQ index reproduction from raw BME680 sensor data.

## Quick Start

```bash
# Setup environment
./setup.sh

# Train models
./venv/bin/python -m airml train --model all --epochs 50

# Start API server  
./venv/bin/python -m app.main --reload

# Test API
./venv/bin/python test_client.py
```

## Features

- **Multi-Model Support**: MLP, KAN, LSTM, CNN architectures
- **YAML Configuration**: Flexible model parameter management
- **CLI Training**: Batch and individual model training
- **FastAPI Service**: RESTful API with auto-docs
- **Real-time Inference**: Sliding window for temporal models
- **Enterprise Ready**: Structured logging, error handling, monitoring

## Technology Stack

- **Backend**: Python 3.9+, FastAPI, PyTorch
- **ML Models**: Multi-layer Perceptron, KAN, LSTM, CNN  
- **Data**: InfluxDB integration (supports 1.x and 2.x)
- **Configuration**: YAML-based model and database configuration
- **Deployment**: Docker-ready, production configuration

## Configuration

AirML uses YAML configuration files for flexible setup:

### Model Configuration (`model_config.yaml`)
- Neural network architecture parameters
- Training hyperparameters  
- Global and model-specific settings

### Database Configuration (`database_config.yaml`)
- InfluxDB connection parameters
- Authentication credentials
- Operational settings and logging options

## Models

| Model | Type | Best For | Features |
|--------|------|-----------|----------|
| MLP | Baseline | Quick predictions, low resources | Dense layers, batch norm |
| KAN | Advanced | Non-linear patterns | Kolmogorov-Arnold networks |
| LSTM | Temporal | Sequential data | Bidirectional, sliding window |
| CNN | Spatiotemporal | Local patterns | Conv1d, adaptive pooling |

## API Endpoints

- `GET /health` - Service status
- `GET /models` - Available models  
- `POST /predict` - Single prediction
- `POST /predict/compare` - Multi-model comparison
- `GET /model/select/{model_type}` - Switch active model

Visit `/docs` for interactive API documentation.
