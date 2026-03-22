"""KAN Sidecar — lightweight FastAPI server for KAN inference.

Endpoints:
    GET  /health  — liveness + model status
    POST /load    — (re)load model from trained_models/kan/
    POST /forward — raw tensor forward pass
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.models import KANRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="iaq4j KAN Sidecar", version="1.0.0")

# Global model state
_model: Optional[KANRegressor] = None
_model_dir = Path("trained_models/kan")


class ForwardRequest(BaseModel):
    tensor: List[List[float]]
    shape: List[int]


class ForwardResponse(BaseModel):
    prediction: List[List[float]]


def _load_model() -> KANRegressor:
    """Load KAN model from disk."""
    config_path = _model_dir / "config.json"
    weights_path = _model_dir / "model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    with open(config_path) as f:
        config = json.load(f)

    model_params = config.get("model_params", {})
    model_data = torch.load(weights_path, map_location="cpu")

    if isinstance(model_data, dict):
        if "input_dim" in model_data:
            model_params["input_dim"] = model_data["input_dim"]
        if "hidden_dims" in model_data:
            model_params["hidden_dims"] = model_data["hidden_dims"]

    if "input_dim" not in model_params:
        window_size = config.get("window_size", 10)
        num_features = config.get("num_features", 15)
        model_params["input_dim"] = window_size * num_features

    model = KANRegressor(**model_params)

    if isinstance(model_data, dict) and "state_dict" in model_data:
        model.load_state_dict(model_data["state_dict"])
    else:
        model.load_state_dict(model_data)

    model.eval()
    logger.info("KAN model loaded from %s", _model_dir)
    return model


@app.on_event("startup")
async def startup():
    """Try to load model on startup."""
    global _model
    try:
        _model = _load_model()
    except Exception as e:
        logger.warning("Model not loaded on startup: %s", e)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/load")
async def load():
    """(Re)load the KAN model from disk."""
    global _model
    try:
        _model = _load_model()
        return {"status": "loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@app.post("/forward", response_model=ForwardResponse)
async def forward(req: ForwardRequest):
    """Run a forward pass on the loaded KAN model."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call POST /load first.")

    try:
        tensor = torch.tensor(req.tensor, dtype=torch.float32)
        with torch.no_grad():
            output = _model(tensor)
        return ForwardResponse(prediction=output.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forward pass failed: {e}")
