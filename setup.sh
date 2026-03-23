#!/bin/bash
# setup.sh — Set up iaq4j development environment
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-venv}

echo "Setting up iaq4j environment..."
echo ""

# ── Create virtual environment if needed ─────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
    echo "  Created."
else
    echo "Virtual environment already exists at $VENV_DIR."
fi

# ── Activate ─────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
echo "Activated: $(python --version) at $(which python)"

# ── Install dependencies ─────────────────────────────────────────────
echo ""
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo "  Done."

# ── Create dummy models for dev (if no trained models exist) ─────────
if [ ! -f "trained_models/mlp/model.pt" ]; then
    echo ""
    echo "No trained models found — creating dummy models for dev/testing..."
    python training/create_dummy_models.py
    echo "  Done."
else
    echo ""
    echo "Trained models found — skipping dummy model creation."
fi

# ── Show available commands ──────────────────────────────────────────
echo ""
echo "Setup complete. Available commands:"
echo ""
echo "  Training:"
echo "    python -m iaq4j train --model mlp --epochs 200"
echo "    python -m iaq4j train --model all --epochs 50"
echo "    python -m iaq4j train --model mlp --early-stopping 15"
echo "    python -m iaq4j train --model mlp --data-source influxdb"
echo "    python -m iaq4j train --model mlp --resume"
echo ""
echo "  Model management:"
echo "    python -m iaq4j list"
echo "    python -m iaq4j version"
echo "    python -m iaq4j verify"
echo ""
echo "  Dev server:"
echo "    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "  Tests:"
echo "    python -m pytest tests/unit/"
echo "    python -m pytest --cov=app --cov=training tests/"
