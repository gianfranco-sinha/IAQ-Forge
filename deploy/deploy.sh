#!/usr/bin/env bash
# Deploy iaq4j to pi@87.106.102.14
# Usage: bash deploy/deploy.sh
set -euo pipefail

SERVER="0.0.0.0"
REMOTE_DIR="/home/pi/iaq4j"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== iaq4j deployment to $SERVER ==="

# 1. Sync project files (exclude dev artifacts)
echo "[1/6] Syncing project files..."
rsync -avz --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'venv' \
  --exclude '.venv' \
  --exclude 'runs/' \
  --exclude '.idea' \
  --exclude '.env' \
  "$PROJECT_DIR/" "$SERVER:$REMOTE_DIR/"

# 2. Set up venv + install deps on server
echo "[2/6] Installing Python dependencies..."
ssh "$SERVER" bash -s <<'REMOTE'
set -euo pipefail
cd /home/pi/iaq4j

# Create venv if missing
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install joblib  # needed for scaler loading
REMOTE

# 3. Create InfluxDB database if it doesn't exist
echo "[3/6] Ensuring InfluxDB database exists..."
ssh "$SERVER" bash -s <<'REMOTE'
set -euo pipefail
# Create iaq_predictions database (idempotent)
influx -execute "CREATE DATABASE iaq_predictions" 2>/dev/null || true
echo "InfluxDB database 'iaq_predictions' ready"
REMOTE

# 4. Install systemd service
echo "[4/6] Installing systemd service..."
ssh "$SERVER" bash -s <<'REMOTE'
set -euo pipefail
sudo cp /home/pi/iaq4j/deploy/iaq4j.service /etc/systemd/system/iaq4j.service
sudo systemctl daemon-reload
sudo systemctl enable iaq4j
sudo systemctl restart iaq4j
sleep 2
sudo systemctl status iaq4j --no-pager || true
REMOTE

# 5. Install nginx config
echo "[5/6] Configuring nginx reverse proxy..."
ssh "$SERVER" bash -s <<'REMOTE'
set -euo pipefail
sudo cp /home/pi/iaq4j/deploy/nginx-iaq4j.conf /etc/nginx/sites-available/iaq4j
sudo ln -sf /etc/nginx/sites-available/iaq4j /etc/nginx/sites-enabled/iaq4j
sudo nginx -t
sudo systemctl reload nginx
REMOTE

# 6. Verify
echo "[6/6] Verifying deployment..."
sleep 3
ssh "$SERVER" "curl -s http://127.0.0.1:8001/ | python3 -m json.tool" || echo "Service may still be starting..."

echo ""
echo "=== Deployment complete ==="
echo "  Service:  http://enviro-sensors.uk/iaq/"
echo "  Docs:     http://enviro-sensors.uk/iaq/docs"
echo "  Health:   http://enviro-sensors.uk/iaq/health/detailed"
echo "  Predict:  POST http://enviro-sensors.uk/iaq/predict"
echo "  InfluxDB: predictions → iaq_predictions.iaq_predictions"
echo "  Logs:     ssh $SERVER 'journalctl -u iaq4j -f'"
