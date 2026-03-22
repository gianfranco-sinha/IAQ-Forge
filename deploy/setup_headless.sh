#!/usr/bin/env bash
# setup_headless.sh — Configure Mac Mini as a headless ML training server
#
# Usage: bash deploy/setup_headless.sh
#
# Run on the Mac Mini itself. Requires sudo for pmset/mdutil.
# Safe to re-run (idempotent).

set -euo pipefail

echo "=== Mac Mini Headless Training Server Setup ==="
echo ""

# ── 1. Prevent sleep ──────────────────────────────────────────────────
echo "[1/6] Disabling sleep and display sleep..."
sudo pmset -a sleep 0
sudo pmset -a disablesleep 1
sudo pmset -a displaysleep 0
sudo pmset -a hibernatemode 0
sudo pmset -a standby 0
sudo pmset -a autopoweroff 0
echo "  Done. Verify with: pmset -g"

# ── 2. Disable App Nap ───────────────────────────────────────────────
echo "[2/6] Disabling App Nap..."
defaults write NSGlobalDomain NSAppNapEnabled -bool NO
echo "  Done."

# ── 3. Disable Spotlight indexing ────────────────────────────────────
echo "[3/6] Disabling Spotlight indexing..."
sudo mdutil -a -i off
echo "  Done."

# ── 4. Enable SSH (Remote Login) ────────────────────────────────────
echo "[4/6] Enabling Remote Login (SSH)..."
if sudo systemsetup -getremotelogin | grep -q "On"; then
    echo "  Already enabled."
else
    sudo systemsetup -setremotelogin on
    echo "  Enabled. Connect with: ssh $(whoami)@$(hostname)"
fi

# ── 5. PyTorch MPS environment ──────────────────────────────────────
echo "[5/6] Setting PyTorch MPS environment variables..."
SHELL_RC="$HOME/.zshrc"
if ! grep -q "PYTORCH_MPS_HIGH_WATERMARK_RATIO" "$SHELL_RC" 2>/dev/null; then
    cat >> "$SHELL_RC" << 'ENVEOF'

# PyTorch MPS — let it use all available GPU memory
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
ENVEOF
    echo "  Added PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to $SHELL_RC"
else
    echo "  Already set in $SHELL_RC"
fi

# ── 6. tmux (check / suggest install) ───────────────────────────────
echo "[6/6] Checking tmux..."
if command -v tmux &>/dev/null; then
    echo "  tmux is installed ($(tmux -V))."
else
    echo "  tmux not found. Install with: brew install tmux"
    echo "  tmux lets you detach training sessions and reconnect via SSH."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Recommended workflow:"
echo "  1. SSH into the Mini: ssh $(whoami)@<mini-ip>"
echo "  2. Start a tmux session: tmux new -s train"
echo "  3. Run training:"
echo "     cd ~/iaq4j"
echo "     git pull"
echo "     nice -n -10 python ablation_log_voc.py --epochs 200 --early-stopping 15"
echo "  4. Detach: Ctrl+B, then D"
echo "  5. Reconnect later: ssh ... && tmux attach -t train"
echo ""
echo "Optional: bump batch_size in model_config.yaml from 32 → 64 or 128"
echo "to better saturate the M4 GPU. Monitor memory with: sudo powermetrics --samplers gpu_power"
