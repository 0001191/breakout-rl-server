#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/breakout_rl_server}"

echo "[1/5] Create project dir: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"

echo "[2/5] Copy project files into place before running this script or clone your repo here."
echo "[3/5] Create venv"
python3 -m venv "$PROJECT_DIR/.venv"
source "$PROJECT_DIR/.venv/bin/activate"

echo "[4/5] Install dependencies"
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

echo "[5/5] Start dashboard server"
cd "$PROJECT_DIR"
nohup "$PROJECT_DIR/.venv/bin/python" server_app.py > "$PROJECT_DIR/artifacts/live/server.log" 2>&1 &
echo "Dashboard server started."
echo "Open: http://<your-server-ip>:8000/dashboard/"
