#!/usr/bin/env bash
# -------------------------------------------------------
# start.sh — One-command setup & launch
# Usage: bash start.sh
# -------------------------------------------------------
set -e

echo "⬡  Geopolitical & Market Intelligence System v2"
echo "------------------------------------------------"

# 1. Create virtual environment if needed
if [ ! -d ".venv" ]; then
  echo "→ Creating virtual environment…"
  python3 -m venv .venv
fi

# 2. Activate
source .venv/bin/activate

# 3. Install dependencies
echo "→ Installing dependencies…"
pip install -q -r requirements.txt

# 4. Launch
echo ""
echo "✓ Starting server at http://localhost:8000"
echo "  Dashboard : http://localhost:8000"
echo "  API docs  : http://localhost:8000/docs"
echo ""
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
