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

# 4. Stability guards — keep OpenMP (torch + xgboost) single-threaded so the
#    background ML/ingestion work can't segfault the worker. (main.py also sets
#    these, but exporting here covers any direct/alternate launch too.)
export OMP_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

# 5. Launch
#    NOTE: no --reload. Reload watches files and, more importantly, does NOT
#    restart a worker that exits on its own — so any worker crash would leave
#    the parent holding the port open and the page hanging. Run plain.
echo ""
echo "✓ Starting server at http://127.0.0.1:8000"
echo "  Dashboard : http://127.0.0.1:8000"
echo "  API docs  : http://127.0.0.1:8000/docs"
echo ""
uvicorn main:app --host 127.0.0.1 --port 8000
