#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run the CodeDebug-RL demo locally (no Docker required)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🐛 CodeDebug-RL — Local Demo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_ROOT"

# Install dependencies if needed
if ! python -c "import codedebug_env" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -e ".[dev]" --quiet
fi

echo ""
echo "🚀 Running demo episode..."
echo ""

python demo.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  To start the server:  uvicorn codedebug_env.server.app:app --reload"
echo "  With web UI:          ENABLE_WEB_INTERFACE=true uvicorn codedebug_env.server.app:app --reload"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
