#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Validate the CodeDebug-RL submission for completeness
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ CodeDebug-RL — Submission Validator"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PASS=0
FAIL=0

check() {
    local desc="$1"
    local result="$2"
    if [ "$result" = "0" ]; then
        echo "  ✅ $desc"
        PASS=$((PASS + 1))
    else
        echo "  ❌ $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "📁 Checking required files..."

# Required files
REQUIRED_FILES=(
    "codedebug_env/__init__.py"
    "codedebug_env/models.py"
    "codedebug_env/client.py"
    "codedebug_env/server/__init__.py"
    "codedebug_env/server/environment.py"
    "codedebug_env/server/app.py"
    "codedebug_env/server/executor.py"
    "codedebug_env/server/reward.py"
    "codedebug_env/server/task_loader.py"
    "codedebug_env/server/diff_utils.py"
    "codedebug_env/server/sandbox.py"
    "codedebug_env/server/telemetry.py"
    "codedebug_env/server/Dockerfile"
    "openenv.yaml"
    "pyproject.toml"
    "requirements.txt"
    "README.md"
    "demo.py"
    "configs/default.yaml"
    "configs/rewards.yaml"
)

for f in "${REQUIRED_FILES[@]}"; do
    test -f "$f" && check "$f" 0 || check "$f" 1
done

echo ""
echo "🔍 Running linters and type checkers..."
if python -m ruff check .; then
    check "Ruff syntax & linting" 0
else
    check "Ruff syntax & linting" 1
fi

if python -m mypy codedebug_env/; then
    check "Mypy type checking (core)" 0
else
    check "Mypy type checking (core)" 1
fi

echo ""
echo "🧪 Running unit tests..."
if python -m pytest tests/ -v --tb=short; then
    check "Test suite passes" 0
else
    check "Test suite passes" 1
fi

echo ""
echo "🐍 Checking imports..."
if python -c "from codedebug_env import CodeDebugAction, CodeDebugObservation" 2>/dev/null; then
    check "Package importable" 0
else
    check "Package importable" 1
fi

echo ""
echo "🎮 Running demo..."
if python demo.py >/dev/null 2>&1; then
    check "Demo runs successfully" 0
else
    check "Demo runs successfully" 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: ${PASS} passed, ${FAIL} failed"
if [ "$FAIL" -eq 0 ]; then
    echo "  🎉 Submission is VALID"
else
    echo "  ⚠️  Submission has issues — please fix the failures above"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $FAIL
