#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Build the CodeDebug-RL Docker image
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${1:-codedebug-rl}"
TAG="${2:-latest}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🐳 Building CodeDebug-RL Docker Image"
echo "  Image: ${IMAGE_NAME}:${TAG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_ROOT"

docker build \
    -f codedebug_env/server/Dockerfile \
    -t "${IMAGE_NAME}:${TAG}" \
    .

echo ""
echo "✅ Image built: ${IMAGE_NAME}:${TAG}"
echo ""
echo "Run with:"
echo "  docker run -p 8000:8000 ${IMAGE_NAME}:${TAG}"
echo ""
echo "With web interface:"
echo "  docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true ${IMAGE_NAME}:${TAG}"
