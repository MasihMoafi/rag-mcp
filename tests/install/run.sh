#!/usr/bin/env bash
# Build the wheel, then run the install acceptance checks in Docker.
# Usage: tests/install/run.sh [--skip-fastembed]
set -euo pipefail
cd "$(dirname "$0")/../.."

uv build >/dev/null
docker build -f tests/install/Dockerfile -t ragmcp-install-test .
if [ "${1:-}" = "--skip-fastembed" ]; then
  docker run --rm -e SKIP_FASTEMBED=1 ragmcp-install-test
else
  docker run --rm ragmcp-install-test
fi
