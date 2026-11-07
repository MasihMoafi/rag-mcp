#!/usr/bin/env bash
# Build the wheel, then run the install acceptance checks in Docker.
# Usage: tests/install/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

uv build >/dev/null
docker build -f tests/install/Dockerfile -t ragmcp-install-test .
docker run --rm ragmcp-install-test
