#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "scripts/bootstrap-showdown.sh is deprecated; forwarding to ./scripts/bootstrap-node-deps.sh" >&2
exec "${ROOT_DIR}/scripts/bootstrap-node-deps.sh" "$@"
