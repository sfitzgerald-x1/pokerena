#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOWDOWN_DIR="${ROOT_DIR}/vendor/pokemon-showdown"

if ! command -v node >/dev/null 2>&1; then
  echo "node is required on PATH (need Node.js 22+)." >&2
  exit 1
fi

NODE_VERSION="$(node --version)"
NODE_MAJOR="${NODE_VERSION#v}"
NODE_MAJOR="${NODE_MAJOR%%.*}"
if [[ "${NODE_MAJOR}" -lt 22 ]]; then
  echo "Node.js 22+ is required; found ${NODE_VERSION}." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required on PATH." >&2
  exit 1
fi

if [[ ! -f "${SHOWDOWN_DIR}/pokemon-showdown" ]]; then
  if command -v git >/dev/null 2>&1 && [[ -d "${ROOT_DIR}/.git" ]]; then
    git -C "${ROOT_DIR}" submodule update --init --recursive vendor/pokemon-showdown
  else
    echo "Pokemon Showdown checkout is missing from ${SHOWDOWN_DIR}. Initialize the submodule first." >&2
    exit 1
  fi
fi

npm --prefix "${ROOT_DIR}" ci
npm --prefix "${SHOWDOWN_DIR}" ci

echo "Pokerena node dependencies are bootstrapped in ${ROOT_DIR}."
