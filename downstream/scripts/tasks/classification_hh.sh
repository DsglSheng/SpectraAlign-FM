#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${DIR}/launch_task.sh" "classification_hh" "${1:-3}" "${@:2}"
