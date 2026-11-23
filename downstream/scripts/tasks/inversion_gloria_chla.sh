#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${DIR}/launch_task.sh" "inversion_gloria_chla" "${1:-0}" "${@:2}"
