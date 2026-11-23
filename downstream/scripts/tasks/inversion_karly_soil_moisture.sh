#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${DIR}/launch_task.sh" "inversion_karly_soil_moisture" "${1:-3}" "${@:2}"
