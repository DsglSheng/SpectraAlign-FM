#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${DIR}/launch_task.sh" "detection_had_viareggio" "${1:-2}" "${@:2}"
