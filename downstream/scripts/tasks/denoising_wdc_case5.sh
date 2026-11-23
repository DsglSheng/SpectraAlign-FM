#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${DIR}/launch_task.sh" "denoising_wdc_case5" "${1:-0}" "${@:2}"
