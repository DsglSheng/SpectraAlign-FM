#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/Data/SpectralData/Aware_open/inversion_benchmark_20260624}"
RAW="${ROOT}/data/raw"

mkdir -p "${RAW}" "${ROOT}/logs"
cd "${RAW}"

curl -L --fail --retry 8 --retry-delay 10 --continue-at - \
  --output lithium_powder.zip \
  "https://zenodo.org/api/records/19142250/files/1-Powder.zip/content"

echo "a1030bb43f1aef99382a76b134fd83f6  lithium_powder.zip" | md5sum -c -
rm -rf "${RAW}/lithium_powder.tmp"
mkdir -p "${RAW}/lithium_powder.tmp"
unzip -q "${RAW}/lithium_powder.zip" -d "${RAW}/lithium_powder.tmp"
rm -rf "${RAW}/lithium_powder"
mv "${RAW}/lithium_powder.tmp" "${RAW}/lithium_powder"
printf '%s\n' "$(date -Iseconds)" > "${RAW}/lithium_powder/.extracted"
find "${RAW}/lithium_powder" -maxdepth 5 -type f -printf '%P\n' | sort \
  > "${ROOT}/logs/lithium_powder_file_inventory.txt"
