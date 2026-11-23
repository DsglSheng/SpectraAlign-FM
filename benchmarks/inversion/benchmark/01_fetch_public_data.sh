#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/Data/SpectralData/Aware_open/inversion_benchmark_20260624}"
RAW="${ROOT}/data/raw"
LOG="${ROOT}/logs"

mkdir -p "${RAW}" "${LOG}"
cd "${RAW}"

download() {
  local url="$1"
  local output="$2"
  curl -L --fail --retry 5 --retry-delay 5 --continue-at - \
    --output "${output}" "${url}"
}

download \
  "https://huggingface.co/datasets/Avatarr05/GreenHyperSpectra/resolve/main/labeled_splits/train.csv?download=true" \
  "greenhyperspectra_train.csv"
download \
  "https://huggingface.co/datasets/Avatarr05/GreenHyperSpectra/resolve/main/labeled_splits/test.csv?download=true" \
  "greenhyperspectra_test.csv"
download \
  "https://zenodo.org/api/records/19142250/files/1-Powder.zip/content" \
  "lithium_powder.zip"

echo "a1030bb43f1aef99382a76b134fd83f6  lithium_powder.zip" | md5sum -c -

if [[ ! -f "${RAW}/lithium_powder/.extracted" ]]; then
  rm -rf "${RAW}/lithium_powder.tmp"
  mkdir -p "${RAW}/lithium_powder.tmp"
  unzip -q "${RAW}/lithium_powder.zip" -d "${RAW}/lithium_powder.tmp"
  mv "${RAW}/lithium_powder.tmp" "${RAW}/lithium_powder"
  printf '%s\n' "$(date -Iseconds)" > "${RAW}/lithium_powder/.extracted"
fi

wc -c greenhyperspectra_train.csv greenhyperspectra_test.csv lithium_powder.zip
find "${RAW}/lithium_powder" -maxdepth 4 -type f -printf '%P\n' | sort \
  > "${ROOT}/logs/lithium_powder_file_inventory.txt"

printf '[%s] public inversion datasets downloaded and extracted\n' "$(date -Iseconds)"
