from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from huggingface_hub import hf_hub_download


REPO_ID = "hkristen/oscd"
FILES = [
    "Onera Satellite Change Detection dataset - Images.zip",
    "Onera Satellite Change Detection dataset - Train Labels.zip",
    "Onera Satellite Change Detection dataset - Test Labels.zip",
]
BANDS = [
    "B01.tif",
    "B02.tif",
    "B03.tif",
    "B04.tif",
    "B05.tif",
    "B06.tif",
    "B07.tif",
    "B08.tif",
    "B8A.tif",
    "B09.tif",
    "B10.tif",
    "B11.tif",
    "B12.tif",
]


def _download(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_paths = [output_dir / filename for filename in FILES]
    if all(path.exists() for path in local_paths):
        return local_paths
    cache_dir = output_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in FILES:
        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(cache_dir),
        )
        paths.append(Path(path))
    return paths


def _extract(zip_paths: list[Path], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_paths:
        marker = raw_dir / f".extracted_{zip_path.stem.replace(' ', '_')}"
        if marker.exists():
            continue
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        marker.write_text("ok\n", encoding="utf-8")


def _find_city_dirs(raw_dir: Path) -> dict[str, Path]:
    city_dirs: dict[str, Path] = {}
    for path in raw_dir.rglob("imgs_1_rect"):
        city_dir = path.parent
        if (city_dir / "imgs_2_rect").exists():
            city_dirs[city_dir.name] = city_dir
    return city_dirs


def _find_label_dirs(raw_dir: Path, label_kind: str) -> dict[str, Path]:
    label_dirs: dict[str, Path] = {}
    needles = {
        "train": "train labels",
        "test": "test labels",
    }
    needle = needles[label_kind]
    for cm in raw_dir.rglob("cm.png"):
        lower = str(cm).lower()
        if needle in lower:
            label_dirs[cm.parent.parent.name] = cm.parent.parent
    return label_dirs


def _read_band(path: Path) -> np.ndarray:
    arr = np.asarray(imageio.imread(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _stack_bands(folder: Path) -> np.ndarray:
    arrays = []
    for band in BANDS:
        band_path = folder / band
        if not band_path.exists():
            raise FileNotFoundError(f"Missing Sentinel-2 band {band_path}")
        arrays.append(_read_band(band_path))
    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Band shapes differ under {folder}: {sorted(shapes)}")
    return np.stack(arrays, axis=-1).astype(np.float32, copy=False)


def _copy_city(city: str, image_dir: Path, label_dir: Path, output_dir: Path) -> None:
    target = output_dir / city
    (target / "imgs_1_rect").mkdir(parents=True, exist_ok=True)
    (target / "imgs_2_rect").mkdir(parents=True, exist_ok=True)
    (target / "cm").mkdir(parents=True, exist_ok=True)

    pre_out = target / "imgs_1_rect" / "ms_data.npy"
    post_out = target / "imgs_2_rect" / "ms_data.npy"
    if not pre_out.exists():
        np.save(pre_out, _stack_bands(image_dir / "imgs_1_rect"))
    if not post_out.exists():
        np.save(post_out, _stack_bands(image_dir / "imgs_2_rect"))
    shutil.copy2(label_dir / "cm" / "cm.png", target / "cm" / "cm.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OSCD data in the layout used by HyperSIGMA MultispectralCD.")
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/Data/SpectralData/Aware_open/data/oscd/original_data"))
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/Data/SpectralData/Aware_open/downloads/oscd"))
    args = parser.parse_args()

    zip_paths = _download(args.work_dir)
    raw_dir = args.work_dir / "raw"
    _extract(zip_paths, raw_dir)

    image_dirs = _find_city_dirs(raw_dir)
    train_labels = _find_label_dirs(raw_dir, "train")
    test_labels = _find_label_dirs(raw_dir, "test")
    if not image_dirs:
        raise RuntimeError(f"No OSCD image city directories found under {raw_dir}")
    if not train_labels or not test_labels:
        raise RuntimeError(f"Missing train/test label directories under {raw_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_names = sorted(set(train_labels) & set(image_dirs))
    test_names = sorted(set(test_labels) & set(image_dirs))
    missing = sorted((set(train_labels) | set(test_labels)) - set(image_dirs))
    if missing:
        raise RuntimeError(f"Labels exist without matching images: {missing}")

    for city in train_names:
        _copy_city(city, image_dirs[city], train_labels[city], args.output_dir)
    for city in test_names:
        _copy_city(city, image_dirs[city], test_labels[city], args.output_dir)

    (args.output_dir / "train.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (args.output_dir / "test.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")
    print(f"prepared={args.output_dir}")
    print(f"train={len(train_names)} {train_names}")
    print(f"test={len(test_names)} {test_names}")


if __name__ == "__main__":
    main()
