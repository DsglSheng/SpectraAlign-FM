from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import h5py
except ImportError:  # pragma: no cover - only used for MATLAB v7.3 fallbacks.
    h5py = None

try:
    from sklearn.decomposition import PCA
except ImportError:  # pragma: no cover - optional dependency for HyperSIGMA-style PCA experiments.
    PCA = None


DATASET_ALIASES: dict[str, str] = {
    "ip": "indian_pines",
    "indian_pines": "indian_pines",
    "indianpines": "indian_pines",
    "pu": "paviau",
    "paviau": "paviau",
    "pavia_centre": "pavia_center",
    "pavia_center": "pavia_center",
    "pavia": "pavia_center",
    "pc": "pavia_center",
    "salinas": "salinas",
    "sa": "salinas",
    "hc": "hanchuan",
    "hanchuan": "hanchuan",
    "hh": "honghu",
    "honghu": "honghu",
    "hu": "houston2013",
    "houston": "houston2013",
    "houston2013": "houston2013",
    "houston_2013": "houston2013",
    "zy": "zy1_02d",
    "zy1": "zy1_02d",
    "zy102d": "zy1_02d",
    "zy1_02d": "zy1_02d",
}

SCENE_CLASS_NAMES: dict[str, list[str]] = {
    "indian_pines": [
        "Alfalfa",
        "Corn-notill",
        "Corn-mintill",
        "Corn",
        "Grass-pasture",
        "Grass-trees",
        "Grass-pasture-mowed",
        "Hay-windrowed",
        "Oats",
        "Soybean-notill",
        "Soybean-mintill",
        "Soybean-clean",
        "Wheat",
        "Woods",
        "Buildings-Grass-Trees-Drives",
        "Stone-Steel-Towers",
    ],
    "paviau": [
        "Asphalt",
        "Meadows",
        "Gravel",
        "Trees",
        "Painted metal sheets",
        "Bare Soil",
        "Bitumen",
        "Self-Blocking Bricks",
        "Shadows",
    ],
    "pavia_center": [
        "Water",
        "Trees",
        "Asphalt",
        "Self-Blocking Bricks",
        "Bitumen",
        "Tiles",
        "Shadows",
        "Meadows",
        "Bare Soil",
    ],
    "salinas": [
        "Brocoli green weeds 1",
        "Brocoli green weeds 2",
        "Fallow",
        "Fallow rough plow",
        "Fallow smooth",
        "Stubble",
        "Celery",
        "Grapes untrained",
        "Soil vineyard develop",
        "Corn senesced green weeds",
        "Lettuce romaine 4wk",
        "Lettuce romaine 5wk",
        "Lettuce romaine 6wk",
        "Lettuce romaine 7wk",
        "Vineyard untrained",
        "Vineyard vertical trellis",
    ],
    "hanchuan": [
        "Strawberry",
        "Cowpea",
        "Soybean",
        "Sorghum",
        "Water spinach",
        "Watermelon",
        "Greens",
        "Trees",
        "Grass",
        "Red roof",
        "Gray roof",
        "Plastic",
        "Bare soil",
        "Road",
        "Bright object",
        "Water",
    ],
    "honghu": [
        "Red roof",
        "Road",
        "Bare soil",
        "Cotton",
        "Cotton firewood",
        "Rape",
        "Chinese cabbage",
        "Pakchoi",
        "Cabbage",
        "Tuber mustard",
        "Brassica parachinensis",
        "Brassica chinensis",
        "Small Brassica chinensis",
        "Lactuca sativa",
        "Celtuce",
        "Film covered lettuce",
        "Romaine lettuce",
        "Carrot",
        "White radish",
        "Garlic sprout",
        "Broad bean",
        "Tree",
    ],
    "houston2013": [
        "Healthy grass",
        "Stressed grass",
        "Synthetic grass",
        "Trees",
        "Soil",
        "Water",
        "Residential",
        "Commercial",
        "Road",
        "Highway",
        "Railway",
        "Parking Lot 1",
        "Parking Lot 2",
        "Tennis Court",
        "Running Track",
    ],
    "zy1_02d": [f"Class {idx}" for idx in range(1, 8)],
}


@dataclass(frozen=True)
class ClassificationSceneConfig:
    dataset: str
    sensor: str
    level: str
    gsd_m: float
    wavelengths: torch.Tensor
    fwhm: torch.Tensor
    metadata_note: str
    class_names: tuple[str, ...]
    cube_filename: str
    cube_key: str | None
    gt_filename: str | None
    gt_key: str | None
    img_size: int
    overlap_size: int
    pca_components: int | None
    train_num: int | None
    val_num: int | None
    split_mode: str = "number"
    train_gt_filename: str | None = None
    train_gt_key: str | None = None
    test_gt_filename: str | None = None
    test_gt_key: str | None = None
    official_val_ratio: float = 0.1


class SceneTileDataset(Dataset):
    def __init__(self, tiles: np.ndarray, labels: np.ndarray, *, augment: bool = False) -> None:
        if tiles.ndim != 4:
            raise ValueError(f"Expected tiles shaped (N, H, W, C), got {tiles.shape}.")
        if labels.ndim != 3:
            raise ValueError(f"Expected labels shaped (N, H, W), got {labels.shape}.")
        if tiles.shape[0] != labels.shape[0] or tiles.shape[1:3] != labels.shape[1:3]:
            raise ValueError("Tile and label shapes are inconsistent.")
        self.tiles = tiles.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return int(self.tiles.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile = self.tiles[index]
        label = self.labels[index]
        if self.augment:
            if random.random() < 0.5:
                tile = np.flip(tile, axis=0)
                label = np.flip(label, axis=0)
            if random.random() < 0.5:
                tile = np.flip(tile, axis=1)
                label = np.flip(label, axis=1)
            rotations = random.randint(0, 3)
            if rotations:
                tile = np.rot90(tile, k=rotations, axes=(0, 1))
                label = np.rot90(label, k=rotations, axes=(0, 1))
        return torch.from_numpy(tile.transpose(2, 0, 1).copy()), torch.from_numpy(label.copy()).long()


class CenteredLabelCropDataset(Dataset):
    """Segmentation-style crops centered on labeled pixels.

    This keeps the dense label map supervision format used by the seg head while
    avoiding the degenerate single-tile training regime on tiny scenes like
    Indian Pines.
    """

    def __init__(
        self,
        cube: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        crop_size: int,
        augment: bool = False,
        spatial_jitter: int = 0,
        repeat_factor: int = 1,
    ) -> None:
        if cube.ndim != 3:
            raise ValueError(f"Expected cube shaped (H, W, C), got {cube.shape}.")
        if labels.ndim != 2:
            raise ValueError(f"Expected label map shaped (H, W), got {labels.shape}.")
        if cube.shape[:2] != labels.shape:
            raise ValueError("Cube and label map spatial shapes are inconsistent.")
        if crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {crop_size}.")
        if repeat_factor <= 0:
            raise ValueError(f"repeat_factor must be positive, got {repeat_factor}.")
        self.cube = cube.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.indices = indices.astype(np.int64).reshape(-1)
        self.crop_size = int(crop_size)
        self.augment = bool(augment)
        self.spatial_jitter = max(0, int(spatial_jitter))
        self.repeat_factor = int(repeat_factor)
        self.height, self.width, _ = self.cube.shape
        self.margin = self.crop_size // 2
        self.padded_cube = np.pad(
            self.cube,
            ((self.margin, self.margin), (self.margin, self.margin), (0, 0)),
            mode="reflect",
        )
        self.padded_labels = np.pad(
            self.labels,
            ((self.margin, self.margin), (self.margin, self.margin)),
            mode="constant",
        )

    def __len__(self) -> int:
        return int(self.indices.shape[0]) * self.repeat_factor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat_index = int(self.indices[index % int(self.indices.shape[0])])
        row = flat_index // self.width
        col = flat_index % self.width
        center_row = row
        center_col = col
        if self.augment and self.spatial_jitter > 0:
            center_row = int(np.clip(row + random.randint(-self.spatial_jitter, self.spatial_jitter), 0, self.height - 1))
            center_col = int(np.clip(col + random.randint(-self.spatial_jitter, self.spatial_jitter), 0, self.width - 1))

        tile = self.padded_cube[center_row : center_row + self.crop_size, center_col : center_col + self.crop_size, :]
        label = self.padded_labels[center_row : center_row + self.crop_size, center_col : center_col + self.crop_size]
        if self.augment:
            if random.random() < 0.5:
                tile = np.flip(tile, axis=0)
                label = np.flip(label, axis=0)
            if random.random() < 0.5:
                tile = np.flip(tile, axis=1)
                label = np.flip(label, axis=1)
            rotations = random.randint(0, 3)
            if rotations:
                tile = np.rot90(tile, k=rotations, axes=(0, 1))
                label = np.rot90(label, k=rotations, axes=(0, 1))
        return torch.from_numpy(tile.transpose(2, 0, 1).copy()), torch.from_numpy(label.copy()).long()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_cube(cube: np.ndarray, mode: str = "minmax") -> tuple[np.ndarray, dict[str, object]]:
    data = cube.astype(np.float32, copy=False)
    normalized_mode = mode.lower()
    if normalized_mode in {"none", "raw"}:
        return data.astype(np.float32, copy=False), {"mode": "none"}
    if normalized_mode == "minmax":
        low = float(np.min(data))
        high = float(np.max(data))
        scale = high - low
        if not np.isfinite(scale) or scale < 1.0e-12:
            scale = 1.0
        return ((data - low) / scale).astype(np.float32), {"mode": "minmax", "min": low, "max": high}
    if normalized_mode == "mean_std":
        center = float(data.mean())
        scale = float(data.std())
        if not np.isfinite(scale) or scale < 1.0e-6:
            scale = 1.0
        return ((data - center) / scale).astype(np.float32), {"mode": "mean_std", "mean": center, "std": scale}
    raise ValueError(f"Unsupported normalization mode {mode!r}.")


def apply_pca_cube(
    cube: np.ndarray,
    *,
    num_components: int,
    whiten: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    if PCA is None:
        raise ImportError("scikit-learn is required for PCA-based classification preprocessing.")
    data = cube.astype(np.float32, copy=False)
    if data.ndim != 3:
        raise ValueError(f"Expected cube shaped (H, W, C), got {data.shape}.")
    height, width, channels = data.shape
    components = int(num_components)
    if components <= 0:
        raise ValueError(f"num_components must be positive, got {components}.")
    if components > channels:
        raise ValueError(f"num_components={components} exceeds channel count={channels}.")
    flat = data.reshape(-1, channels)
    pca = PCA(n_components=components, whiten=bool(whiten))
    transformed = pca.fit_transform(flat).astype(np.float32, copy=False).reshape(height, width, components)
    explained = getattr(pca, "explained_variance_ratio_", None)
    return transformed, {
        "applied": True,
        "num_components": components,
        "whiten": bool(whiten),
        "original_channels": int(channels),
        "explained_variance_ratio_sum": None if explained is None else float(np.sum(explained)),
    }


def canonicalize_dataset_name(dataset: str) -> str:
    normalized = str(dataset).lower().strip().replace("-", "_")
    return DATASET_ALIASES.get(normalized, normalized)


def default_wavelengths(dataset: str, num_bands: int) -> torch.Tensor:
    normalized = canonicalize_dataset_name(dataset)
    if normalized == "indian_pines":
        if num_bands == 200:
            removed = set(range(104, 108)) | set(range(150, 164)) | {220}
            full = [400.0 + (2500.0 - 400.0) * idx / 219.0 for idx in range(220) if (idx + 1) not in removed]
            if len(full) == 200:
                return torch.tensor(full, dtype=torch.float32)
        return torch.linspace(400.0, 2500.0, int(num_bands), dtype=torch.float32)
    if normalized == "paviau":
        return torch.linspace(430.0, 860.0, int(num_bands), dtype=torch.float32)
    if normalized == "pavia_center":
        return torch.linspace(430.0, 860.0, int(num_bands), dtype=torch.float32)
    if normalized == "salinas":
        if num_bands == 204:
            removed = set(range(108, 113)) | set(range(154, 168)) | {224}
            full = [400.0 + (2500.0 - 400.0) * idx / 223.0 for idx in range(224) if (idx + 1) not in removed]
            if len(full) == 204:
                return torch.tensor(full, dtype=torch.float32)
        return torch.linspace(400.0, 2500.0, int(num_bands), dtype=torch.float32)
    if normalized in {"hanchuan", "honghu"}:
        return torch.linspace(400.0, 1000.0, int(num_bands), dtype=torch.float32)
    if normalized == "houston2013":
        return torch.linspace(364.0, 1046.1, int(num_bands), dtype=torch.float32)
    if normalized == "zy1_02d":
        return torch.linspace(400.0, 2500.0, int(num_bands), dtype=torch.float32)
    raise ValueError(f"Unsupported dataset {dataset!r}.")


def fwhm_from_wavelengths(wavelengths_nm: torch.Tensor) -> torch.Tensor:
    wavelengths = wavelengths_nm.float().flatten()
    if wavelengths.numel() == 1:
        return torch.ones_like(wavelengths)
    diffs = torch.diff(wavelengths).abs()
    left = torch.cat([diffs[:1], diffs])
    right = torch.cat([diffs, diffs[-1:]])
    return ((left + right) * 0.5).clamp_min(1.0)


def load_mat_array(path: Path, preferred_key: str | None, *, expected_ndim: int | tuple[int, ...]) -> np.ndarray:
    ndim_tuple = (expected_ndim,) if isinstance(expected_ndim, int) else tuple(int(item) for item in expected_ndim)

    def choose_candidate(payload: dict[str, Any]) -> np.ndarray:
        if preferred_key is not None and preferred_key in payload:
            array = np.asarray(payload[preferred_key])
            if array.ndim in ndim_tuple:
                return array
        candidates: list[tuple[int, str, np.ndarray]] = []
        for key, value in payload.items():
            if key.startswith("__"):
                continue
            array = np.asarray(value)
            if array.ndim not in ndim_tuple or array.size <= 0:
                continue
            candidates.append((int(array.size), key, array))
        if not candidates:
            raise KeyError(f"{path.name} does not contain any array with ndim in {ndim_tuple}.")
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][2]

    try:
        return choose_candidate(sio.loadmat(path))
    except NotImplementedError:
        if h5py is None:
            raise
        with h5py.File(path, "r") as handle:
            payload = {key: np.array(handle[key]) for key in handle.keys()}
        array = choose_candidate(payload)
        if array.ndim == 2:
            return array.T
        return np.transpose(array, axes=tuple(reversed(range(array.ndim))))


def resolve_scene_config(dataset: str) -> ClassificationSceneConfig:
    normalized = canonicalize_dataset_name(dataset)
    if normalized == "indian_pines":
        wavelengths = default_wavelengths(normalized, 200)
        return ClassificationSceneConfig(
            dataset=normalized,
            sensor="eo1",
            level="L1",
            gsd_m=20.0,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="Raw Indian Pines bands use approximate AVIRIS wavelengths after common water-band removal.",
            class_names=tuple(SCENE_CLASS_NAMES[normalized]),
            cube_filename="Indian_pines_corrected.mat",
            cube_key="data",
            gt_filename="Indian_pines_gt.mat",
            gt_key="groundT",
            img_size=64,
            overlap_size=32,
            pca_components=None,
            train_num=50,
            val_num=16,
        )
    if normalized == "paviau":
        wavelengths = default_wavelengths(normalized, 103)
        return ClassificationSceneConfig(
            dataset=normalized,
            sensor="rosis",
            level="surface",
            gsd_m=1.3,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="Raw PaviaU bands use approximate ROSIS wavelengths across the visible to near-infrared range.",
            class_names=tuple(SCENE_CLASS_NAMES[normalized]),
            cube_filename="PaviaU.mat",
            cube_key="paviaU",
            gt_filename="PaviaU_gt.mat",
            gt_key="paviaU_gt",
            img_size=128,
            overlap_size=32,
            pca_components=None,
            train_num=20,
            val_num=20,
        )
    if normalized == "pavia_center":
        wavelengths = default_wavelengths(normalized, 102)
        return ClassificationSceneConfig(
            dataset=normalized,
            sensor="rosis",
            level="surface",
            gsd_m=1.3,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="Pavia Centre ROSIS urban scene from EHU, held out from the main Aware-HSI comparison tables.",
            class_names=tuple(SCENE_CLASS_NAMES[normalized]),
            cube_filename="Pavia.mat",
            cube_key="pavia",
            gt_filename="Pavia_gt.mat",
            gt_key="pavia_gt",
            img_size=128,
            overlap_size=32,
            pca_components=None,
            train_num=50,
            val_num=20,
        )
    if normalized == "salinas":
        wavelengths = default_wavelengths(normalized, 204)
        return ClassificationSceneConfig(
            dataset=normalized,
            sensor="eo1",
            level="L1",
            gsd_m=3.7,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="Salinas Valley AVIRIS crop scene from EHU. The downstream metadata uses the EO-1-style wide VNIR-SWIR spectral vocabulary used by the project for AVIRIS-like public scenes.",
            class_names=tuple(SCENE_CLASS_NAMES[normalized]),
            cube_filename="Salinas_corrected.mat",
            cube_key="salinas_corrected",
            gt_filename="Salinas_gt.mat",
            gt_key="salinas_gt",
            img_size=128,
            overlap_size=32,
            pca_components=None,
            train_num=50,
            val_num=20,
        )
    if normalized == "hanchuan":
        wavelengths = default_wavelengths(normalized, 274)
        return ClassificationSceneConfig(
            dataset=normalized,
            sensor="headwall",
            level="surface",
            gsd_m=0.109,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="WHU-Hi HanChuan UAV hyperspectral scene with approximate 400-1000 nm coverage.",
            class_names=tuple(SCENE_CLASS_NAMES[normalized]),
            cube_filename="WHU_Hi_HanChuan.mat",
            cube_key="WHU_Hi_HanChuan",
            gt_filename="WHU_Hi_HanChuan_gt.mat",
            gt_key="WHU_Hi_HanChuan_gt",
            img_size=128,
            overlap_size=32,
            pca_components=None,
            train_num=50,
            val_num=20,
        )
    if normalized == "honghu":
        wavelengths = default_wavelengths(normalized, 270)
        return ClassificationSceneConfig(
            dataset=normalized,
            sensor="headwall",
            level="surface",
            gsd_m=0.043,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="WHU-Hi HongHu UAV hyperspectral scene with approximate 400-1000 nm coverage.",
            class_names=tuple(SCENE_CLASS_NAMES[normalized]),
            cube_filename="WHU_Hi_HongHu.mat",
            cube_key="WHU_Hi_HongHu",
            gt_filename="WHU_Hi_HongHu_gt.mat",
            gt_key="WHU_Hi_HongHu_gt",
            img_size=128,
            overlap_size=32,
            pca_components=None,
            train_num=50,
            val_num=20,
        )
    if normalized == "houston2013":
        wavelengths = default_wavelengths(normalized, 144)
        return ClassificationSceneConfig(
            dataset=normalized,
            sensor="casi",
            level="L1",
            gsd_m=2.5,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="Houston 2013 official IEEE GRSS Data Fusion Contest split with approximate CASI wavelengths.",
            class_names=tuple(SCENE_CLASS_NAMES[normalized]),
            cube_filename="Houston2013.mat",
            cube_key="houston2013",
            gt_filename=None,
            gt_key=None,
            img_size=128,
            overlap_size=32,
            pca_components=None,
            train_num=None,
            val_num=20,
            split_mode="official",
            train_gt_filename="Houston2013_train_gt.mat",
            train_gt_key="houston2013_train_gt",
            test_gt_filename="Houston2013_test_gt.mat",
            test_gt_key="houston2013_test_gt",
            official_val_ratio=0.1,
        )
    if normalized == "zy1_02d":
        wavelengths = default_wavelengths(normalized, 147)
        return ClassificationSceneConfig(
            dataset=normalized,
            sensor="gf5",
            level="surface",
            gsd_m=30.0,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="ZY1-02D AHSI Yancheng wetland scene with approximate 400-2500 nm coverage.",
            class_names=tuple(SCENE_CLASS_NAMES[normalized]),
            cube_filename="ZY_YC_data147.mat",
            cube_key="Data",
            gt_filename="ZY_YC_gt7.mat",
            gt_key="DataClass",
            img_size=128,
            overlap_size=32,
            pca_components=None,
            train_num=20,
            val_num=20,
        )
    raise ValueError(f"Unsupported dataset {dataset!r}. Expected one of {sorted(SCENE_CLASS_NAMES)} or aliases {sorted(DATASET_ALIASES)}.")


def load_scene(data_root: Path, config: ClassificationSceneConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    cube_path = data_root / config.cube_filename
    if not cube_path.exists():
        raise FileNotFoundError(f"Missing cube file for {config.dataset} under {data_root}: {config.cube_filename}.")

    cube = load_mat_array(cube_path, config.cube_key, expected_ndim=3).astype(np.float32)
    if config.split_mode == "official":
        if config.train_gt_filename is None or config.test_gt_filename is None:
            raise ValueError(f"{config.dataset} uses split_mode='official' but is missing train/test label files.")
        train_gt_path = data_root / config.train_gt_filename
        test_gt_path = data_root / config.test_gt_filename
        if not train_gt_path.exists() or not test_gt_path.exists():
            raise FileNotFoundError(
                f"Missing official split files for {config.dataset} under {data_root}. "
                f"Expected {config.train_gt_filename} and {config.test_gt_filename}."
            )
        train_labels = np.squeeze(
            load_mat_array(train_gt_path, config.train_gt_key, expected_ndim=(2, 3)),
        ).astype(np.int64)
        test_labels = np.squeeze(
            load_mat_array(test_gt_path, config.test_gt_key, expected_ndim=(2, 3)),
        ).astype(np.int64)
        labels = np.maximum(train_labels, test_labels)
        return cube, labels, train_labels, test_labels

    if config.gt_filename is None:
        raise ValueError(f"{config.dataset} requires gt_filename for split_mode={config.split_mode!r}.")
    gt_path = data_root / config.gt_filename
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Missing files for {config.dataset} under {data_root}. Expected {config.cube_filename} and {config.gt_filename}."
        )
    labels = np.squeeze(load_mat_array(gt_path, config.gt_key, expected_ndim=(2, 3))).astype(np.int64)
    return cube, labels, None, None


def split_labeled_indices(
    gt_flat: np.ndarray,
    class_count: int,
    *,
    train_num: int,
    val_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_index: list[np.ndarray] = []
    val_index: list[np.ndarray] = []
    test_index: list[np.ndarray] = []
    for class_id in range(1, class_count + 1):
        idx = np.where(gt_flat == class_id)[0]
        np.random.shuffle(idx)
        current_train = min(int(train_num), int(idx.shape[0]))
        remaining = max(int(idx.shape[0]) - current_train, 0)
        current_val = min(int(val_num), remaining)
        train_index.append(idx[:current_train])
        val_index.append(idx[current_train : current_train + current_val])
        test_index.append(idx[current_train + current_val :])
    return np.concatenate(train_index), np.concatenate(val_index), np.concatenate(test_index)


def split_ratio_labeled_indices(
    gt_flat: np.ndarray,
    class_count: int,
    *,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_index: list[np.ndarray] = []
    val_index: list[np.ndarray] = []
    test_index: list[np.ndarray] = []
    for class_id in range(1, class_count + 1):
        idx = np.where(gt_flat == class_id)[0]
        np.random.shuffle(idx)
        total = int(idx.shape[0])
        if total == 0:
            continue
        current_train = int(np.ceil(float(train_ratio) * float(total))) if train_ratio > 0.0 else 0
        current_train = min(max(current_train, 1 if train_ratio > 0.0 else 0), total)
        remaining = total - current_train
        if remaining <= 0:
            current_val = 0
        else:
            current_val = int(np.ceil(float(val_ratio) * float(total))) if val_ratio > 0.0 else 0
            min_val = 1 if val_ratio > 0.0 and remaining > 1 else 0
            max_val = remaining - 1 if remaining > 1 else remaining
            current_val = min(max(current_val, min_val), max_val)
        train_index.append(idx[:current_train])
        val_index.append(idx[current_train : current_train + current_val])
        test_index.append(idx[current_train + current_val :])
    return np.concatenate(train_index), np.concatenate(val_index), np.concatenate(test_index)


def split_official_training_indices(
    train_gt_flat: np.ndarray,
    class_count: int,
    *,
    val_num: int | None,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_index: list[np.ndarray] = []
    val_index: list[np.ndarray] = []
    for class_id in range(1, class_count + 1):
        idx = np.where(train_gt_flat == class_id)[0]
        np.random.shuffle(idx)
        if idx.size <= 1:
            current_val = 0
        elif val_num is not None and val_num > 0:
            current_val = min(int(val_num), int(idx.size) - 1)
        elif val_ratio > 0.0:
            current_val = min(max(int(round(float(idx.size) * float(val_ratio))), 1), int(idx.size) - 1)
        else:
            current_val = 0
        val_index.append(idx[:current_val])
        train_index.append(idx[current_val:])
    train_concat = np.concatenate(train_index) if train_index else np.empty((0,), dtype=np.int64)
    val_concat = np.concatenate(val_index) if val_index else np.empty((0,), dtype=np.int64)
    return train_concat, val_concat


def indices_to_label_map(gt_flat: np.ndarray, indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    output = np.zeros_like(gt_flat, dtype=np.int64)
    output[indices.astype(np.int64)] = gt_flat[indices.astype(np.int64)]
    return output.reshape(shape)


def pad_to_tile_grid(cube: np.ndarray, labels: np.ndarray, *, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width, _ = cube.shape
    padded_cube = cube
    padded_labels = labels
    if height < img_size:
        gap = img_size - height
        padded_cube = np.concatenate([padded_cube, padded_cube[(height - gap) : height, :, :]], axis=0)
        padded_labels = np.concatenate([padded_labels, padded_labels[(height - gap) : height, :]], axis=0)
    if width < img_size:
        gap = img_size - width
        padded_cube = np.concatenate([padded_cube, padded_cube[:, (width - gap) : width, :]], axis=1)
        padded_labels = np.concatenate([padded_labels, padded_labels[:, (width - gap) : width]], axis=1)
    height, width, _ = padded_cube.shape
    num_h = height // img_size
    num_w = width // img_size
    if height % img_size != 0:
        gap = (num_h + 1) * img_size - height
        padded_cube = np.concatenate([padded_cube, padded_cube[(height - gap) : height, :, :]], axis=0)
        padded_labels = np.concatenate([padded_labels, padded_labels[(height - gap) : height, :]], axis=0)
    if width % img_size != 0:
        gap = (num_w + 1) * img_size - width
        padded_cube = np.concatenate([padded_cube, padded_cube[:, (width - gap) : width, :]], axis=1)
        padded_labels = np.concatenate([padded_labels, padded_labels[:, (width - gap) : width]], axis=1)
    return padded_cube, padded_labels


def tile_scene(cube: np.ndarray, labels: np.ndarray, *, img_size: int, overlap_size: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    if img_size <= 0:
        raise ValueError("img_size must be positive.")
    if overlap_size < 0 or overlap_size >= img_size:
        raise ValueError("overlap_size must be in [0, img_size).")
    cube, labels = pad_to_tile_grid(cube, labels, img_size=img_size)
    height, width, _ = cube.shape
    input_size = (img_size, img_size)
    ly_end, lx_end = np.subtract((height, width), input_size)
    if ly_end < 0 or lx_end < 0:
        raise ValueError(f"Scene {(height, width)} is smaller than img_size={img_size}.")
    step = img_size - overlap_size
    lx = np.linspace(0, lx_end, int(np.ceil(lx_end / float(step))) + 1, endpoint=True).astype("int")
    ly = np.linspace(0, ly_end, int(np.ceil(ly_end / float(step))) + 1, endpoint=True).astype("int")
    tiles = []
    tile_labels = []
    starts = []
    for row in ly:
        for col in lx:
            tiles.append(cube[row : row + img_size, col : col + img_size, :])
            tile_labels.append(labels[row : row + img_size, col : col + img_size])
            starts.append((int(row), int(col)))
    return np.asarray(tiles, dtype=np.float32), np.asarray(tile_labels, dtype=np.int64), starts


def merge_tile_predictions(
    tile_predictions: list[np.ndarray],
    starts: list[tuple[int, int]],
    *,
    height: int,
    width: int,
    img_size: int,
    overlap_size: int,
) -> np.ndarray:
    merged = np.zeros((height, width), dtype=np.int64)
    half_overlap = int(overlap_size / 2)
    for idx, ((row, col), pred) in enumerate(zip(starts, tile_predictions, strict=True)):
        row_end = row + img_size
        col_end = col + img_size
        tile_row_start = 0
        tile_col_start = 0
        out_row_start = row
        out_col_start = col
        if idx != 0:
            if row > 0:
                tile_row_start = half_overlap
                out_row_start = row + half_overlap
            if col > 0:
                tile_col_start = half_overlap
                out_col_start = col + half_overlap
        merged[out_row_start:row_end, out_col_start:col_end] = pred[tile_row_start:, tile_col_start:]
    return merged


def build_segmentation_scene(
    *,
    data_root: Path,
    config: ClassificationSceneConfig,
    split_mode: str | None = None,
    img_size: int | None = None,
    overlap_size: int | None = None,
    train_num: int | None = None,
    val_num: int | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    batch_size: int,
    num_workers: int,
    augment: bool,
    normalization: str = "minmax",
    pca_components: int | None = None,
    pca_whiten: bool = True,
    train_repeat_factor: int = 1,
    train_spatial_jitter: int = 0,
    centered_training: bool | None = None,
    official_val_ratio: float | None = None,
    train_shuffle: bool = True,
    train_sampling: str = "shuffle",
    train_drop_last: bool = False,
    eval_drop_last: bool = False,
) -> dict[str, Any]:
    cube, labels, official_train_labels, official_test_labels = load_scene(data_root, config)
    cube, normalization_config = normalize_cube(cube, normalization)
    if pca_components is None:
        pca_info: dict[str, object] = {"applied": False}
    else:
        cube, pca_info = apply_pca_cube(cube, num_components=int(pca_components), whiten=bool(pca_whiten))
    original_height, original_width, channels = cube.shape
    if channels != int(config.wavelengths.numel()):
        wavelengths = default_wavelengths(config.dataset, channels)
        config = ClassificationSceneConfig(
            dataset=config.dataset,
            sensor=config.sensor,
            level=config.level,
            gsd_m=config.gsd_m,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note=config.metadata_note,
            class_names=config.class_names,
            cube_filename=config.cube_filename,
            cube_key=config.cube_key,
            gt_filename=config.gt_filename,
            gt_key=config.gt_key,
            img_size=config.img_size,
            overlap_size=config.overlap_size,
            pca_components=None if pca_components is None else int(pca_components),
            train_num=config.train_num,
            val_num=config.val_num,
            split_mode=config.split_mode,
            train_gt_filename=config.train_gt_filename,
            train_gt_key=config.train_gt_key,
            test_gt_filename=config.test_gt_filename,
            test_gt_key=config.test_gt_key,
            official_val_ratio=config.official_val_ratio,
        )
    resolved_img_size = int(config.img_size if img_size is None else img_size)
    resolved_overlap = int(config.overlap_size if overlap_size is None else overlap_size)
    resolved_split_mode = str(config.split_mode if split_mode is None else split_mode).strip().lower()
    if resolved_split_mode not in {"number", "official", "ratio"}:
        raise ValueError(f"Unsupported split_mode {resolved_split_mode!r}.")
    resolved_train_num = None if config.train_num is None and train_num is None else int(config.train_num if train_num is None else train_num)
    resolved_val_num = None if config.val_num is None and val_num is None else int(config.val_num if val_num is None else val_num)
    resolved_train_ratio = None if train_ratio is None else float(train_ratio)
    resolved_val_ratio = None if val_ratio is None else float(val_ratio)

    class_count = int(labels.max())
    if resolved_split_mode == "official":
        if official_train_labels is None or official_test_labels is None:
            raise ValueError(f"{config.dataset} requested official split but the train/test masks are missing.")
        train_gt_flat = official_train_labels.reshape(-1)
        train_index, val_index = split_official_training_indices(
            train_gt_flat,
            class_count,
            val_num=resolved_val_num,
            val_ratio=config.official_val_ratio if official_val_ratio is None else float(official_val_ratio),
        )
        test_index = np.where(official_test_labels.reshape(-1) != 0)[0]
        train_gt = indices_to_label_map(train_gt_flat, train_index, (original_height, original_width))
        val_gt = indices_to_label_map(train_gt_flat, val_index, (original_height, original_width))
        test_gt = official_test_labels.astype(np.int64, copy=False)
    elif resolved_split_mode == "ratio":
        if resolved_train_ratio is None:
            raise ValueError(f"{config.dataset} requires train_ratio for split_mode='ratio'.")
        gt_flat = labels.reshape(-1)
        train_index, val_index, test_index = split_ratio_labeled_indices(
            gt_flat,
            class_count,
            train_ratio=resolved_train_ratio,
            val_ratio=0.0 if resolved_val_ratio is None else resolved_val_ratio,
        )
        train_gt = indices_to_label_map(gt_flat, train_index, (original_height, original_width))
        val_gt = indices_to_label_map(gt_flat, val_index, (original_height, original_width))
        test_gt = indices_to_label_map(gt_flat, test_index, (original_height, original_width))
    else:
        if resolved_train_num is None or resolved_val_num is None:
            raise ValueError(f"{config.dataset} requires both train_num and val_num for split_mode='number'.")
        gt_flat = labels.reshape(-1)
        train_index, val_index, test_index = split_labeled_indices(
            gt_flat,
            class_count,
            train_num=resolved_train_num,
            val_num=resolved_val_num,
        )
        train_gt = indices_to_label_map(gt_flat, train_index, (original_height, original_width))
        val_gt = indices_to_label_map(gt_flat, val_index, (original_height, original_width))
        test_gt = indices_to_label_map(gt_flat, test_index, (original_height, original_width))

    use_centered_training = (int(train_index.shape[0]) <= 2000) if centered_training is None else bool(centered_training)

    if use_centered_training:
        train_dataset: Dataset = CenteredLabelCropDataset(
            cube,
            train_gt,
            train_index,
            crop_size=resolved_img_size,
            augment=augment,
            spatial_jitter=train_spatial_jitter,
            repeat_factor=train_repeat_factor,
        )
    else:
        train_tiles, train_labels, _ = tile_scene(
            cube,
            train_gt,
            img_size=resolved_img_size,
            overlap_size=resolved_overlap,
        )
        train_dataset = SceneTileDataset(train_tiles, train_labels, augment=augment)

    val_tiles, val_labels, val_starts = tile_scene(
        cube,
        val_gt,
        img_size=resolved_img_size,
        overlap_size=resolved_overlap,
    )
    test_tiles, test_labels, test_starts = tile_scene(
        cube,
        test_gt,
        img_size=resolved_img_size,
        overlap_size=resolved_overlap,
    )
    padded_height = int(test_tiles.shape[1]) if len(test_starts) == 1 else max(row for row, _ in test_starts) + resolved_img_size
    padded_width = int(test_tiles.shape[2]) if len(test_starts) == 1 else max(col for _, col in test_starts) + resolved_img_size

    sampler = None
    resolved_train_sampling = str(train_sampling).lower().strip()
    if resolved_train_sampling not in {"shuffle", "inverse", "inverse_sqrt"}:
        raise ValueError(f"Unsupported train_sampling mode {train_sampling!r}.")
    if use_centered_training and resolved_train_sampling != "shuffle":
        center_labels = train_gt.reshape(-1)[train_index.astype(np.int64)]
        counts = np.bincount(center_labels.astype(np.int64), minlength=class_count + 1).astype(np.float64, copy=False)
        counts[counts <= 0.0] = 1.0
        if resolved_train_sampling == "inverse":
            sample_weights = 1.0 / counts[center_labels.astype(np.int64)]
        else:
            sample_weights = 1.0 / np.sqrt(counts[center_labels.astype(np.int64)])
        sample_weights = np.repeat(sample_weights.astype(np.float32, copy=False), int(train_repeat_factor))
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=int(sample_weights.shape[0]),
            replacement=True,
        )
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        train_dataset,
        shuffle=bool(train_shuffle) if sampler is None else False,
        sampler=sampler,
        drop_last=bool(train_drop_last),
        **common,
    )
    val_loader = DataLoader(SceneTileDataset(val_tiles, val_labels), shuffle=False, drop_last=bool(eval_drop_last), **common)
    test_loader = DataLoader(SceneTileDataset(test_tiles, test_labels), shuffle=False, drop_last=bool(eval_drop_last), **common)

    return {
        "config": config,
        "cube": cube,
        "labels": labels,
        "train_gt": train_gt,
        "val_gt": val_gt,
        "test_gt": test_gt,
        "starts": test_starts,
        "val_starts": val_starts,
        "test_starts": test_starts,
        "height": original_height,
        "width": original_width,
        "padded_height": padded_height,
        "padded_width": padded_width,
        "channels": channels,
        "class_count": class_count,
        "img_size": resolved_img_size,
        "overlap_size": resolved_overlap,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "split_summary": {
            "dataset": config.dataset,
            "train_samples": int(train_index.shape[0]),
            "val_samples": int(val_index.shape[0]),
            "test_samples": int(test_index.shape[0]),
            "normalization": normalization_config,
            "metadata_note": config.metadata_note,
            "gsd_m": float(config.gsd_m),
            "img_size": resolved_img_size,
            "overlap_size": resolved_overlap,
            "train_num": None if resolved_train_num is None else int(resolved_train_num),
            "val_num": None if resolved_val_num is None else int(resolved_val_num),
            "train_ratio": None if resolved_train_ratio is None else float(resolved_train_ratio),
            "val_ratio": None if resolved_val_ratio is None else float(resolved_val_ratio),
            "split_mode": resolved_split_mode,
            "train_mode": "centered_crops" if use_centered_training else "scene_tiles",
            "train_shuffle": bool(train_shuffle),
            "train_sampling": resolved_train_sampling,
            "train_drop_last": bool(train_drop_last),
            "eval_drop_last": bool(eval_drop_last),
            "train_repeat_factor": int(train_repeat_factor),
            "train_spatial_jitter": int(train_spatial_jitter),
            "pca": pca_info,
            "class_names": list(config.class_names),
        },
    }


def confusion_matrix_np(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels.astype(np.int64), predictions.astype(np.int64), strict=False):
        if 0 <= label < num_classes and 0 <= pred < num_classes:
            conf[label, pred] += 1
    return conf


def kappa_from_confusion(conf: np.ndarray) -> float:
    total = float(conf.sum())
    if total <= 0:
        return 0.0
    observed = float(np.trace(conf)) / total
    expected = float((conf.sum(axis=0) * conf.sum(axis=1)).sum()) / (total * total)
    denom = 1.0 - expected
    if abs(denom) < 1.0e-12:
        return 0.0
    return (observed - expected) / denom


def metrics_from_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_count: int,
    loss: float,
    *,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    conf = confusion_matrix_np(labels, predictions, class_count)
    support = conf.sum(axis=1)
    per_class = conf.diagonal() / np.maximum(support, 1)
    class_metrics = [
        {
            "class_id": idx + 1,
            "name": class_names[idx] if idx < len(class_names) else str(idx + 1),
            "accuracy": float(per_class[idx]),
            "support": int(support[idx]),
        }
        for idx in range(class_count)
    ]
    return {
        "loss": float(loss),
        "oa": float(np.mean(labels == predictions)) if labels.size else 0.0,
        "aa": float(np.mean(per_class)) if per_class.size else 0.0,
        "kappa": float(kappa_from_confusion(conf)),
        "per_class_accuracy": class_metrics,
        "confusion_matrix": conf.tolist(),
    }
