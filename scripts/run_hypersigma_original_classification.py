from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYPERSIGMA_ROOT = REPO_ROOT.parent / "HyperSIGMA-main"


@contextmanager
def allow_argparse_namespace_unpickle():
    try:
        with torch.serialization.safe_globals([argparse.Namespace]):
            yield
    except AttributeError:
        yield


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_indian_pines(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    cube_path = data_root / "Indian_pines_corrected.mat"
    gt_path = data_root / "Indian_pines_gt.mat"
    if not cube_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Missing Indian Pines files under {data_root}.")
    cube = sio.loadmat(cube_path)["data"].astype(np.float32)
    gt = sio.loadmat(gt_path)["groundT"].astype(np.float32)
    return cube, gt


def apply_pca_original(x: np.ndarray, num_components: int) -> tuple[np.ndarray, PCA]:
    flat = np.reshape(x, (-1, x.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    projected = pca.fit_transform(flat)
    projected = np.reshape(projected, (x.shape[0], x.shape[1], num_components))
    return projected.astype(np.float32), pca


def pad_with_zeros(x: np.ndarray, margin: int) -> np.ndarray:
    padded = np.zeros((x.shape[0] + 2 * margin, x.shape[1] + 2 * margin, x.shape[2]), dtype=np.float32)
    padded[margin : x.shape[0] + margin, margin : x.shape[1] + margin, :] = x
    return padded


def create_patches_original(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window_size: int,
    remove_zero_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    margin = int((window_size - 1) / 2)
    padded = pad_with_zeros(x, margin=margin)
    patches = np.zeros((x.shape[0] * x.shape[1], window_size, window_size, x.shape[2]), dtype=np.float32)
    labels = np.zeros((x.shape[0] * x.shape[1],), dtype=np.int64)
    patch_index = 0
    for row in range(margin, padded.shape[0] - margin):
        for col in range(margin, padded.shape[1] - margin):
            patch = padded[row - margin : row + margin + 1, col - margin : col + margin + 1]
            patches[patch_index, :, :, :] = patch
            labels[patch_index] = int(y[row - margin, col - margin])
            patch_index += 1
    if remove_zero_labels:
        keep = labels > 0
        patches = patches[keep]
        labels = labels[keep]
    return patches, labels


def split_data_original(
    gt_flat: np.ndarray,
    class_num: int,
    train_ratio: float,
    val_ratio: float,
    train_num: int,
    val_num: int,
    samples_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_index: list[np.ndarray] = []
    val_index: list[np.ndarray] = []
    test_index: list[np.ndarray] = []
    if samples_type == "ratio":
        for class_idx in range(class_num):
            idx = np.where(gt_flat == class_idx + 1)[-1]
            samples_count = len(idx)
            current_train_num = np.ceil(samples_count * train_ratio).astype("int32")
            current_val_num = np.ceil(samples_count * val_ratio).astype("int32")
            np.random.shuffle(idx)
            train_index.append(idx[:current_train_num])
            val_index.append(idx[current_train_num : current_train_num + current_val_num])
            test_index.append(idx[current_train_num + current_val_num :])
    else:
        sample_num = train_num
        for class_idx in range(class_num):
            idx = np.where(gt_flat == class_idx + 1)[-1]
            samples_count = len(idx)
            max_index = np.max(samples_count) + 1
            np.random.shuffle(idx)
            if sample_num > max_index:
                sample_num = 10
            else:
                sample_num = train_num
            train_index.append(idx[:sample_num])
            val_index.append(idx[sample_num : sample_num + val_num])
            test_index.append(idx[sample_num + val_num :])
    return np.concatenate(train_index, axis=0), np.concatenate(val_index, axis=0), np.concatenate(test_index, axis=0)


class PatchArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x_data = torch.FloatTensor(x)
        self.y_data = torch.LongTensor(y.reshape(-1))

    def __len__(self) -> int:
        return int(self.x_data.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[index], self.y_data[index]


def load_hypersigma_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    with allow_argparse_namespace_unpickle():
        payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint payload in {path}: {type(payload)!r}")


def load_weights_like_original_notebook(model: torch.nn.Module, spatial_checkpoint: Path, spectral_checkpoint: Path) -> dict[str, int]:
    model_params = model.state_dict()

    spat_net = load_hypersigma_checkpoint(spatial_checkpoint)
    spat_net = dict(spat_net)
    for key in list(spat_net.keys()):
        if "patch_embed.proj" in key or "spat_map" in key or "spat_output_maps" in key or "pos_embed" in key:
            del spat_net[key]
    spat_weights = {f"spat_encoder.{key}": value for key, value in spat_net.items()}

    spec_net = load_hypersigma_checkpoint(spectral_checkpoint)
    spec_net = dict(spec_net)
    for key in list(spec_net.keys()):
        if "patch_embed.proj" in key or "spat_map" in key or "fpn1.0.weight" in key:
            del spec_net[key]
    spec_weights = {f"spec_encoder.{key}": value for key, value in spec_net.items()}
    for key in list(spec_weights.keys()):
        if "spec_encoder.patch_embed" in key:
            del spec_weights[key]

    merged_params = {**spat_weights, **spec_weights}
    same_params = {key: value for key, value in merged_params.items() if key in model_params.keys()}
    model_params.update(same_params)
    model.load_state_dict(model_params)
    return {
        "spatial_loaded": int(sum(1 for key in spat_weights if key in same_params)),
        "spectral_loaded": int(sum(1 for key in spec_weights if key in same_params)),
        "total_loaded": int(len(same_params)),
    }


def confusion_matrix_label_pred(labels: np.ndarray, predictions: np.ndarray, class_num: int) -> np.ndarray:
    conf = np.zeros((class_num + 1, class_num + 1), dtype=np.int64)
    for label, pred in zip(labels.astype(np.int64), predictions.astype(np.int64), strict=False):
        if 0 <= label <= class_num and 0 <= pred <= class_num:
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


def standard_metrics(labels: np.ndarray, predictions: np.ndarray, class_num: int, loss: float) -> dict[str, Any]:
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    conf = confusion_matrix_label_pred(labels, predictions, class_num)
    support = conf.sum(axis=1)[1:]
    per_class = conf.diagonal()[1:] / np.maximum(support, 1)
    return {
        "loss": float(loss),
        "oa": float(np.mean(labels == predictions)),
        "aa": float(np.mean(per_class)),
        "kappa": float(kappa_from_confusion(conf)),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": conf.tolist(),
    }


def original_notebook_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    # The notebook passes (prediction, label) to sklearn. OA is symmetric, but
    # the confusion matrix orientation affects each_acc/AA, so keep it explicit.
    conf = metrics.confusion_matrix(predictions, labels)
    diag = np.diag(conf)
    row_sum = np.sum(conf, axis=1)
    each_acc = np.nan_to_num(diag / np.maximum(row_sum, 1))
    return {
        "oa": float(metrics.accuracy_score(predictions, labels)),
        "kappa": float(metrics.cohen_kappa_score(predictions, labels)),
        "each_acc_pred_label_orientation": each_acc.tolist(),
        "average_acc_pred_label_orientation": float(np.mean(each_acc)),
        "confusion_matrix_pred_label_orientation": conf.tolist(),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_num: int,
    criterion: nn.Module,
) -> dict[str, Any]:
    model.eval()
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).reshape(-1)
        logits = model(x)
        loss = criterion(logits, y)
        predictions = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * int(y.numel())
        total_count += int(y.numel())
        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    pred_np = np.concatenate(all_predictions)
    label_np = np.concatenate(all_labels)
    standard = standard_metrics(label_np, pred_np, class_num, total_loss / max(total_count, 1))
    original = original_notebook_metrics(label_np, pred_np)
    return {"standard": standard, "original_notebook": original}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HyperSIGMA Indian Pines classification with the original notebook protocol.")
    parser.add_argument("--hypersigma-root", type=Path, default=DEFAULT_HYPERSIGMA_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "hypersigma_original_indian_pines")
    parser.add_argument("--spatial-checkpoint", type=Path, default=None)
    parser.add_argument("--spectral-checkpoint", type=Path, default=None)
    parser.add_argument("--img-size", type=int, default=33)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--train-num", type=int, default=10)
    parser.add_argument("--val-num", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--split-type", choices=("number", "ratio"), default="number")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay-rate", type=float, default=0.9)
    parser.add_argument("--scheduler-epochs", type=int, default=300)
    parser.add_argument("--model-size", choices=("base", "large", "huge"), default="base")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hypersigma_image_cls_root = args.hypersigma_root / "ImageClassification"
    if str(hypersigma_image_cls_root) not in sys.path:
        sys.path.insert(0, str(hypersigma_image_cls_root))
    from mmcv_custom import custom_layer_decay_optimizer_constructor, layer_decay_optimizer_constructor_vit  # noqa: F401,E402
    from mmengine.optim import build_optim_wrapper  # noqa: E402
    from model import ss_fusion_cls  # noqa: E402

    data_root = args.data_root or (hypersigma_image_cls_root / "data")
    spatial_checkpoint = args.spatial_checkpoint or (
        args.hypersigma_root / "Pretrain" / "spat-vit-base-ultra-checkpoint-1599.pth"
    )
    spectral_checkpoint = args.spectral_checkpoint or (
        args.hypersigma_root / "Pretrain" / "spec-vit-base-ultra-checkpoint-1599.pth"
    )

    data, data_gt = load_indian_pines(data_root)
    height, width, bands = data.shape
    gt_flat = np.reshape(data_gt, [-1])
    class_num = int(np.max(data_gt))
    train_index, val_index, test_index = split_data_original(
        gt_flat,
        class_num,
        args.train_ratio,
        args.train_ratio,
        args.train_num,
        args.val_num,
        args.split_type,
    )
    data, pca = apply_pca_original(data, num_components=args.pca_components)
    data_all, data_label_all = create_patches_original(
        data,
        data_gt,
        window_size=args.img_size,
        remove_zero_labels=False,
    )

    train_index = train_index.astype(int).reshape(-1, 1)
    val_index = val_index.astype(int).reshape(-1, 1)
    test_index = test_index.astype(int).reshape(-1, 1)

    train_label = data_label_all[train_index].reshape(-1)
    val_label = data_label_all[val_index].reshape(-1)
    test_label = data_label_all[test_index].reshape(-1)

    train = data_all[train_index, :, :, :].reshape(-1, args.img_size, args.img_size, args.pca_components)
    val = data_all[val_index, :, :, :].reshape(-1, args.img_size, args.img_size, args.pca_components)
    test = data_all[test_index, :, :, :].reshape(-1, args.img_size, args.img_size, args.pca_components)

    train = train.transpose(0, 3, 1, 2)
    val = val.transpose(0, 3, 1, 2)
    test = test.transpose(0, 3, 1, 2)

    train_loader = DataLoader(PatchArrayDataset(train, train_label), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(PatchArrayDataset(val, val_label), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(PatchArrayDataset(test, test_label), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ss_fusion_cls.SSFusionFramework(
        img_size=args.img_size,
        in_channels=args.pca_components,
        patch_size=args.patch_size,
        classes=class_num + 1,
        model_size=args.model_size,
    )
    loaded_summary = load_weights_like_original_notebook(model, spatial_checkpoint, spectral_checkpoint)

    optim_wrapper = dict(
        optimizer=dict(type="AdamW", lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay),
        constructor="LayerDecayOptimizerConstructor_ViT",
        paramwise_cfg=dict(num_layers=12, layer_decay_rate=args.layer_decay_rate),
    )
    optimizer_wrapper = build_optim_wrapper(model, optim_wrapper)
    optimizer = optimizer_wrapper.optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.scheduler_epochs, eta_min=0, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    config = vars(args).copy()
    config.update(
        {
            "height": int(height),
            "width": int(width),
            "source_bands": int(bands),
            "class_num": int(class_num),
            "train_samples": int(train_label.shape[0]),
            "val_samples": int(val_label.shape[0]),
            "test_samples": int(test_label.shape[0]),
            "spatial_checkpoint": str(spatial_checkpoint),
            "spectral_checkpoint": str(spectral_checkpoint),
            "loaded_summary": loaded_summary,
            "protocol_note": "Matches HyperSIGMA demo cls hypersigma.ipynb: PCA30, 33x33 patches, 10 train and 5 val samples per class, original manual checkpoint merge, save by train loss every 10 epochs.",
        }
    )
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=True, default=str)

    best_loss = math.inf
    best_epoch = -1
    best_path = args.output_dir / "model_original_best_train_loss_every10.pth"
    latest_path = args.output_dir / "model_original_latest.pth"
    log_path = args.output_dir / "classification_log.jsonl"

    for epoch in range(args.epochs + 1):
        model.train()
        correct = 0
        total = 0
        raw_loss_sum = 0.0
        true_loss_sum = 0.0
        true_count = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).reshape(-1)
            logits = model(x)
            train_loss = criterion(logits, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            raw_loss_sum += float(train_loss.item())
            true_loss_sum += float(train_loss.item()) * int(y.numel())
            true_count += int(y.numel())

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        scheduler.step()

        notebook_epoch_loss = raw_loss_sum / max(len(train_loader.dataset), 1)
        true_train_loss = true_loss_sum / max(true_count, 1)
        train_acc = correct / max(total, 1)
        record = {
            "epoch": int(epoch),
            "notebook_epoch_loss": float(notebook_epoch_loss),
            "true_train_loss": float(true_train_loss),
            "train_acc": float(train_acc),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if epoch % 10 == 0:
            val_metrics = evaluate(model, val_loader, device, class_num, criterion)
            record["val_standard"] = {
                "loss": val_metrics["standard"]["loss"],
                "oa": val_metrics["standard"]["oa"],
                "aa": val_metrics["standard"]["aa"],
                "kappa": val_metrics["standard"]["kappa"],
            }
            if notebook_epoch_loss < best_loss:
                best_loss = notebook_epoch_loss
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
                torch.save(optimizer.state_dict(), args.output_dir / "optimizer_original_best_train_loss_every10.pth")
                record["saved_best_train_loss_every10"] = True
            print(json.dumps(record, ensure_ascii=True), flush=True)

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    torch.save(model.state_dict(), latest_path)

    results: dict[str, Any] = {}
    for name, path in (("best_train_loss_every10", best_path), ("latest", latest_path)):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        test_metrics = evaluate(model, test_loader, device, class_num, criterion)
        results[name] = {
            "checkpoint": str(path),
            "epoch": int(best_epoch if name == "best_train_loss_every10" else args.epochs),
            "test": test_metrics,
        }
    print(json.dumps(results, ensure_ascii=True), flush=True)
    with (args.output_dir / "test_metrics_original_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
