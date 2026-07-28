from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .io_utils import ensure_dir, read_csv_rows, write_csv_rows
from .masks import make_mask_bank, make_soft_angular_mask
from .preprocessing import preprocess_iris_to_polar


EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
REQUIRED_SPLIT_FILES = [
    "train_subjects.csv",
    "val_subjects.csv",
    "test_subjects.csv",
    "train_images.csv",
    "val_images.csv",
    "test_images.csv",
]


def discover_casia(root: str | os.PathLike[str], min_samples: int = 3) -> tuple[list[str], list[int], list[str], dict[int, str]]:
    root = str(root)
    all_paths: list[str] = []
    all_labels: list[int] = []
    label_names: list[str] = []
    subject_map: dict[int, str] = {}
    label_idx = 0

    for subj in sorted(os.listdir(root)):
        subj_dir = os.path.join(root, subj)
        if not os.path.isdir(subj_dir):
            continue
        for eye in ["L", "R"]:
            eye_dir = os.path.join(subj_dir, eye)
            if not os.path.isdir(eye_dir):
                continue
            images = [
                os.path.join(eye_dir, f)
                for f in sorted(os.listdir(eye_dir))
                if os.path.splitext(f)[1].lower() in EXTS
            ]
            if len(images) >= min_samples:
                all_paths.extend(images)
                all_labels.extend([label_idx] * len(images))
                label_names.append(f"{subj}_{eye}")
                subject_map[label_idx] = subj
                label_idx += 1

    return all_paths, all_labels, label_names, subject_map


def split_files_exist(split_dir: str | os.PathLike[str]) -> bool:
    split_dir = Path(split_dir)
    return all((split_dir / name).is_file() for name in REQUIRED_SPLIT_FILES)


def _rel_to_root(path: str, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _subject_rows(subjects: set[str]) -> list[dict[str, str]]:
    return [{"subject": subj} for subj in sorted(subjects)]


def _image_rows(
    root: Path,
    paths: list[str],
    raw_labels: list[int],
    subject_map: dict[int, str],
    label_names: list[str],
    train_class_remap: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, label in zip(paths, raw_labels):
        label_name = label_names[label]
        eye = label_name.rsplit("_", 1)[-1]
        row: dict[str, Any] = {
            "path": _rel_to_root(path, root),
            "subject": subject_map[label],
            "eye": eye,
            "raw_label": label,
            "label_name": label_name,
        }
        if train_class_remap is not None:
            row["train_label"] = train_class_remap[label]
        rows.append(row)
    return rows


def create_subject_exclusive_split(
    dataset_root: str | os.PathLike[str],
    split_dir: str | os.PathLike[str],
    seed: int = 42,
    train_subject_frac: float = 0.70,
    val_subject_frac: float = 0.10,
    min_samples: int = 3,
) -> dict[str, Any]:
    root = Path(dataset_root)
    all_paths, all_labels, label_names, subject_map = discover_casia(root, min_samples=min_samples)
    if not all_paths:
        raise RuntimeError(f"No images found under dataset_root={root}")

    unique_subjects = sorted({subject_map[i] for i in range(len(label_names))})
    rng = random.Random(seed)
    rng.shuffle(unique_subjects)
    n_total = len(unique_subjects)
    n_train = max(1, int(train_subject_frac * n_total))
    n_val = max(1, int(val_subject_frac * n_total))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)

    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train : n_train + n_val])
    test_subjects = set(unique_subjects[n_train + n_val :])
    if not test_subjects:
        raise RuntimeError("Subject split produced an empty test set. Check dataset size and split fractions.")

    train_class_set = {i for i in range(len(label_names)) if subject_map[i] in train_subjects}
    train_old_class_ids = sorted(train_class_set)
    train_class_remap = {old: new for new, old in enumerate(train_old_class_ids)}

    train_pairs = [(p, lbl) for p, lbl in zip(all_paths, all_labels) if subject_map[lbl] in train_subjects]
    train_paths_raw = [p for p, _ in train_pairs]
    train_labels_raw = [lbl for _, lbl in train_pairs]
    val_paths_raw = [p for p, lbl in zip(all_paths, all_labels) if subject_map[lbl] in val_subjects]
    val_labels_raw = [lbl for p, lbl in zip(all_paths, all_labels) if subject_map[lbl] in val_subjects]
    test_paths_raw = [p for p, lbl in zip(all_paths, all_labels) if subject_map[lbl] in test_subjects]
    test_labels_raw = [lbl for p, lbl in zip(all_paths, all_labels) if subject_map[lbl] in test_subjects]

    split_dir = ensure_dir(split_dir)
    write_csv_rows(split_dir / "train_subjects.csv", _subject_rows(train_subjects), ["subject"])
    write_csv_rows(split_dir / "val_subjects.csv", _subject_rows(val_subjects), ["subject"])
    write_csv_rows(split_dir / "test_subjects.csv", _subject_rows(test_subjects), ["subject"])
    image_fields = ["path", "subject", "eye", "raw_label", "label_name", "train_label"]
    write_csv_rows(
        split_dir / "train_images.csv",
        _image_rows(root, train_paths_raw, train_labels_raw, subject_map, label_names, train_class_remap),
        image_fields,
    )
    eval_fields = ["path", "subject", "eye", "raw_label", "label_name"]
    write_csv_rows(split_dir / "val_images.csv", _image_rows(root, val_paths_raw, val_labels_raw, subject_map, label_names), eval_fields)
    write_csv_rows(split_dir / "test_images.csv", _image_rows(root, test_paths_raw, test_labels_raw, subject_map, label_names), eval_fields)

    return load_split(dataset_root=root, split_dir=split_dir)


def load_or_create_split(
    dataset_root: str | os.PathLike[str],
    split_dir: str | os.PathLike[str],
    seed: int = 42,
    train_subject_frac: float = 0.70,
    val_subject_frac: float = 0.10,
    min_samples: int = 3,
    regenerate: bool = False,
) -> dict[str, Any]:
    if split_files_exist(split_dir) and not regenerate:
        return load_split(dataset_root, split_dir)
    return create_subject_exclusive_split(dataset_root, split_dir, seed, train_subject_frac, val_subject_frac, min_samples)


def load_split(dataset_root: str | os.PathLike[str], split_dir: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(dataset_root)
    split_dir = Path(split_dir)
    missing = [name for name in REQUIRED_SPLIT_FILES if not (split_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing split files in {split_dir}: {missing}")

    def load_images(name: str, train: bool) -> dict[str, Any]:
        rows = read_csv_rows(split_dir / name)
        paths = [str(root / row["path"]) if not Path(row["path"]).is_absolute() else row["path"] for row in rows]
        labels = [int(row["train_label"] if train else row["raw_label"]) for row in rows]
        raw_labels = [int(row["raw_label"]) for row in rows]
        label_names = [row["label_name"] for row in rows]
        subjects = [row["subject"] for row in rows]
        eyes = [row.get("eye", "") for row in rows]
        return {
            "rows": rows,
            "paths": paths,
            "labels": labels,
            "raw_labels": raw_labels,
            "label_names": label_names,
            "subjects": subjects,
            "eyes": eyes,
        }

    train = load_images("train_images.csv", train=True)
    val = load_images("val_images.csv", train=False)
    test = load_images("test_images.csv", train=False)
    train_label_name_by_id: dict[int, str] = {}
    for row in train["rows"]:
        train_label_name_by_id[int(row["train_label"])] = row["label_name"]
    train_label_names = [train_label_name_by_id[i] for i in sorted(train_label_name_by_id)]
    num_classes = len({int(row["train_label"]) for row in train["rows"]})
    return {
        "dataset_root": str(root),
        "split_dir": str(split_dir),
        "train": train,
        "val": val,
        "test": test,
        "num_classes": num_classes,
        "train_label_names": train_label_names,
        "train_subjects": [row["subject"] for row in read_csv_rows(split_dir / "train_subjects.csv")],
        "val_subjects": [row["subject"] for row in read_csv_rows(split_dir / "val_subjects.csv")],
        "test_subjects": [row["subject"] for row in read_csv_rows(split_dir / "test_subjects.csv")],
    }


class IrisDataset(Dataset):
    """Preload polar iris strips into RAM and filter failed segmentations."""

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        config: dict[str, Any],
        augment: bool = False,
        split_name: str = "split",
        skip_invalid: bool = True,
    ):
        self.config = config
        self.polar_h = int(config.get("polar_height", 64))
        self.polar_w = int(config.get("polar_width", 512))
        self.radial_inner = float(config.get("radial_inner", 0.10))
        self.radial_outer = float(config.get("radial_outer", 0.87))
        self.ido_center_search = int(config.get("ido_center_search", 4))
        self.augment = augment
        self.mean = float(config.get("norm_mean", 0.449))
        self.std = float(config.get("norm_std", 0.226))
        self.split_name = split_name

        self.use_angular_mask = bool(config.get("use_angular_mask", True))
        self.angular_mask = make_soft_angular_mask(
            polar_w=self.polar_w,
            keep_frac=float(config.get("angular_keep_frac", 0.60)),
            floor=float(config.get("angular_mask_floor", 0.15)),
            soft_edge=int(config.get("angular_soft_edge", 24)),
            enabled=self.use_angular_mask,
        ).unsqueeze(0).unsqueeze(0)
        self.mask_bank = None
        if self.augment and self.use_angular_mask and bool(config.get("randomize_angular_mask", False)):
            self.mask_bank = make_mask_bank(
                polar_w=self.polar_w,
                bank_size=int(config.get("mask_bank_size", 32)),
                keep_frac_min=float(config.get("angular_keep_frac_min", 0.50)),
                keep_frac_max=float(config.get("angular_keep_frac_max", 0.75)),
                floor=float(config.get("angular_mask_floor", 0.15)),
                soft_edge=int(config.get("angular_soft_edge", 24)),
            )

        self.failed: list[dict[str, Any]] = []
        self.meta: list[dict[str, Any]] = []
        n_workers = int(config.get("preload_workers", 0)) or min(8, (os.cpu_count() or 2) * 2)
        print(f"Preloading {len(image_paths)} images for {split_name} (polar {self.polar_h}x{self.polar_w})...", flush=True)
        prev_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)

        def load_one(path: str) -> tuple[np.ndarray | None, dict[str, Any]]:
            return preprocess_iris_to_polar(
                path,
                polar_h=self.polar_h,
                polar_w=self.polar_w,
                radial_inner=self.radial_inner,
                radial_outer=self.radial_outer,
                ido_center_search=self.ido_center_search,
                return_meta=True,
            )

        try:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                loaded = list(ex.map(load_one, image_paths))
        except Exception as exc:
            print(f"  parallel preload failed ({exc}); falling back to sequential", flush=True)
            loaded = [load_one(p) for p in image_paths]
        finally:
            cv2.setNumThreads(prev_threads)

        self.cache: list[np.ndarray] = []
        self.labels: list[int] = []
        self.image_paths: list[str] = []
        method_counts: dict[str, int] = {}
        for path, label, (polar, meta) in zip(image_paths, labels, loaded):
            if polar is None or not meta.get("ok", False):
                self.failed.append(meta)
                if not skip_invalid:
                    self.cache.append(np.zeros((self.polar_h, self.polar_w), dtype=np.uint8))
                    self.labels.append(label)
                    self.image_paths.append(path)
                continue
            self.cache.append(polar)
            self.labels.append(int(label))
            self.image_paths.append(path)
            self.meta.append(meta)
            method = str(meta.get("method", "unknown"))
            method_counts[method] = method_counts.get(method, 0) + 1

        if not self.cache:
            raise RuntimeError(
                f"No valid iris segmentations for {split_name}. Check dataset_root and detection parameters."
            )
        n_total = len(image_paths)
        n_ok = len(self.cache)
        n_fail = n_total - n_ok
        self.segmentation_stats = {
            "split": split_name,
            "total": n_total,
            "success": n_ok,
            "fail": n_fail,
            "fail_rate": n_fail / max(1, n_total),
            "methods": method_counts,
        }
        print(
            f"Preload complete ({n_workers} workers) - kept {n_ok}/{n_total}; "
            f"failed {n_fail} ({100*n_fail/max(1,n_total):.2f}%). Methods: {method_counts}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.cache[idx].copy()
        label = self.labels[idx]
        if self.augment:
            img = self._augment(img)
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        tensor = (tensor - self.mean) / self.std
        if self.use_angular_mask:
            if self.augment and self.mask_bank is not None:
                mask_idx = np.random.randint(0, self.mask_bank.shape[0])
                tensor = tensor * self.mask_bank[mask_idx]
            else:
                tensor = tensor * self.angular_mask
        return tensor, label

    def _augment(self, img: np.ndarray) -> np.ndarray:
        max_shift = max(1, int(self.polar_w * float(self.config.get("aug_roll_frac", 1 / 16))))
        shift = np.random.randint(-max_shift, max_shift + 1)
        img = np.roll(img, shift, axis=1)

        if np.random.rand() < 0.7:
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-10, 10)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if np.random.rand() < float(self.config.get("aug_erase_prob", 0.15)):
            h, w = img.shape
            ew = np.random.randint(max(2, w // 32), max(3, w // 10))
            eh = np.random.randint(max(2, h // 10), max(3, h // 4))
            x0 = np.random.randint(0, max(1, w - ew))
            y0 = np.random.randint(0, max(1, h - eh))
            img[y0 : y0 + eh, x0 : x0 + ew] = int(np.median(img))
        return img
