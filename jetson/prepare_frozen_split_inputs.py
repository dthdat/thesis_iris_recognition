#!/usr/bin/env python3
"""Export a frozen iris split as model-ready tensors for Jetson evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_iris_to_polar


def soft_mask(width: int, keep_frac: float, floor: float, soft_edge: int) -> np.ndarray:
    x = np.arange(width, dtype=np.float32)
    total_keep = int(round(width * float(np.clip(keep_frac, 0.05, 1.0))))
    half_core = max(1, total_keep // 2) / 2.0
    mask = np.full(width, float(np.clip(floor, 0.0, 1.0)), dtype=np.float32)
    for center in (0.0, width / 2.0):
        dist = np.minimum(np.abs(x - center), width - np.abs(x - center))
        local = np.full(width, floor, dtype=np.float32)
        local[dist <= half_core] = 1.0
        edge = (dist > half_core) & (dist <= half_core + soft_edge)
        t = (dist[edge] - half_core) / float(max(1, soft_edge))
        local[edge] = floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(np.pi * t))
        mask = np.maximum(mask, local)
    return mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Completed run directory containing config.yaml.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", required=True, help="Output .npy path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run).resolve()
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    dataset_root = Path(args.dataset_root).resolve()
    split_dir = ROOT / config.get("split_dir", "splits")
    with (split_dir / f"{args.split}_images.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    height = int(config.get("polar_height", 64))
    width = int(config.get("polar_width", 512))

    def load(row: dict[str, str]):
        path = dataset_root / row["path"]
        polar, meta = preprocess_iris_to_polar(
            path,
            polar_h=height,
            polar_w=width,
            radial_inner=float(config.get("radial_inner", 0.10)),
            radial_outer=float(config.get("radial_outer", 0.87)),
            ido_center_search=int(config.get("ido_center_search", 4)),
            return_meta=True,
        )
        return row, path, polar, meta

    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=min(8, len(rows))) as pool:
            loaded = list(pool.map(load, rows))
    finally:
        cv2.setNumThreads(previous_threads)

    mask = soft_mask(
        width,
        float(config.get("angular_keep_frac", 0.60)),
        float(config.get("angular_mask_floor", 0.15)),
        int(config.get("angular_soft_edge", 24)),
    ).reshape(1, width)
    mean = float(config.get("norm_mean", 0.449))
    std = float(config.get("norm_std", 0.226))
    tensors, labels, paths, failures = [], [], [], []
    for row, path, polar, meta in loaded:
        if polar is None or not meta.get("ok", False):
            failures.append({"path": row["path"], "reason": meta.get("reason", "unknown")})
            continue
        value = ((polar.astype(np.float32) / 255.0 - mean) / std) * mask
        tensors.append(value.astype(np.float16, copy=False)[None, :, :])
        labels.append(int(row["raw_label"]))
        paths.append(str(path))

    array = np.stack(tensors)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, array)
    np.savez_compressed(
        output.with_name(output.stem + "_metadata.npz"),
        labels=np.asarray(labels, dtype=np.int32),
        paths=np.asarray(paths),
    )
    manifest = {
        "split": args.split,
        "run_id": config["run_id"],
        "count": len(array),
        "failed": len(failures),
        "tensor_shape": list(array.shape),
        "tensor_dtype": str(array.dtype),
        "inputs": str(output),
        "metadata": str(output.with_name(output.stem + "_metadata.npz")),
        "failures": failures,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "failures"}, indent=2))


if __name__ == "__main__":
    main()
