from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def write_yaml(path: str | os.PathLike[str], data: Mapping[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False)


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=True)
        f.write("\n")


def to_jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is expected for this project.
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_csv_rows(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        rows = list(rows)
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv_rows(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_csv_row(path: str | os.PathLike[str], row: Mapping[str, Any], fieldnames: list[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def copytree_clean(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    dst_path = Path(dst)
    if dst_path.exists():
        shutil.rmtree(dst_path)
    shutil.copytree(src, dst)


def resolve_dataset_root(config: Mapping[str, Any]) -> Path:
    env_root = os.environ.get("IRIS_DATASET_ROOT")
    root = env_root or str(config.get("dataset_root", "")).strip()
    if not root:
        raise FileNotFoundError(
            "dataset_root is not set. Set IRIS_DATASET_ROOT or edit the YAML config."
        )
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {path}\n"
            "Expected CASIA-Iris-Thousand layout with subject folders containing L/ and R/ eye folders. "
            "Set IRIS_DATASET_ROOT=/path/to/CASIA-Iris-Thousand or update dataset_root in the config."
        )
    return path


def resolve_run_dir(config: Mapping[str, Any]) -> Path:
    output_root = Path(os.environ.get("IRIS_OUTPUT_DIR", str(config.get("output_dir", "runs"))))
    return output_root / str(config["run_id"])


def resolve_split_dir(config: Mapping[str, Any]) -> Path:
    return Path(os.environ.get("IRIS_SPLIT_DIR", str(config.get("split_dir", "splits"))))
