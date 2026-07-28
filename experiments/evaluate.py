from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import IrisDataset, load_split
from src.io_utils import ensure_dir, load_yaml, resolve_dataset_root, resolve_split_dir, write_json
from src.metrics import compact_metrics, compute_verification_metrics, save_hist_plot, save_roc_plot, save_score_distribution
from src.models import build_model
from src.train_utils import extract_embeddings, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained iris run on frozen validation/test splits.")
    parser.add_argument("--run", required=True, help="Run directory, e.g. runs/b1_arciris_nomask.")
    return parser.parse_args()


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_loader(dataset: IrisDataset, config: dict) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    kwargs = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, **kwargs)


def evaluate_split(name: str, dataset: IrisDataset, labels: list[int], model, device, config: dict, selected_threshold: float | None):
    loader = make_loader(dataset, config)
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    embeds, labels_arr = extract_embeddings(loader, model, device, use_amp=use_amp)
    n_pairs = int(config.get(f"{name}_n_pairs", config.get("test_n_pairs", 100_000)))
    metrics = compute_verification_metrics(
        embeds,
        labels_arr,
        n_pairs=n_pairs,
        seed=int(config.get("seed", 42)),
        target_far=float(config.get("target_far", 0.001)),
        impostor_multiplier=int(config.get("impostor_multiplier", 5)),
        selected_threshold=selected_threshold,
    )
    return embeds, labels_arr, metrics


def eyes_for_dataset(dataset: IrisDataset, rows: list[dict[str, str]], dataset_root: Path) -> np.ndarray:
    eye_by_abs = {}
    for row in rows:
        path = Path(row["path"])
        abs_path = path if path.is_absolute() else dataset_root / path
        eye_by_abs[str(abs_path)] = row.get("eye", "")
    return np.array([eye_by_abs.get(str(Path(path)), "") for path in dataset.image_paths])


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    config = load_yaml(run_dir / "config.yaml")
    set_seed(int(config.get("seed", 42)))
    device = get_device()
    dataset_root = resolve_dataset_root(config)
    split = load_split(dataset_root, resolve_split_dir(config))

    ckpt_path = run_dir / "best_model.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_threshold_far = float(ckpt.get("val_threshold_far", ckpt.get("val_threshold", 1.0)))
    val_ds = IrisDataset(split["val"]["paths"], split["val"]["labels"], config, augment=False, split_name="open_val")
    test_ds = IrisDataset(split["test"]["paths"], split["test"]["labels"], config, augment=False, split_name="test")

    val_embeds, val_labels, val_metrics = evaluate_split("val", val_ds, split["val"]["labels"], model, device, config, selected_threshold=val_threshold_far)
    test_embeds, test_labels, test_metrics = evaluate_split("test", test_ds, split["test"]["labels"], model, device, config, selected_threshold=val_threshold_far)

    ensure_dir(run_dir)
    np.savez_compressed(
        run_dir / "val_embeddings.npz",
        embeddings=val_embeds,
        labels=val_labels,
        paths=np.array(val_ds.image_paths),
        eyes=eyes_for_dataset(val_ds, split["val"]["rows"], dataset_root),
    )
    np.savez_compressed(
        run_dir / "test_embeddings.npz",
        embeddings=test_embeds,
        labels=test_labels,
        paths=np.array(test_ds.image_paths),
        eyes=eyes_for_dataset(test_ds, split["test"]["rows"], dataset_root),
    )
    write_json(run_dir / "val_metrics.json", compact_metrics(val_metrics))
    write_json(run_dir / "test_metrics.json", compact_metrics(test_metrics))
    save_score_distribution(run_dir / "score_distribution_val.csv", val_metrics)
    save_score_distribution(run_dir / "score_distribution_test.csv", test_metrics)
    save_roc_plot(run_dir / "roc_curve.png", test_metrics, title="Open-Set Test ROC Curve")
    save_hist_plot(run_dir / "genuine_impostor_hist.png", test_metrics, title="Open-Set Test Score Distribution")

    print("Open-set test metrics")
    print(f"  EER: {test_metrics['eer']:.3f}%")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  TAR@0.1%FAR: {test_metrics['tar_at_01far']*100:.2f}%")
    print(f"  Val-selected threshold: {val_threshold_far:.4f}")
    print(f"  FAR/FRR at val-selected threshold: {test_metrics['selected_far']*100:.3f}% / {test_metrics['selected_frr']*100:.3f}%")


if __name__ == "__main__":
    main()
