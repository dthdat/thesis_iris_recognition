from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_split
from src.io_utils import ensure_dir, load_yaml, resolve_dataset_root, resolve_split_dir, write_json
from src.metrics import compact_metrics, metrics_from_scores, sample_pair_scores, save_roc_plot, save_score_distribution
from src.preprocessing import preprocess_iris_to_polar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a non-training Daugman-style log-Gabor iris-code baseline."
    )
    parser.add_argument("--config", default="experiments/configs/b3_arciris_softmask.yaml")
    parser.add_argument("--output-dir", default="runs/classical_iriscode")
    parser.add_argument("--max-rotation", type=int, default=8)
    return parser.parse_args()


def log_gabor_iris_code(
    polar: np.ndarray,
    wavelength: float = 18.0,
    sigma_on_f: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a normalized strip into two phase bits plus an amplitude validity mask."""
    image = np.asarray(polar, dtype=np.float32) / 255.0
    image = image - image.mean(axis=1, keepdims=True)
    width = image.shape[1]
    radius = np.abs(np.fft.fftfreq(width))
    radius[0] = 1.0
    center = 1.0 / wavelength
    log_gabor = np.exp(-(np.log(radius / center) ** 2) / (2.0 * np.log(sigma_on_f) ** 2))
    log_gabor[0] = 0.0
    response = np.fft.ifft(np.fft.fft(image, axis=1) * log_gabor[None, :], axis=1)
    amplitude = np.abs(response)
    threshold = np.percentile(amplitude, 20.0, axis=1, keepdims=True)
    valid = amplitude > threshold
    code = np.stack((response.real >= 0.0, response.imag >= 0.0), axis=0)
    mask = np.broadcast_to(valid[None, :, :], code.shape).copy()
    return code, mask


def iris_code_similarity(
    code_a: np.ndarray,
    mask_a: np.ndarray,
    code_b: np.ndarray,
    mask_b: np.ndarray,
    max_rotation: int = 8,
) -> float:
    """Return one minus the best masked Hamming distance across angular shifts."""
    best = 1.0
    for shift in range(-max_rotation, max_rotation + 1):
        shifted_code = np.roll(code_b, shift, axis=-1)
        shifted_mask = np.roll(mask_b, shift, axis=-1)
        valid = mask_a & shifted_mask
        count = int(valid.sum())
        if count:
            distance = float(np.logical_xor(code_a, shifted_code)[valid].mean())
            best = min(best, distance)
    return 1.0 - best


def iris_code_similarities(
    codes: np.ndarray,
    masks: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    max_rotation: int = 8,
    chunk_size: int = 64,
) -> np.ndarray:
    """Vectorized masked-Hamming scores for the full sampled-pair protocol."""
    scores = np.empty(len(pair_a), dtype=np.float32)
    reduce_axes = (1, 2, 3)
    for start in range(0, len(pair_a), chunk_size):
        end = min(start + chunk_size, len(pair_a))
        code_a = codes[pair_a[start:end]]
        mask_a = masks[pair_a[start:end]]
        code_b = codes[pair_b[start:end]]
        mask_b = masks[pair_b[start:end]]
        best = np.ones(end - start, dtype=np.float32)
        for shift in range(-max_rotation, max_rotation + 1):
            shifted_code = np.roll(code_b, shift, axis=-1)
            valid = mask_a & np.roll(mask_b, shift, axis=-1)
            counts = np.count_nonzero(valid, axis=reduce_axes)
            mismatches = np.count_nonzero(np.logical_xor(code_a, shifted_code) & valid, axis=reduce_axes)
            distance = np.divide(
                mismatches,
                counts,
                out=np.ones_like(best),
                where=counts > 0,
            )
            best = np.minimum(best, distance)
        scores[start:end] = 1.0 - best
        if end % 10_000 == 0 or end == len(pair_a):
            print(f"Scored {end}/{len(pair_a)} pairs", flush=True)
    return scores


def encode_split(paths: list[str], labels: list[int], config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    kept_labels: list[int] = []
    for index, (path, label) in enumerate(zip(paths, labels), start=1):
        polar = preprocess_iris_to_polar(
            path,
            polar_h=int(config.get("polar_height", 64)),
            polar_w=int(config.get("polar_width", 512)),
            radial_inner=float(config.get("radial_inner", 0.10)),
            radial_outer=float(config.get("radial_outer", 0.87)),
            ido_center_search=int(config.get("ido_center_search", 4)),
        )
        if polar is not None:
            code, mask = log_gabor_iris_code(polar)
            codes.append(code)
            masks.append(mask)
            kept_labels.append(label)
        if index % 250 == 0:
            print(f"Encoded {index}/{len(paths)} images; kept {len(codes)}", flush=True)
    if not codes:
        raise RuntimeError("No images passed iris segmentation; no classical metrics were produced.")
    return np.stack(codes), np.stack(masks), np.asarray(kept_labels)


def evaluate_codes(
    codes: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    n_pairs: int,
    seed: int,
    impostor_multiplier: int,
    target_far: float,
    max_rotation: int,
    selected_threshold: float | None = None,
) -> dict:
    pair_data = sample_pair_scores(
        np.zeros((len(labels), 1), dtype=np.float32),
        labels,
        n_pairs=n_pairs,
        seed=seed,
        impostor_multiplier=impostor_multiplier,
    )
    pair_a = np.asarray(pair_data["pair_a"])
    pair_b = np.asarray(pair_data["pair_b"])
    scores = iris_code_similarities(codes, masks, pair_a, pair_b, max_rotation=max_rotation)
    metrics = metrics_from_scores(
        scores,
        np.asarray(pair_data["is_genuine"]),
        target_far=target_far,
        selected_threshold=selected_threshold,
    )
    metrics.update(
        pair_a=pair_a,
        pair_b=pair_b,
        n_genuine_pairs=int(pair_data["n_genuine_pairs"]),
        n_impostor_pairs=int(pair_data["n_impostor_pairs"]),
    )
    return metrics


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_root = resolve_dataset_root(config)
    split = load_split(dataset_root, resolve_split_dir(config))
    output_dir = ensure_dir(args.output_dir)
    common = {
        "seed": int(config.get("seed", 42)),
        "impostor_multiplier": int(config.get("impostor_multiplier", 5)),
        "target_far": float(config.get("target_far", 0.001)),
        "max_rotation": args.max_rotation,
    }

    val_codes, val_masks, val_labels = encode_split(split["val"]["paths"], split["val"]["labels"], config)
    val_metrics = evaluate_codes(
        val_codes, val_masks, val_labels, n_pairs=int(config.get("val_n_pairs", 100_000)), **common
    )
    selected_threshold = float(val_metrics["threshold_far"])
    test_codes, test_masks, test_labels = encode_split(split["test"]["paths"], split["test"]["labels"], config)
    test_metrics = evaluate_codes(
        test_codes,
        test_masks,
        test_labels,
        n_pairs=int(config.get("test_n_pairs", 200_000)),
        selected_threshold=selected_threshold,
        **common,
    )
    metadata = {
        "method": "Daugman-style log-Gabor iris code (not an exact OSIRIS reproduction)",
        "training": False,
        "config_source": str(args.config),
        "max_rotation_columns": int(args.max_rotation),
        "validation_images": len(val_codes),
        "test_images": len(test_codes),
    }
    write_json(output_dir / "method.json", metadata)
    write_json(output_dir / "val_metrics.json", compact_metrics(val_metrics))
    write_json(output_dir / "test_metrics.json", compact_metrics(test_metrics))
    save_score_distribution(output_dir / "score_distribution_val.csv", val_metrics)
    save_score_distribution(output_dir / "score_distribution_test.csv", test_metrics)
    save_roc_plot(output_dir / "roc_curve.png", test_metrics, title="Classical Iris-Code Test ROC")
    print(f"Test EER: {test_metrics['eer']:.4f}%")
    print(f"Test AUC: {test_metrics['auc']:.6f}")
    print(f"Test TAR@0.1%FAR: {test_metrics['tar_at_01far'] * 100:.4f}%")


if __name__ == "__main__":
    main()
