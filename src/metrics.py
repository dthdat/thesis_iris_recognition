from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from .io_utils import ensure_dir, write_csv_rows


def threshold_for_far(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, target_far: float = 0.001) -> tuple[float, float, float]:
    valid = np.where((fpr <= target_far) & np.isfinite(thresholds))[0]
    if len(valid) == 0:
        finite = np.where(np.isfinite(thresholds))[0]
        idx = int(finite[0]) if len(finite) else 0
    else:
        idx = int(valid[np.argmax(tpr[valid])])
    return float(thresholds[idx]), float(tpr[idx]), float(fpr[idx])


def chunked_pair_scores(
    embeds: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    chunk: int = 8192,
) -> np.ndarray:
    pair_a = np.asarray(pair_a)
    pair_b = np.asarray(pair_b)
    if len(pair_a) != len(pair_b):
        raise ValueError("pair_a and pair_b must have the same length")
    chunk = max(1, int(chunk))
    scores = np.empty(len(pair_a), dtype=np.float32)
    for start in range(0, len(pair_a), chunk):
        end = min(start + chunk, len(pair_a))
        scores[start:end] = (
            embeds[pair_a[start:end]] * embeds[pair_b[start:end]]
        ).sum(axis=1).astype(np.float32, copy=False)
    return scores


def sample_pair_scores(
    embeds: np.ndarray,
    labels: np.ndarray,
    n_pairs: int | None = 100_000,
    seed: int = 0,
    impostor_multiplier: int = 5,
) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    embeds = np.asarray(embeds)
    n_items = len(embeds)
    idx_all = np.arange(n_items)

    genuine_pairs: list[tuple[int, int]] = []
    for label in np.unique(labels):
        idxs = idx_all[labels == label]
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                genuine_pairs.append((int(idxs[i]), int(idxs[j])))

    if n_pairs is not None:
        max_genuine = max(1, int(n_pairs) // (1 + int(impostor_multiplier)))
        if len(genuine_pairs) > max_genuine:
            keep = rng.choice(len(genuine_pairs), size=max_genuine, replace=False)
            genuine_pairs = [genuine_pairs[int(i)] for i in keep]

    n_genuine = len(genuine_pairs)
    if n_pairs is None:
        target_impostors = max(n_genuine, n_genuine * int(impostor_multiplier))
    else:
        target_impostors = max(1, min(int(n_pairs) - n_genuine, n_genuine * int(impostor_multiplier)))

    impostor_pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(impostor_pairs) < target_impostors and attempts < 50:
        need = max(1024, (target_impostors - len(impostor_pairs)) * 2)
        a = rng.integers(0, n_items, size=need)
        b = rng.integers(0, n_items, size=need)
        mask = labels[a] != labels[b]
        impostor_pairs.extend((int(x), int(y)) for x, y in zip(a[mask], b[mask]))
        attempts += 1
    impostor_pairs = impostor_pairs[:target_impostors]

    all_pairs = genuine_pairs + impostor_pairs
    is_genuine = np.array([1] * len(genuine_pairs) + [0] * len(impostor_pairs), dtype=np.int32)
    if not all_pairs:
        return {
            "scores": np.array([], dtype=np.float32),
            "is_genuine": is_genuine,
            "pair_a": np.array([], dtype=np.int32),
            "pair_b": np.array([], dtype=np.int32),
            "n_genuine_pairs": 0,
            "n_impostor_pairs": 0,
        }

    pair_a = np.array([p[0] for p in all_pairs], dtype=np.int32)
    pair_b = np.array([p[1] for p in all_pairs], dtype=np.int32)
    scores = chunked_pair_scores(embeds, pair_a, pair_b)
    return {
        "scores": scores,
        "is_genuine": is_genuine,
        "pair_a": pair_a,
        "pair_b": pair_b,
        "n_genuine_pairs": len(genuine_pairs),
        "n_impostor_pairs": len(impostor_pairs),
    }


def _score_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    if len(values) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def rates_at_threshold(scores: np.ndarray, is_genuine: np.ndarray, threshold: float) -> dict[str, float]:
    scores = np.asarray(scores)
    y = np.asarray(is_genuine).astype(int)
    pred = (scores >= threshold).astype(int)
    impostor = y == 0
    genuine = y == 1
    far = float(((pred == 1) & impostor).sum() / max(1, impostor.sum()))
    frr = float(((pred == 0) & genuine).sum() / max(1, genuine.sum()))
    tar = 1.0 - frr
    return {"threshold": float(threshold), "far": far, "frr": frr, "tar": tar}


def metrics_from_scores(
    scores: np.ndarray,
    is_genuine: np.ndarray,
    target_far: float = 0.001,
    selected_threshold: float | None = None,
) -> dict[str, Any]:
    scores = np.asarray(scores)
    is_genuine = np.asarray(is_genuine).astype(int)
    if len(scores) == 0 or len(np.unique(is_genuine)) < 2:
        return {
            "eer": 50.0,
            "auc": 0.5,
            "tar_at_01far": 0.0,
            "threshold_eer": 1.0,
            "threshold_far": 1.0,
            "actual_far": 0.0,
            "selected_threshold": 1.0 if selected_threshold is None else float(selected_threshold),
            "selected_far": 0.0,
            "selected_frr": 1.0,
            "selected_tar": 0.0,
            "fpr": np.array([0.0, 1.0]),
            "tpr": np.array([0.0, 1.0]),
            "scores": scores,
            "is_genuine": is_genuine,
            "genuine_stats": _score_stats(np.array([])),
            "impostor_stats": _score_stats(np.array([])),
        }

    fpr, tpr, thresholds = roc_curve(is_genuine, scores)
    fnr = 1 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2 * 100)
    threshold_eer = float(thresholds[eer_idx]) if np.isfinite(thresholds[eer_idx]) else float(np.max(scores) + 1e-6)
    threshold_far, tar_at_far, actual_far = threshold_for_far(fpr, tpr, thresholds, target_far=target_far)
    selected = threshold_far if selected_threshold is None else float(selected_threshold)
    selected_rates = rates_at_threshold(scores, is_genuine, selected)
    eer_rates = rates_at_threshold(scores, is_genuine, threshold_eer)

    genuine_scores = scores[is_genuine == 1]
    impostor_scores = scores[is_genuine == 0]
    return {
        "eer": eer,
        "auc": float(auc(fpr, tpr)),
        "tar_at_01far": float(tar_at_far),
        "threshold_eer": threshold_eer,
        "threshold_far": float(threshold_far),
        "actual_far": float(actual_far),
        "acc_eer": float(((scores >= threshold_eer).astype(int) == is_genuine).mean() * 100),
        "eer_far": eer_rates["far"],
        "eer_frr": eer_rates["frr"],
        "selected_threshold": selected,
        "selected_far": selected_rates["far"],
        "selected_frr": selected_rates["frr"],
        "selected_tar": selected_rates["tar"],
        "fpr": fpr,
        "tpr": tpr,
        "scores": scores,
        "is_genuine": is_genuine,
        "genuine_stats": _score_stats(genuine_scores),
        "impostor_stats": _score_stats(impostor_scores),
    }


def compute_verification_metrics(
    embeds: np.ndarray,
    labels: np.ndarray,
    n_pairs: int | None = 100_000,
    seed: int = 0,
    target_far: float = 0.001,
    impostor_multiplier: int = 5,
    selected_threshold: float | None = None,
) -> dict[str, Any]:
    pair_data = sample_pair_scores(embeds, labels, n_pairs=n_pairs, seed=seed, impostor_multiplier=impostor_multiplier)
    metrics = metrics_from_scores(
        pair_data["scores"],  # type: ignore[arg-type]
        pair_data["is_genuine"],  # type: ignore[arg-type]
        target_far=target_far,
        selected_threshold=selected_threshold,
    )
    metrics.update(
        {
            "pair_a": pair_data["pair_a"],
            "pair_b": pair_data["pair_b"],
            "n_genuine_pairs": int(pair_data["n_genuine_pairs"]),
            "n_impostor_pairs": int(pair_data["n_impostor_pairs"]),
        }
    )
    return metrics


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    skip = {"fpr", "tpr", "scores", "is_genuine", "pair_a", "pair_b"}
    return {k: v for k, v in metrics.items() if k not in skip}


def save_score_distribution(path: str | Path, metrics: dict[str, Any]) -> None:
    scores = np.asarray(metrics.get("scores", []))
    is_genuine = np.asarray(metrics.get("is_genuine", []))
    pair_a = np.asarray(metrics.get("pair_a", np.arange(len(scores))))
    pair_b = np.asarray(metrics.get("pair_b", np.arange(len(scores))))
    rows = (
        {
            "pair_a": int(pair_a[i]) if i < len(pair_a) else "",
            "pair_b": int(pair_b[i]) if i < len(pair_b) else "",
            "score": float(scores[i]),
            "is_genuine": int(is_genuine[i]),
        }
        for i in range(len(scores))
    )
    write_csv_rows(path, rows, ["pair_a", "pair_b", "score", "is_genuine"])


def save_roc_plot(path: str | Path, metrics: dict[str, Any], title: str = "ROC Curve") -> None:
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(7, 5))
    plt.plot(metrics["fpr"], metrics["tpr"], lw=2, label=f"AUC={metrics['auc']:.4f} EER={metrics['eer']:.2f}%")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("FAR")
    plt.ylabel("TAR")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_hist_plot(path: str | Path, metrics: dict[str, Any], title: str = "Genuine vs Impostor Scores") -> None:
    ensure_dir(Path(path).parent)
    scores = np.asarray(metrics.get("scores", []))
    is_genuine = np.asarray(metrics.get("is_genuine", []))
    gen_scores = scores[is_genuine == 1]
    imp_scores = scores[is_genuine == 0]
    bins = np.linspace(-0.2, 1.0, 80)
    plt.figure(figsize=(7, 5))
    plt.hist(imp_scores, bins=bins, alpha=0.6, label="Impostor", color="red", density=True)
    plt.hist(gen_scores, bins=bins, alpha=0.6, label="Genuine", color="green", density=True)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
