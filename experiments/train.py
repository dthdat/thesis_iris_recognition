from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import IrisDataset, load_or_create_split
from src.io_utils import append_csv_row, ensure_dir, load_yaml, resolve_dataset_root, resolve_run_dir, resolve_split_dir, write_json, write_yaml
from src.losses import ArcFaceHead
from src.metrics import compact_metrics, compute_verification_metrics, save_hist_plot, save_roc_plot, save_score_distribution
from src.models import build_model, describe_model
from src.train_utils import (
    atomic_torch_save,
    autocast_ctx,
    capture_rng_state,
    extract_embeddings,
    forward_embeddings,
    get_device,
    log_resources,
    make_grad_scaler,
    make_resume_metadata,
    model_summary_text,
    restore_rng_state,
    safe_torch_load,
    set_seed,
    unwrap_model,
    validate_resume_metadata,
)


LOG_FIELDS = [
    "epoch",
    "train_loss",
    "train_acc",
    "val_eer",
    "val_auc",
    "val_tar_at_01far",
    "val_threshold_eer",
    "val_threshold_far",
    "learning_rate",
    "epoch_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one iris-recognition baseline config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override epochs for smoke tests.")
    parser.add_argument("--regenerate-split", action="store_true", help="Regenerate split CSVs instead of reusing existing frozen split.")
    parser.add_argument("--resume-state", default=None, help="Optional path for save/resume state between process restarts.")
    return parser.parse_args()


def make_loader(dataset: IrisDataset, config: dict, shuffle: bool) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    kwargs = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=shuffle, **kwargs)


def save_best_artifacts(run_dir: Path, metrics: dict, prefix: str = "val") -> None:
    write_json(run_dir / f"{prefix}_metrics.json", compact_metrics(metrics))
    save_score_distribution(run_dir / f"score_distribution_{prefix}.csv", metrics)
    save_roc_plot(run_dir / f"roc_curve_{prefix}.png", metrics, title=f"{prefix.upper()} ROC Curve")
    save_hist_plot(run_dir / f"genuine_impostor_hist_{prefix}.png", metrics, title=f"{prefix.upper()} Score Distribution")


def main() -> None:
    job_start = time.time()
    args = parse_args()
    config = load_yaml(args.config)
    if args.max_epochs is not None:
        config["epochs"] = int(args.max_epochs)
        config["max_epochs_override"] = int(args.max_epochs)

    run_id = str(config["run_id"])
    run_dir = ensure_dir(resolve_run_dir(config))
    write_yaml(run_dir / "config.yaml", config)
    set_seed(int(config.get("seed", 42)))
    device = get_device()
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")

    dataset_root = resolve_dataset_root(config)
    split_dir = ensure_dir(resolve_split_dir(config))
    split = load_or_create_split(
        dataset_root=dataset_root,
        split_dir=split_dir,
        seed=int(config.get("seed", 42)),
        train_subject_frac=float(config.get("train_subject_frac", 0.70)),
        val_subject_frac=float(config.get("val_subject_frac", 0.10)),
        min_samples=int(config.get("min_samples", 3)),
        regenerate=bool(args.regenerate_split),
    )

    print(f"Run ID: {run_id}", flush=True)
    print(f"Dataset root: {dataset_root}", flush=True)
    print(f"Split dir: {split_dir}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Device: {device}; AMP: {'ON' if use_amp else 'OFF'}", flush=True)
    print(
        f"Subjects train/val/test: {len(split['train_subjects'])}/"
        f"{len(split['val_subjects'])}/{len(split['test_subjects'])}",
        flush=True,
    )

    train_ds = IrisDataset(split["train"]["paths"], split["train"]["labels"], config, augment=True, split_name="train")
    val_ds = IrisDataset(split["val"]["paths"], split["val"]["labels"], config, augment=False, split_name="open_val")
    train_loader = make_loader(train_ds, config, shuffle=True)
    val_loader = make_loader(val_ds, config, shuffle=False)
    log_resources("after_dataset_preload", device, job_start)

    model = build_model(config).to(device)
    base_model = model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        base_model = model.module
    arcface = ArcFaceHead(
        int(config.get("embedding_dim", 512)),
        int(split["num_classes"]),
        s=float(config.get("arcface_scale", 64.0)),
        m=float(config.get("arcface_margin", 0.25)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.02)))
    optimizer = optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad]}, {"params": arcface.parameters()}],
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 3e-4)),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config["epochs"])), eta_min=1e-6)
    scaler = make_grad_scaler(use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log_resources("after_model_setup", device, job_start)

    summary = model_summary_text(model, arcface, config)
    summary += f"num_classes: {split['num_classes']}\n"
    summary += f"model_stats: {describe_model(base_model)}\n"
    (run_dir / "model_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)

    best_val_eer = float("inf")
    best_val_tar_far = -1.0
    best_val_threshold = None
    best_val_threshold_far = None
    patience_ctr = 0
    saw_val_metrics = False
    start_epoch = 0
    resume_path = Path(args.resume_state) if args.resume_state else None

    if resume_path is not None and resume_path.is_file():
        resume = safe_torch_load(resume_path, map_location=device)
        validate_resume_metadata(resume, config, int(split["num_classes"]))
        unwrap_model(model).load_state_dict(resume["model_state_dict"])
        arcface.load_state_dict(resume["arcface_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        best_val_eer = float(resume["best_val_eer"])
        best_val_tar_far = float(resume["best_val_tar_far"])
        best_val_threshold = resume["best_val_threshold"]
        best_val_threshold_far = resume["best_val_threshold_far"]
        patience_ctr = int(resume["patience_ctr"])
        saw_val_metrics = bool(resume["saw_val_metrics"])
        start_epoch = int(resume["next_epoch"])
        restore_rng_state(resume["rng_state"])
        print(f"Resuming from completed epoch {start_epoch}/{config['epochs']}: {resume_path}", flush=True)

    for epoch in range(start_epoch, int(config["epochs"])):
        start = time.time()
        model.train()
        arcface.train()
        run_loss = 0.0
        correct = 0
        total = 0

        n_train_batches = len(train_loader)
        heartbeat_every = max(1, min(25, n_train_batches // 4 or 1))
        for batch_idx, (imgs, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp):
                embeds = forward_embeddings(model, imgs)
                logits = arcface(embeds, labels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(arcface.parameters()), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            run_loss += float(loss.item()) * imgs.size(0)
            correct += int((logits.detach().argmax(1) == labels).sum().item())
            total += int(labels.size(0))
            if batch_idx == 1 or batch_idx == n_train_batches or batch_idx % heartbeat_every == 0:
                elapsed = time.time() - start
                print(
                    f"Epoch {epoch+1:02d}/{config['epochs']} batch {batch_idx}/{n_train_batches} "
                    f"loss {run_loss / max(1, total):.4f} acc {correct / max(1, total):.4f} | {elapsed:.1f}s",
                    flush=True,
                )
                log_resources(f"epoch_{epoch+1}_batch_{batch_idx}", device, job_start)

        train_loss = run_loss / max(1, total)
        train_acc = correct / max(1, total)
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{config['epochs']} validation start", flush=True)
        log_resources(f"epoch_{epoch+1}_before_validation", device, job_start)
        val_embeds, val_labels = extract_embeddings(val_loader, unwrap_model(model), device, use_amp=use_amp)
        log_resources(f"epoch_{epoch+1}_after_embedding_extraction", device, job_start)
        print(f"Epoch {epoch+1:02d}/{config['epochs']} metrics start", flush=True)
        val_metrics = compute_verification_metrics(
            val_embeds,
            val_labels,
            n_pairs=int(config.get("val_n_pairs", 100_000)),
            seed=int(config.get("seed", 42)) + epoch,
            target_far=float(config.get("target_far", 0.001)),
            impostor_multiplier=int(config.get("impostor_multiplier", 5)),
        )
        saw_val_metrics = True
        log_resources(f"epoch_{epoch+1}_after_metrics", device, job_start)

        improved_any = val_metrics["eer"] < best_val_eer
        improved_meaningful = val_metrics["eer"] < (best_val_eer - float(config.get("early_stop_min_delta", 0.0)))
        if improved_any:
            best_val_eer = float(val_metrics["eer"])
            best_val_threshold = float(val_metrics["threshold_eer"])
            best_val_threshold_far = float(val_metrics["threshold_far"])
            checkpoint = {
                "model_state_dict": unwrap_model(model).state_dict(),
                "arcface_state_dict": arcface.state_dict(),
                "num_classes": int(split["num_classes"]),
                "label_names": split["train_label_names"],
                "old_class_ids": sorted({int(row["raw_label"]) for row in split["train"]["rows"]}),
                "train_class_remap": {
                    str(row["raw_label"]): int(row["train_label"]) for row in split["train"]["rows"]
                },
                "epoch": epoch,
                "val_eer": best_val_eer,
                "val_threshold": best_val_threshold,
                "val_threshold_far": best_val_threshold_far,
                "target_far": float(config.get("target_far", 0.001)),
                "val_tar_at_target_far": float(val_metrics["tar_at_01far"]),
                "val_actual_far": float(val_metrics["actual_far"]),
                "val_auc": float(val_metrics["auc"]),
                "config": config,
            }
            atomic_torch_save(checkpoint, run_dir / "best_model.pth")
            save_best_artifacts(run_dir, val_metrics, prefix="val")
            print(
                f"  Saved best EER model: val_eer={best_val_eer:.3f}% "
                f"eer_thr={best_val_threshold:.4f} far_thr={best_val_threshold_far:.4f}",
                flush=True,
            )

        if val_metrics["tar_at_01far"] > best_val_tar_far:
            best_val_tar_far = float(val_metrics["tar_at_01far"])
            atomic_torch_save(
                {
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "arcface_state_dict": arcface.state_dict(),
                    "num_classes": int(split["num_classes"]),
                    "label_names": split["train_label_names"],
                    "epoch": epoch,
                    "val_eer": float(val_metrics["eer"]),
                    "val_threshold": float(val_metrics["threshold_eer"]),
                    "val_threshold_far": float(val_metrics["threshold_far"]),
                    "target_far": float(config.get("target_far", 0.001)),
                    "val_tar_at_target_far": float(val_metrics["tar_at_01far"]),
                    "val_actual_far": float(val_metrics["actual_far"]),
                    "val_auc": float(val_metrics["auc"]),
                    "config": config,
                },
                run_dir / "best_model_by_tar001.pth",
            )

        epoch_seconds = time.time() - start
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_eer": val_metrics["eer"],
            "val_auc": val_metrics["auc"],
            "val_tar_at_01far": val_metrics["tar_at_01far"],
            "val_threshold_eer": val_metrics["threshold_eer"],
            "val_threshold_far": val_metrics["threshold_far"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
        }
        append_csv_row(run_dir / "training_log.csv", row, LOG_FIELDS)
        print(
            f"Epoch {epoch+1:02d}/{config['epochs']} | loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"open-val EER {val_metrics['eer']:.3f}% AUC {val_metrics['auc']:.4f} "
            f"TAR@0.1%FAR {val_metrics['tar_at_01far']*100:.2f}% | {epoch_seconds:.1f}s",
            flush=True,
        )

        del val_embeds, val_labels, val_metrics
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log_resources(f"epoch_{epoch+1}_after_cleanup", device, job_start)

        should_stop = False
        if improved_meaningful:
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= int(config.get("patience", 8)):
                should_stop = True

        if resume_path is not None:
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(
                {
                    "resume_metadata": make_resume_metadata(config, int(split["num_classes"])),
                    "next_epoch": epoch + 1,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "arcface_state_dict": arcface.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_eer": best_val_eer,
                    "best_val_tar_far": best_val_tar_far,
                    "best_val_threshold": best_val_threshold,
                    "best_val_threshold_far": best_val_threshold_far,
                    "patience_ctr": patience_ctr,
                    "saw_val_metrics": saw_val_metrics,
                    "rng_state": capture_rng_state(),
                },
                resume_path,
            )
            print(f"Saved resume state after epoch {epoch+1}: {resume_path}", flush=True)
            log_resources(f"epoch_{epoch+1}_after_resume_save", device, job_start)

        if should_stop:
            print(f"Early stopping at epoch {epoch+1}", flush=True)
            break

    if not saw_val_metrics:
        raise RuntimeError("Training finished without validation metrics.")
    if resume_path is not None:
        resume_path.unlink(missing_ok=True)
    print(f"Training complete. Best open-val EER: {best_val_eer:.3f}%", flush=True)


if __name__ == "__main__":
    main()
