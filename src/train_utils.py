from __future__ import annotations

import os
import random
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


RESUME_CONFIG_KEYS = (
    "run_id",
    "model_name",
    "use_msff",
    "embedding_dim",
    "polar_height",
    "polar_width",
    "use_angular_mask",
    "mask_type",
    "arcface_margin",
    "arcface_scale",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "seed",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _proc_mib(path: str, key: str) -> float | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    value_kib = float(parts[1])
                    return value_kib / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def log_resources(tag: str, device: torch.device, started_at: float | None = None) -> None:
    fields = [f"[resources] {tag}", f"pid={os.getpid()}"]
    if started_at is not None:
        fields.append(f"elapsed_hours={(time.time() - started_at) / 3600.0:.3f}")

    current_rss = _proc_mib("/proc/self/status", "VmRSS")
    peak_rss = _proc_mib("/proc/self/status", "VmHWM")
    available_ram = _proc_mib("/proc/meminfo", "MemAvailable")
    if current_rss is not None:
        fields.append(f"rss_mib={current_rss:.1f}")
    if peak_rss is not None:
        fields.append(f"rss_peak_mib={peak_rss:.1f}")
    if available_ram is not None:
        fields.append(f"system_available_mib={available_ram:.1f}")

    if device.type == "cuda":
        fields.extend(
            [
                f"cuda_allocated_mib={torch.cuda.memory_allocated(device) / 2**20:.1f}",
                f"cuda_reserved_mib={torch.cuda.memory_reserved(device) / 2**20:.1f}",
                f"cuda_peak_allocated_mib={torch.cuda.max_memory_allocated(device) / 2**20:.1f}",
                f"cuda_peak_reserved_mib={torch.cuda.max_memory_reserved(device) / 2**20:.1f}",
            ]
        )
    print(" ".join(fields), flush=True)


def autocast_ctx(use_amp: bool):
    if not use_amp:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def describe_rng_state_type(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return (
            f"{type(value).__name__}(dtype={value.dtype}, device={value.device}, "
            f"shape={tuple(value.shape)})"
        )
    if isinstance(value, np.ndarray):
        return f"ndarray(dtype={value.dtype}, shape={value.shape})"
    if isinstance(value, (list, tuple)):
        entries = []
        for item in value[:4]:
            if isinstance(item, (torch.Tensor, np.ndarray)):
                entries.append(describe_rng_state_type(item))
            else:
                entries.append(type(item).__name__)
        if len(value) > 4:
            entries.append("...")
        return f"{type(value).__name__}(len={len(value)}, items=[{', '.join(entries)}])"
    return type(value).__name__


def as_cpu_byte_tensor(value: Any) -> torch.ByteTensor:
    if isinstance(value, torch.ByteTensor):
        return value.detach().cpu()
    if not isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
        raise TypeError(f"unsupported RNG state type: {type(value).__name__}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu").clone()


def restore_rng_state(state: dict[str, Any]) -> None:
    print(
        "RNG state types before restoration: "
        + ", ".join(
            f"{name}={describe_rng_state_type(state.get(name))}"
            for name in ("python", "numpy", "torch", "cuda")
        ),
        flush=True,
    )

    try:
        random.setstate(state["python"])
    except Exception as exc:
        warnings.warn(f"Could not restore Python RNG state; continuing resume: {exc}", RuntimeWarning)

    try:
        np.random.set_state(state["numpy"])
    except Exception as exc:
        warnings.warn(f"Could not restore NumPy RNG state; continuing resume: {exc}", RuntimeWarning)

    try:
        torch.set_rng_state(as_cpu_byte_tensor(state["torch"]))
    except Exception as exc:
        warnings.warn(f"Could not restore PyTorch CPU RNG state; continuing resume: {exc}", RuntimeWarning)

    if torch.cuda.is_available() and "cuda" in state:
        try:
            cuda_state = state["cuda"]
            if not isinstance(cuda_state, (list, tuple)):
                raise TypeError(f"unsupported CUDA RNG state container: {type(cuda_state).__name__}")
            torch.cuda.set_rng_state_all([as_cpu_byte_tensor(item) for item in cuda_state])
        except Exception as exc:
            warnings.warn(f"Could not restore PyTorch CUDA RNG state; continuing resume: {exc}", RuntimeWarning)


def safe_torch_load(path: Path, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(data, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def make_resume_metadata(config: dict[str, Any], num_classes: int) -> dict[str, Any]:
    metadata = {key: config.get(key) for key in RESUME_CONFIG_KEYS}
    metadata["num_classes"] = int(num_classes)
    return metadata


def validate_resume_metadata(
    resume: dict[str, Any],
    config: dict[str, Any],
    num_classes: int,
) -> None:
    saved = resume.get("resume_metadata")
    if saved is None:
        warnings.warn(
            "Legacy resume state has no configuration metadata; relying on strict state-dict loading.",
            RuntimeWarning,
        )
        return

    expected = make_resume_metadata(config, num_classes)
    mismatches = [
        f"{key}: saved={saved.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if saved.get(key) != value
    ]
    if mismatches:
        raise RuntimeError("Incompatible resume state:\n  " + "\n  ".join(mismatches))


def forward_embeddings(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    output = model(imgs)
    if isinstance(output, tuple):
        return output[0]
    return output


def extract_embeddings(loader, model: nn.Module, device: torch.device, use_amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    embeds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast_ctx(use_amp and device.type == "cuda"):
                embeds = forward_embeddings(model, imgs)
            embeds_list.append(embeds.float().cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(embeds_list), np.concatenate(labels_list)


def model_summary_text(
    model: nn.Module,
    arcface: nn.Module | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    base_model = unwrap_model(model)
    total = sum(p.numel() for p in base_model.parameters())
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    lines = [
        f"model_class: {base_model.__class__.__name__}",
        f"parameters_total: {total}",
        f"parameters_trainable: {trainable}",
    ]
    if arcface is not None:
        lines.append(f"arcface_parameters: {sum(p.numel() for p in arcface.parameters())}")
    if config:
        for key in ["run_id", "model_name", "use_msff", "use_angular_mask", "mask_type", "polar_height", "polar_width"]:
            lines.append(f"{key}: {config.get(key)}")
    return "\n".join(lines) + "\n"
