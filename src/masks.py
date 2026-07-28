from __future__ import annotations

import numpy as np
import torch


def make_soft_angular_mask(
    polar_w: int = 512,
    keep_frac: float = 0.60,
    floor: float = 0.15,
    soft_edge: int = 24,
    enabled: bool = True,
) -> torch.Tensor:
    if not enabled:
        return torch.ones(polar_w).float()

    keep_frac = float(np.clip(keep_frac, 0.05, 1.00))
    floor = float(np.clip(floor, 0.00, 1.00))
    soft_edge = int(max(0, soft_edge))

    x = np.arange(polar_w, dtype=np.float32)
    centers = [0.0, polar_w / 2.0]
    total_keep = int(round(polar_w * keep_frac))
    side_keep = max(1, total_keep // 2)
    half_core = side_keep / 2.0
    mask = np.full(polar_w, floor, dtype=np.float32)

    for center in centers:
        dist = np.abs(x - center)
        dist = np.minimum(dist, polar_w - dist)
        local = np.full(polar_w, floor, dtype=np.float32)
        core = dist <= half_core
        local[core] = 1.0
        if soft_edge > 0:
            edge = (dist > half_core) & (dist <= half_core + soft_edge)
            t = (dist[edge] - half_core) / float(soft_edge)
            local[edge] = floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(np.pi * t))
        mask = np.maximum(mask, local)

    return torch.from_numpy(mask).float()


def make_mask_bank(
    polar_w: int,
    bank_size: int,
    keep_frac_min: float,
    keep_frac_max: float,
    floor: float,
    soft_edge: int,
) -> torch.Tensor:
    fracs = np.linspace(keep_frac_min, keep_frac_max, max(2, int(bank_size)))
    return torch.stack(
        [
            make_soft_angular_mask(
                polar_w=polar_w,
                keep_frac=float(frac),
                floor=floor,
                soft_edge=soft_edge,
                enabled=True,
            )
            for frac in fracs
        ],
        dim=0,
    ).unsqueeze(1).unsqueeze(1)
