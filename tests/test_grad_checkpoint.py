"""Fix #2: gradient checkpointing must not change the math. Output and input
gradients with checkpointing ON must match a plain forward/backward exactly."""
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models import IrisMobileFaceNet, maybe_checkpoint
from src.train_utils import (
    atomic_torch_save,
    capture_rng_state,
    make_resume_metadata,
    restore_rng_state,
    safe_torch_load,
    validate_resume_metadata,
)


def test_maybe_checkpoint_matches_direct_output_and_gradient():
    torch.manual_seed(0)
    block = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
    block.eval()

    x = torch.randn(2, 4, 8, 8)
    x_direct = x.clone().requires_grad_(True)
    out_direct = block(x_direct)
    (g_direct,) = torch.autograd.grad(out_direct.sum(), x_direct)

    x_ckpt = x.clone().requires_grad_(True)
    out_ckpt = maybe_checkpoint(block, x_ckpt, enabled=True)
    (g_ckpt,) = torch.autograd.grad(out_ckpt.sum(), x_ckpt)

    torch.testing.assert_close(out_ckpt, out_direct)
    torch.testing.assert_close(g_ckpt, g_direct)


def test_maybe_checkpoint_disabled_is_plain_call():
    block = nn.Sequential(nn.Linear(4, 4))
    block.eval()
    x = torch.randn(3, 4)
    torch.testing.assert_close(maybe_checkpoint(block, x, enabled=False), block(x))


def test_mobilefacenet_forward_runs_with_checkpointing_in_training():
    torch.manual_seed(0)
    model = IrisMobileFaceNet(num_features=64, input_size=(64, 512))
    model.train()
    x = torch.randn(2, 1, 64, 512)
    embeds, mid, deep = model(x)
    assert embeds.shape == (2, 64)
    # backward must work through the checkpointed stages
    embeds.sum().backward()


def test_resume_state_round_trip_restores_rng_and_epoch(tmp_path):
    config = {"run_id": "unit_mobilefacenet", "model_name": "mobilefacenet"}
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    rng_state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    path = tmp_path / "resume_state.pth"
    atomic_torch_save(
        {
            "next_epoch": 4,
            "rng_state": rng_state,
            "resume_metadata": make_resume_metadata(config, num_classes=1400),
        },
        path,
    )
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp").exists()

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    loaded = safe_torch_load(path, map_location="cpu")
    validate_resume_metadata(loaded, config, num_classes=1400)
    restore_rng_state(loaded["rng_state"])
    actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert loaded["next_epoch"] == 4
    assert actual == pytest.approx(expected)


def test_resume_state_rejects_model_mismatch():
    saved_config = {"run_id": "unit_mobilefacenet", "model_name": "mobilenet_v2"}
    current_config = {"run_id": "unit_mobilefacenet", "model_name": "mobilefacenet"}
    resume = {"resume_metadata": make_resume_metadata(saved_config, num_classes=1400)}

    with pytest.raises(RuntimeError, match="model_name"):
        validate_resume_metadata(resume, current_config, num_classes=1400)
