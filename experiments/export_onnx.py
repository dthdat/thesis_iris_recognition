from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import load_yaml, write_json
from src.models import build_model
from src.train_utils import get_device


class ONNXWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        output = self.backbone(x)
        return output[0] if isinstance(output, tuple) else output


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained embedding model to ONNX.")
    parser.add_argument("--run", required=True, help="Run directory, e.g. runs/b1_arciris_nomask.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    config = load_yaml(run_dir / "config.yaml")
    device = get_device()
    ckpt = safe_torch_load(run_dir / "best_model.pth", map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    export_model = ONNXWrapper(model).to(device).eval()
    dummy = torch.randn(1, 1, int(config["polar_height"]), int(config["polar_width"]), device=device)
    onnx_path = run_dir / f"{config['run_id']}_embedding.onnx"
    export_errors = []
    exported_opset = None
    for opset in [17, 13, 12]:
        try:
            torch.onnx.export(
                export_model,
                dummy,
                onnx_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["iris_polar"],
                output_names=["embedding"],
                dynamic_axes={"iris_polar": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            )
            exported_opset = opset
            break
        except Exception as exc:
            export_errors.append((opset, repr(exc)))
    if exported_opset is None:
        raise RuntimeError(f"ONNX export failed: {export_errors}")
    metadata = {
        "onnx_path": str(onnx_path),
        "opset": exported_opset,
        "polar_height": config["polar_height"],
        "polar_width": config["polar_width"],
        "norm_mean": config.get("norm_mean", 0.449),
        "norm_std": config.get("norm_std", 0.226),
        "radial_inner": config["radial_inner"],
        "radial_outer": config["radial_outer"],
        "use_angular_mask": config.get("use_angular_mask", True),
        "val_threshold_eer": ckpt.get("val_threshold"),
        "val_threshold_target_far": ckpt.get("val_threshold_far"),
        "target_far": ckpt.get("target_far"),
    }
    write_json(run_dir / "deployment_metadata.json", metadata)
    (run_dir / "deployment_pipeline_note.md").write_text(
        "\n".join(
            [
                "# Deployment Pipeline Note",
                "",
                "The ONNX file exports the embedding network only.",
                "Deployment still requires acquisition, segmentation, Daugman normalization, radial crop, optional angular mask, normalization, embedding inference, cosine matching, and thresholding.",
                "",
                json.dumps(metadata, indent=2),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Exported ONNX: {onnx_path} (opset {exported_opset})")


if __name__ == "__main__":
    main()
