from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from .models import build_model
from .transforms import build_transforms


def resolve_module(model: torch.nn.Module, dotted_path: str) -> torch.nn.Module:
    module = model
    for token in dotted_path.split("."):
        module = module[int(token)] if token.isdigit() else getattr(module, token)
    return module


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def __call__(self, image: torch.Tensor, target_class: int) -> tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        logits[:, target_class].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("目标层没有产生 Grad-CAM 激活或梯度。")
        activations = self.activations
        gradients = self.gradients
        if activations.ndim != 4:
            raise ValueError(f"目标层输出应为 NCHW，实际为 {tuple(activations.shape)}")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-12)
        return cam.detach().cpu().numpy(), logits.detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def safe_name(row: dict[str, str]) -> str:
    raw = f"{row['individual_id']}_{row['tooth_type']}_{Path(row['relative_path']).stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def full_overlay(image: Image.Image, cam: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heatmap = plt.get_cmap("jet")(cam)[..., :3].astype(np.float32)
    overlay = np.clip((1.0 - alpha) * base + alpha * heatmap, 0.0, 1.0)
    return heatmap, overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="生成论文使用的 ConvNeXt Grad-CAM，无 cutoff 或 mask。")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", default="metadata/dataset_manifest.csv")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--models", default="configs/models_31.json")
    parser.add_argument("--model-key", default="convnext_base")
    parser.add_argument("--target-layer", default="stages.3")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--source-type", default="original")
    parser.add_argument("--tooth-type", choices=["M1", "M2", "M3"], default=None)
    parser.add_argument("--target", choices=["true", "predicted"], default="true")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--output", default="runs/gradcam")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    models = json.loads(Path(args.models).read_text(encoding="utf-8"))["models"]
    model_config = next(row for row in models if row["key"] == args.model_key)
    mean = tuple(config["data"]["imagenet_mean"])
    std = tuple(config["data"]["imagenet_std"])
    _, transform = build_transforms(int(model_config["input_size"]), mean, std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config, checkpoint=args.checkpoint, pretrained=False).to(device).eval()
    cam_generator = GradCAM(model, resolve_module(model, args.target_layer))

    with Path(args.manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if row["split"] == args.split
        and row["source_type"] == args.source_type
        and (args.tooth_type is None or row["tooth_type"] == args.tooth_type)
    ]
    output_root = Path(args.output)
    for subdir in ("original", "heatmap", "overlay"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for row in rows:
        path = Path(args.data_root) / row["relative_path"]
        with Image.open(path) as opened:
            original = opened.convert("RGB").resize(
                (int(model_config["input_size"]), int(model_config["input_size"])),
                Image.Resampling.BILINEAR,
            )
            tensor = transform(opened.convert("RGB")).unsqueeze(0).to(device)
        true_label = int(row["label"])
        with torch.enable_grad():
            if args.target == "predicted":
                predicted = int(model(tensor).argmax(dim=1).item())
                cam, logits = cam_generator(tensor, predicted)
                target_class = predicted
            else:
                cam, logits = cam_generator(tensor, true_label)
                target_class = true_label
        heatmap, overlay = full_overlay(original, cam, args.alpha)
        name = safe_name(row) + ".png"
        original.save(output_root / "original" / name)
        plt.imsave(output_root / "heatmap" / name, heatmap)
        plt.imsave(output_root / "overlay" / name, overlay)
        summary.append(
            {
                **row,
                "target_class": target_class,
                "predicted_class": int(logits.argmax(dim=1).item()),
                "overlay_mode": "full_no_cutoff_no_mask",
                "target_layer": args.target_layer,
                "output_name": name,
            }
        )
    cam_generator.close()
    with (output_root / "gradcam_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()

