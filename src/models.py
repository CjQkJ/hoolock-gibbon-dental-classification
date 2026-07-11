from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn


def _unwrap_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    return {str(key).removeprefix("module."): value for key, value in checkpoint.items() if torch.is_tensor(value)}


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path)
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = _unwrap_checkpoint(torch.load(path, map_location="cpu", weights_only=False))
    model_state = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in model_state and model_state[key].shape == value.shape}
    model.load_state_dict(compatible, strict=False)


def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(classifier, nn.Module):
        for parameter in classifier.parameters():
            parameter.requires_grad = True
        return
    for name, parameter in model.named_parameters():
        if any(token in name.lower() for token in ("classifier", "head", "fc")):
            parameter.requires_grad = True


def build_model(
    model_config: dict[str, Any],
    checkpoint: str | Path | None = None,
    pretrained: bool = True,
) -> nn.Module:
    model = timm.create_model(
        model_config["timm_id"],
        pretrained=bool(pretrained and checkpoint is None),
        num_classes=2,
    )
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
    if model_config.get("freeze_backbone", False):
        freeze_backbone(model)
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
