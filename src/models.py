from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

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
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)


def _classifier_module(model: nn.Module, classifier_names: Iterable[str] | None):
    if classifier_names:
        for name in classifier_names:
            try:
                return model.get_submodule(name)
            except AttributeError:
                continue
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(classifier, nn.Module):
        return classifier
    for name, module in model.named_modules():
        if any(token in name.lower() for token in ("classifier", "head", "fc")):
            return module
    raise RuntimeError(f"No classifier found for {type(model).__name__}")


def freeze_backbone(model: nn.Module, classifier_names: Iterable[str] | None = None) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier = _classifier_module(model, classifier_names)
    for parameter in classifier.parameters():
        parameter.requires_grad = True


def build_model(
    model_config: dict[str, Any],
    checkpoint: str | Path | None = None,
    pretrained: bool = True,
) -> nn.Module:
    model_id = str(model_config.get("model_id") or model_config["timm_id"])
    model = timm.create_model(
        model_id,
        pretrained=bool(pretrained and checkpoint is None),
        num_classes=2,
    )
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
    if model_config.get("freeze_backbone", False):
        freeze_backbone(model, model_config.get("classifier_names"))
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
