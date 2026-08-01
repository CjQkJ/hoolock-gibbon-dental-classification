from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import timm
import torch
from torch import nn


class WeightLoadError(RuntimeError):
    """Raised when a released weight is incompatible with the registered model."""


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise WeightLoadError("Checkpoint does not contain a tensor state dictionary")
    return {
        str(key): value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }


def _normalized_state_keys(key: str) -> list[str]:
    candidates = [key]
    prefixes = ("model.", "module.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for candidate in list(candidates):
            for prefix in prefixes:
                if candidate.startswith(prefix):
                    stripped = candidate[len(prefix) :]
                    if stripped not in candidates:
                        candidates.append(stripped)
                        changed = True
    return candidates


def _is_classifier_key(key: str, classifier_names: Iterable[str]) -> bool:
    return any(key == name or key.startswith(f"{name}.") for name in classifier_names)


def _is_regenerated_buffer_key(key: str) -> bool:
    return key.endswith(".relative_position_index") or key.endswith(".attn_mask")


def _load_source_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Model weight is missing: {path}")
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return _unwrap_checkpoint(torch.load(path, map_location="cpu", weights_only=False))


def _validate_released_weight(path: Path, model_config: dict[str, Any]) -> None:
    expected_hash = model_config.get("weight_sha256")
    if expected_hash and path.suffix.lower() == ".safetensors":
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise WeightLoadError(
                f"Weight SHA-256 mismatch for {path}: {actual_hash} != {expected_hash}"
            )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    classifier_names: Iterable[str],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Load a registered pretrained weight or a complete training checkpoint strictly."""
    path = Path(checkpoint_path)
    _validate_released_weight(path, model_config)
    state = _load_source_state(path)
    target = model.state_dict()
    classifier_names = list(classifier_names)
    if not classifier_names:
        raise WeightLoadError(f"No classifier metadata for {model_config.get('model_id')}")

    filtered: dict[str, torch.Tensor] = {}
    classifier_shape_mismatch: list[str] = []
    regenerated_buffers_absent_from_target: list[str] = []
    invalid_missing_in_model: list[str] = []
    invalid_shape_mismatch: list[str] = []
    for source_key, value in state.items():
        candidates = _normalized_state_keys(source_key)
        target_key = next((candidate for candidate in candidates if candidate in target), None)
        normalized = candidates[-1]
        if target_key is None:
            if _is_classifier_key(normalized, classifier_names):
                continue
            if _is_regenerated_buffer_key(normalized):
                regenerated_buffers_absent_from_target.append(normalized)
                continue
            invalid_missing_in_model.append(normalized)
            continue
        if tuple(target[target_key].shape) != tuple(value.shape):
            if _is_classifier_key(target_key, classifier_names):
                classifier_shape_mismatch.append(target_key)
            else:
                invalid_shape_mismatch.append(
                    f"{target_key}: source={tuple(value.shape)}, "
                    f"target={tuple(target[target_key].shape)}"
                )
            continue
        filtered[target_key] = value

    if invalid_missing_in_model or invalid_shape_mismatch:
        raise WeightLoadError(
            "Weight is incompatible outside the classifier; "
            f"missing_in_model={invalid_missing_in_model[:10]}, "
            f"shape_mismatch={invalid_shape_mismatch[:10]}"
        )

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    invalid_missing_after_load = [
        key
        for key in missing
        if not _is_classifier_key(key, classifier_names)
        and not _is_regenerated_buffer_key(key)
    ]
    invalid_unexpected = [
        key for key in unexpected if not _is_classifier_key(key, classifier_names)
    ]
    if invalid_missing_after_load or invalid_unexpected:
        raise WeightLoadError(
            "Strict model-state loading failed; "
            f"missing={invalid_missing_after_load[:10]}, "
            f"unexpected={invalid_unexpected[:10]}"
        )
    return {
        "path": str(path),
        "source_tensors": len(state),
        "matched_tensors": len(filtered),
        "classifier_shape_mismatch": sorted(set(classifier_shape_mismatch)),
        "regenerated_buffers_absent_from_target": sorted(
            set(regenerated_buffers_absent_from_target)
        ),
        "missing_after_load": sorted(missing),
        "unexpected_after_load": sorted(unexpected),
    }


def _classifier_module(model: nn.Module, classifier_names: Iterable[str]) -> nn.Module:
    names = list(classifier_names)
    if not names:
        raise WeightLoadError("Classifier metadata is required to freeze the backbone")
    if len(names) != 1:
        raise WeightLoadError(f"Expected one classifier module, got {names}")
    try:
        return model.get_submodule(names[0])
    except AttributeError as exc:
        raise WeightLoadError(f"Classifier module does not exist: {names[0]}") from exc


def freeze_backbone(model: nn.Module, classifier_names: Iterable[str]) -> None:
    names = list(classifier_names)
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier = _classifier_module(model, names)
    for parameter in classifier.parameters():
        parameter.requires_grad = True
    invalid_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not _is_classifier_key(name, names)
    ]
    if invalid_trainable:
        raise WeightLoadError(
            f"Backbone freeze left non-classifier parameters trainable: {invalid_trainable[:10]}"
        )


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
        load_checkpoint(
            model,
            checkpoint,
            model_config.get("classifier_names") or [],
            model_config,
        )
    if model_config.get("freeze_backbone", False):
        freeze_backbone(model, model_config.get("classifier_names") or [])
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
