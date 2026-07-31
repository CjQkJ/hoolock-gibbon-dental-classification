"""Exact-restoration non-adaptive SAM update used by the fixed SAM-31 protocol."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn


class SAMError(RuntimeError):
    """Raised when a SAM update cannot satisfy the fixed protocol."""


def finite_global_grad_norm(model: nn.Module, pass_name: str) -> torch.Tensor:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach().float(), ord=2)
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not norms:
        raise SAMError(f"SAM {pass_name} pass produced no trainable gradients")
    global_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    if not bool(torch.isfinite(global_norm).item()):
        raise FloatingPointError(
            f"SAM {pass_name} global gradient norm is not finite: {global_norm.item()}"
        )
    return global_norm


def _microbatches(
    images: torch.Tensor,
    labels: torch.Tensor,
    size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    for start in range(0, int(labels.shape[0]), size):
        end = min(start + size, int(labels.shape[0]))
        yield images[start:end], labels[start:end]


def _cross_entropy_denominator(
    criterion: nn.CrossEntropyLoss, labels: torch.Tensor
) -> float:
    valid = labels != int(criterion.ignore_index)
    if not bool(valid.any().item()):
        raise SAMError("Logical batch contains no valid cross-entropy labels")
    valid_labels = labels[valid].detach().cpu().long()
    if criterion.weight is None:
        return float(valid_labels.numel())
    weights = criterion.weight.detach().cpu().float()
    denominator = float(weights[valid_labels].sum().item())
    if denominator <= 0.0:
        raise SAMError(f"Cross-entropy denominator is invalid: {denominator}")
    return denominator


def _cross_entropy_sum(
    criterion: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        labels,
        weight=criterion.weight,
        ignore_index=criterion.ignore_index,
        reduction="sum",
        label_smoothing=float(criterion.label_smoothing),
    )


def _forward_backward_pass(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    physical_microbatch: int,
    *,
    collect_logits: bool,
) -> tuple[float, torch.Tensor | None]:
    logical_batch = int(labels.shape[0])
    if physical_microbatch >= logical_batch:
        device_images = images.to(device, non_blocking=True)
        device_labels = labels.to(device, non_blocking=True)
        logits = model(device_images)
        loss = criterion(logits.float(), device_labels)
        if loss.ndim != 0:
            raise SAMError("SAM criterion must return one scalar for a full logical batch")
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError(f"SAM loss is not finite: {loss.item()}")
        loss.backward()
        return float(loss.detach().item()), logits.detach().cpu() if collect_logits else None

    denominator = _cross_entropy_denominator(criterion, labels)
    total_loss = 0.0
    logits_parts: list[torch.Tensor] = []
    for image_chunk, label_chunk in _microbatches(images, labels, physical_microbatch):
        device_images = image_chunk.to(device, non_blocking=True)
        device_labels = label_chunk.to(device, non_blocking=True)
        logits = model(device_images)
        loss_sum = _cross_entropy_sum(criterion, logits.float(), device_labels)
        loss = loss_sum / denominator
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError(f"SAM microbatch loss is not finite: {loss.item()}")
        loss.backward()
        total_loss += float(loss.detach().item())
        if collect_logits:
            logits_parts.append(logits.detach().cpu())
    combined = torch.cat(logits_parts, dim=0) if collect_logits else None
    return total_loss, combined


def sam_optimizer_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    rho: float = 0.05,
    norm_epsilon: float = 1e-12,
    physical_microbatch: int | None = None,
) -> dict[str, Any]:
    """Perform one exact-restoration, non-adaptive SAM + optimizer update."""
    if not isinstance(criterion, nn.CrossEntropyLoss):
        raise SAMError("The fixed SAM-31 protocol requires nn.CrossEntropyLoss")
    if criterion.reduction != "mean":
        raise SAMError("The fixed SAM-31 CrossEntropyLoss must use reduction='mean'")
    if images.shape[0] != labels.shape[0] or labels.ndim != 1:
        raise SAMError(
            f"Invalid logical batch shapes: images={tuple(images.shape)}, labels={tuple(labels.shape)}"
        )
    logical_batch = int(labels.shape[0])
    if logical_batch <= 0:
        raise SAMError("Logical batch is empty")
    microbatch = int(physical_microbatch or logical_batch)
    if microbatch <= 0 or microbatch > logical_batch:
        raise SAMError(
            f"physical_microbatch must be between 1 and {logical_batch}, got {microbatch}"
        )
    if rho <= 0.0 or norm_epsilon <= 0.0:
        raise SAMError(f"Invalid SAM constants: rho={rho}, norm_epsilon={norm_epsilon}")

    optimizer.zero_grad(set_to_none=True)
    first_loss, unperturbed_logits = _forward_backward_pass(
        model,
        images,
        labels,
        criterion,
        device,
        microbatch,
        collect_logits=True,
    )
    first_grad_norm = finite_global_grad_norm(model, "unperturbed")
    parameters_with_grad = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    originals = [
        (name, parameter, parameter.detach().clone(memory_format=torch.preserve_format))
        for name, parameter in parameters_with_grad
    ]
    scale = float(rho) / (float(first_grad_norm.item()) + float(norm_epsilon))
    with torch.no_grad():
        for _, parameter in parameters_with_grad:
            parameter.add_(parameter.grad, alpha=scale)

    optimizer.zero_grad(set_to_none=True)
    second_loss: float | None = None
    second_grad_norm: torch.Tensor | None = None
    try:
        second_loss, _ = _forward_backward_pass(
            model,
            images,
            labels,
            criterion,
            device,
            microbatch,
            collect_logits=False,
        )
        second_grad_norm = finite_global_grad_norm(model, "perturbed")
    finally:
        with torch.no_grad():
            for _, parameter, original in originals:
                parameter.copy_(original)
        restoration_failures = [
            name for name, parameter, original in originals if not torch.equal(parameter, original)
        ]
        if restoration_failures:
            raise SAMError(
                "SAM failed to restore parameters before optimizer.step(): "
                f"{restoration_failures[:10]}"
            )

    if second_loss is None or second_grad_norm is None or unperturbed_logits is None:
        raise SAMError("SAM perturbed pass did not complete")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "logits": unperturbed_logits,
        "unperturbed_loss": first_loss,
        "perturbed_loss": second_loss,
        "first_grad_norm": float(first_grad_norm.detach().item()),
        "second_grad_norm": float(second_grad_norm.detach().item()),
        "logical_batch_size": logical_batch,
        "physical_microbatch": microbatch,
        "microbatch_count": (logical_batch + microbatch - 1) // microbatch,
        "rho": float(rho),
        "adaptive": False,
        "restoration": "exact",
    }
