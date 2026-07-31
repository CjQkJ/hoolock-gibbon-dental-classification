from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import ManifestDataset, collate_batch
from .metrics import classification_metrics
from .models import build_model, trainable_parameter_count
from .sam import sam_optimizer_step
from .transforms import build_transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_models(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)["models"]
    return {row["key"]: row for row in rows}


def class_weights(dataset: ManifestDataset, device: torch.device) -> torch.Tensor:
    counts = dataset.class_counts()
    total = sum(counts.values())
    return torch.tensor(
        [total / (2 * counts[label]) for label in range(2)],
        dtype=torch.float32,
        device=device,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    for images, labels, rows in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        losses.append(float(criterion(logits, labels).item()))
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        for metadata, truth, prediction, probability in zip(
            rows,
            labels.cpu().tolist(),
            predictions.cpu().tolist(),
            probabilities.cpu().tolist(),
        ):
            prediction_rows.append(
                {
                    **metadata,
                    "true_label": truth,
                    "predicted_label": prediction,
                    "probability_hoolock": probability[0],
                    "probability_leuconedys": probability[1],
                }
            )
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = float(np.mean(losses))
    return metrics, prediction_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the fixed SAM-31 Hoolock dental classification protocol.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--models", default="configs/models_31.json")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--checkpoint", default=None, help="Load a local model weight or checkpoint for evaluation or training.")
    parser.add_argument("--physical-microbatch", type=int, default=None)
    parser.add_argument("--gpu-count", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download pretrained weights; useful for structural validation.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model_config = load_models(args.models)[args.model_key]
    training = {**config["training"], **model_config.get("training_overrides", {})}
    seed = int(config["seed"])
    seed_everything(seed)

    manifest = Path(args.manifest or config["data"]["manifest"])
    mean = tuple(float(value) for value in config["data"]["imagenet_mean"])
    std = tuple(float(value) for value in config["data"]["imagenet_std"])
    image_size = int(model_config["input_size"])
    train_transform, test_transform = build_transforms(image_size, mean, std)
    train_dataset = ManifestDataset(manifest, args.data_root, "train", train_transform)
    test_dataset = ManifestDataset(manifest, args.data_root, "test", test_transform)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_config,
        checkpoint=args.checkpoint,
        pretrained=not args.no_pretrained,
    ).to(device)

    requested_gpus = int(
        args.gpu_count
        or model_config.get("initial_gpu_count")
        or model_config.get("recommended_gpus", 1)
    )
    use_data_parallel = args.data_parallel or requested_gpus > 1
    if use_data_parallel and device.type == "cuda" and torch.cuda.device_count() >= requested_gpus:
        model = nn.DataParallel(model, device_ids=list(range(requested_gpus)))

    logical_batch = int(training.get("logical_batch_size", training["batch_size"]))
    physical_microbatch = int(
        args.physical_microbatch
        or model_config.get("initial_physical_microbatch")
        or training.get("default_physical_microbatch", logical_batch)
    )
    if physical_microbatch <= 0 or physical_microbatch > logical_batch:
        raise ValueError(
            f"physical_microbatch must be in [1, {logical_batch}], got {physical_microbatch}"
        )

    effective = {
        "protocol_id": config.get("protocol_id"),
        "model": model_config,
        "training": training,
        "logical_batch_size": logical_batch,
        "physical_microbatch": physical_microbatch,
        "gpu_count": requested_gpus,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "train_class_counts": dict(train_dataset.class_counts()),
        "test_class_counts": dict(test_dataset.class_counts()),
        "selection_split": "test",
        "selection_metric": training.get("selection_metric", "macro_f1"),
        "trainable_parameters": trainable_parameter_count(model),
        "device": str(device),
        "seed": seed,
    }
    if args.dry_run:
        print(json.dumps(effective, ensure_ascii=False, indent=2))
        return

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=logical_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=logical_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    run_dir = Path(args.output_root) / f"{datetime.now():%Y%m%d_%H%M%S}_{args.model_key}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "effective_config.json").open("w", encoding="utf-8") as handle:
        json.dump(effective, handle, ensure_ascii=False, indent=2)
    writer = SummaryWriter(run_dir / "tensorboard")

    weighted_criterion = nn.CrossEntropyLoss(weight=class_weights(train_dataset, device))
    evaluation_criterion = nn.CrossEntropyLoss()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    patience = int(training["early_stopping_patience"])
    sam_config = training.get("sam") or {}
    rho = float(sam_config.get("rho", 0.05))
    norm_epsilon = float(sam_config.get("norm_epsilon", 1e-12))
    class_names = ["hoolock", "leuconedys"]

    best_metric = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        perturbed_losses: list[float] = []
        first_norms: list[float] = []
        second_norms: list[float] = []
        train_true: list[int] = []
        train_pred: list[int] = []
        for images, labels, _ in train_loader:
            result = sam_optimizer_step(
                model,
                images,
                labels,
                weighted_criterion,
                optimizer,
                device,
                rho=rho,
                norm_epsilon=norm_epsilon,
                physical_microbatch=min(physical_microbatch, int(labels.shape[0])),
            )
            logits = result["logits"].float()
            train_losses.append(float(result["unperturbed_loss"]))
            perturbed_losses.append(float(result["perturbed_loss"]))
            first_norms.append(float(result["first_grad_norm"]))
            second_norms.append(float(result["second_grad_norm"]))
            train_true.extend(labels.tolist())
            train_pred.extend(torch.argmax(logits, dim=1).tolist())

        scheduler.step()
        test_metrics, prediction_rows = evaluate(
            model, test_loader, evaluation_criterion, device, class_names
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_sam_perturbed_loss": float(np.mean(perturbed_losses)),
            "sam_first_grad_norm": float(np.mean(first_norms)),
            "sam_second_grad_norm": float(np.mean(second_norms)),
            "train_accuracy": float(accuracy_score(train_true, train_pred)),
            "train_balanced_accuracy": float(balanced_accuracy_score(train_true, train_pred)),
            "train_macro_f1": float(f1_score(train_true, train_pred, average="macro", zero_division=0)),
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        write_csv(run_dir / "epoch_metrics.csv", history)
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)

        current_metric = float(test_metrics[training.get("selection_metric", "macro_f1")])
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                    "model_config": model_config,
                    "training_config": training,
                    "epoch": epoch,
                    "class_names": class_names,
                },
                run_dir / "best_model.pth",
            )
            with (run_dir / "best_test_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(test_metrics, handle, ensure_ascii=False, indent=2)
            write_csv(run_dir / "best_test_predictions.csv", prediction_rows)
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch} train_macro_f1={row['train_macro_f1']:.4f} "
            f"test_macro_f1={row['test_macro_f1']:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            break

    summary = {
        "model_key": args.model_key,
        "best_epoch": best_epoch,
        "selection_split": "test",
        "selection_metric": training.get("selection_metric", "macro_f1"),
        "best_test_macro_f1": best_metric,
        "logical_batch_size": logical_batch,
        "physical_microbatch": physical_microbatch,
        "gpu_count": requested_gpus,
        "epochs_completed": len(history),
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    writer.close()


if __name__ == "__main__":
    main()
