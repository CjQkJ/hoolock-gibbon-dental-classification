from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict[str, Any]:
    """计算论文 Table S4 使用的二分类指标。"""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
        "classification_report": report,
    }

