import pytest

from src.metrics import classification_metrics


def test_convnext_confusion_matrix_metrics():
    y_true = [0] * 28 + [1] * 54
    y_pred = [0] * 20 + [1] * 8 + [1] * 54
    result = classification_metrics(y_true, y_pred, ["hoolock", "leuconedys"])
    assert result["confusion_matrix"] == [[20, 8], [0, 54]]
    assert result["accuracy"] == pytest.approx(0.9024390243902439)
    assert result["balanced_accuracy"] == pytest.approx(0.8571428571428572)
    assert result["macro_f1"] == pytest.approx(0.882183908045977)

