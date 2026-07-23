import pytest

from eval.metrics import CLASS_LABELS, compute_classification_metrics
from service.models import Classification


def test_perfect_predictions() -> None:
    labels = [
        Classification.EMERGENT,
        Classification.URGENT,
        Classification.ROUTINE,
        Classification.ADMIN,
    ]
    metrics = compute_classification_metrics(labels, labels)

    assert metrics["n_examples"] == 4
    assert metrics["emergent_recall"] == 1.0
    assert metrics["macro"]["f1"] == 1.0
    for label in CLASS_LABELS:
        assert metrics["per_class"][label]["f1"] == 1.0


def test_emergent_recall_on_miss() -> None:
    y_true = [Classification.EMERGENT, Classification.EMERGENT, Classification.URGENT]
    y_pred = [Classification.URGENT, Classification.EMERGENT, Classification.URGENT]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["emergent_recall"] == 0.5
    assert metrics["per_class"]["emergent"]["recall"] == 0.5


def test_confusion_matrix_shape() -> None:
    y_true = [Classification.ADMIN, Classification.ROUTINE]
    y_pred = [Classification.ROUTINE, Classification.ROUTINE]

    metrics = compute_classification_metrics(y_true, y_pred)
    matrix = metrics["confusion_matrix"]

    assert matrix["labels"] == list(CLASS_LABELS)
    assert len(matrix["matrix"]) == len(CLASS_LABELS)
    assert len(matrix["matrix"][0]) == len(CLASS_LABELS)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_classification_metrics(
            [Classification.ADMIN],
            [Classification.ADMIN, Classification.ROUTINE],
        )
