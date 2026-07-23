from collections.abc import Sequence

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from service.models import Classification


CLASS_LABELS: tuple[str, ...] = tuple(c.value for c in Classification)


def _to_label_strings(labels: Sequence[Classification | str]) -> list[str]:
    return [
        label.value if isinstance(label, Classification) else label for label in labels
    ]


def compute_classification_metrics(
    y_true: Sequence[Classification | str],
    y_pred: Sequence[Classification | str],
) -> dict:
    """Compute per-class and aggregate metrics for primary-label classification."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred length mismatch: {len(y_true)} vs {len(y_pred)}"
        )

    true_labels = _to_label_strings(y_true)
    pred_labels = _to_label_strings(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=list(CLASS_LABELS),
        zero_division=0,
    )

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(CLASS_LABELS)
    }

    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())

    emergent_idx = CLASS_LABELS.index(Classification.EMERGENT.value)
    emergent_recall = float(recall[emergent_idx])

    matrix = confusion_matrix(true_labels, pred_labels, labels=list(CLASS_LABELS))

    return {
        "per_class": per_class,
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "emergent_recall": emergent_recall,
        "confusion_matrix": {
            "labels": list(CLASS_LABELS),
            "matrix": matrix.tolist(),
        },
        "classification_report": classification_report(
            true_labels,
            pred_labels,
            labels=list(CLASS_LABELS),
            zero_division=0,
        ),
        "n_examples": len(true_labels),
    }
