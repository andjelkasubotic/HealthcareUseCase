import json
from pathlib import Path

from service.models import Classification

from eval.schemas import DEFAULT_CLASS_MINIMUMS, EvalExample


def load_jsonl(path: Path | str) -> list[EvalExample]:
    """Load and validate eval examples from a JSONL file."""
    file_path = Path(path)
    examples: list[EvalExample] = []
    for line_number, line in enumerate(file_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
            examples.append(EvalExample.model_validate(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"{file_path}:{line_number}: invalid eval row — {exc}"
            ) from exc
    return examples


def count_by_label(examples: list[EvalExample]) -> dict[Classification, int]:
    counts = {label: 0 for label in Classification}
    for example in examples:
        counts[example.label] += 1
    return counts


def validate_class_minimums(
    examples: list[EvalExample],
    minimums: dict[Classification, int] | None = None,
) -> list[str]:
    """Return human-readable errors when per-class counts fall below minimums."""
    mins = minimums if minimums is not None else DEFAULT_CLASS_MINIMUMS
    counts = count_by_label(examples)
    errors: list[str] = []
    for label, required in mins.items():
        actual = counts[label]
        if actual < required:
            errors.append(f"{label.value}: {actual} examples (minimum {required})")
    return errors


def filter_by_source(examples: list[EvalExample], *sources: str) -> list[EvalExample]:
    allowed = set(sources)
    return [ex for ex in examples if ex.source in allowed]


def filter_by_tag(examples: list[EvalExample], tag: str) -> list[EvalExample]:
    return [ex for ex in examples if tag in ex.tags]
