from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.dataset import count_by_label, load_jsonl, validate_class_minimums
from eval.schemas import EvalExample
from service.models import Classification

DEFAULT_OUTPUT = Path("data/eval/labeled.jsonl")
DEFAULT_AUTO = Path("data/eval/auto_labeled.jsonl")
DEFAULT_ADJUDICATED = Path("data/eval/review_adjudicated.jsonl")
DEFAULT_HANDCRAFTED = Path("data/eval/handcrafted.jsonl")
MAX_DATASET_SIZE = 100


def _load_labeled_rows(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if "label" not in payload:
            raise ValueError(f"{path}:{line_number}: missing required field 'label'")
        examples.append(EvalExample.model_validate(payload))
    return examples


def merge_datasets(
    *,
    auto_labeled: Path,
    adjudicated: Path,
    handcrafted: Path,
) -> list[EvalExample]:
    merged: list[EvalExample] = []
    seen_ids: set[str] = set()

    for path in (auto_labeled, adjudicated, handcrafted):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset source: {path}")
        for example in _load_labeled_rows(path):
            if example.id in seen_ids:
                raise ValueError(f"Duplicate id {example.id!r} while merging {path}")
            seen_ids.add(example.id)
            merged.append(example)

    merged.sort(key=lambda example: example.id)
    return merged


def validate_dataset(examples: list[EvalExample]) -> list[str]:
    errors: list[str] = []
    if len(examples) > MAX_DATASET_SIZE:
        errors.append(f"dataset has {len(examples)} rows (maximum {MAX_DATASET_SIZE})")
    errors.extend(validate_class_minimums(examples))
    return errors


def write_labeled_jsonl(path: Path, examples: list[EvalExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(example.model_dump(mode="json"), ensure_ascii=False)
        for example in examples
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def format_summary(examples: list[EvalExample]) -> str:
    counts = count_by_label(examples)
    lines = [
        f"Total examples: {len(examples)}",
        "Per-class counts:",
    ]
    for label in Classification:
        lines.append(f"  {label.value}: {counts[label]}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge labeled sources into gold labeled.jsonl."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--auto-labeled", type=Path, default=DEFAULT_AUTO)
    parser.add_argument("--adjudicated", type=Path, default=DEFAULT_ADJUDICATED)
    parser.add_argument("--handcrafted", type=Path, default=DEFAULT_HANDCRAFTED)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if per-class minimums are not met.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        examples = merge_datasets(
            auto_labeled=args.auto_labeled,
            adjudicated=args.adjudicated,
            handcrafted=args.handcrafted,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Merge failed: {exc}", file=sys.stderr)
        return 1

    errors = validate_dataset(examples)
    write_labeled_jsonl(args.output, examples)

    print(format_summary(examples))
    print(f"Wrote {args.output}")

    if errors:
        print("\nValidation notes:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
