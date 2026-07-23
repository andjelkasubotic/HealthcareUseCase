import json
from pathlib import Path

import pytest

from eval.data.build_dataset import (
    merge_datasets,
    validate_dataset,
    write_labeled_jsonl,
)
from eval.dataset import load_jsonl
from service.models import Classification

ROOT = Path(__file__).resolve().parent.parent


def test_merge_produces_75_examples() -> None:
    examples = merge_datasets(
        auto_labeled=ROOT / "data/eval/auto_labeled.jsonl",
        adjudicated=ROOT / "data/eval/review_adjudicated.jsonl",
        handcrafted=ROOT / "data/eval/handcrafted.jsonl",
    )
    assert len(examples) == 75
    assert len({example.id for example in examples}) == 75


def test_validate_dataset_reports_admin_shortfall() -> None:
    examples = merge_datasets(
        auto_labeled=ROOT / "data/eval/auto_labeled.jsonl",
        adjudicated=ROOT / "data/eval/review_adjudicated.jsonl",
        handcrafted=ROOT / "data/eval/handcrafted.jsonl",
    )
    errors = validate_dataset(examples)
    assert any("admin" in error for error in errors)


def test_write_labeled_jsonl_round_trip(tmp_path: Path) -> None:
    examples = merge_datasets(
        auto_labeled=ROOT / "data/eval/auto_labeled.jsonl",
        adjudicated=ROOT / "data/eval/review_adjudicated.jsonl",
        handcrafted=ROOT / "data/eval/handcrafted.jsonl",
    )
    output = tmp_path / "labeled.jsonl"
    write_labeled_jsonl(output, examples)
    loaded = load_jsonl(output)
    assert len(loaded) == len(examples)
