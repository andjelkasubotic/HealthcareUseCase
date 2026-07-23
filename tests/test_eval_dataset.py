from pathlib import Path

import pytest

from eval.dataset import (
    count_by_label,
    filter_by_source,
    filter_by_tag,
    load_jsonl,
    validate_class_minimums,
)
from eval.schemas import EvalExample
from service.models import Classification


FIXTURES = Path(__file__).resolve().parent.parent / "data" / "eval" / "fixtures"
HANDCRAFTED = (
    Path(__file__).resolve().parent.parent / "data" / "eval" / "handcrafted.jsonl"
)


def test_load_seed_fixture() -> None:
    examples = load_jsonl(FIXTURES / "seed.jsonl")
    assert len(examples) >= 10
    assert all(isinstance(ex.label, Classification) for ex in examples)


def test_load_handcrafted_fixture() -> None:
    examples = load_jsonl(HANDCRAFTED)
    assert len(examples) == 25
    ambiguous = filter_by_tag(examples, "ambiguous")
    assert len(ambiguous) == 10
    assert all(ex.label_rationale for ex in ambiguous)


def test_ambiguous_requires_rationale() -> None:
    with pytest.raises(ValueError, match="label_rationale"):
        EvalExample(
            id="bad-001",
            message="test",
            label=Classification.ROUTINE,
            source="handcrafted",
            tags=["ambiguous"],
        )


def test_validate_class_minimums_passes_relaxed() -> None:
    examples = load_jsonl(FIXTURES / "seed.jsonl")
    relaxed = {label: 1 for label in Classification}
    assert validate_class_minimums(examples, minimums=relaxed) == []


def test_validate_class_minimums_fails() -> None:
    examples = [
        EvalExample(
            id="only-admin",
            message="billing question",
            label=Classification.ADMIN,
            source="test",
        )
    ]
    errors = validate_class_minimums(examples)
    assert any("emergent" in err for err in errors)


def test_count_and_filter_by_source() -> None:
    examples = load_jsonl(FIXTURES / "seed.jsonl")
    counts = count_by_label(examples)
    assert counts[Classification.EMERGENT] >= 2
    seed_only = filter_by_source(examples, "seed")
    assert len(seed_only) == len(examples)
