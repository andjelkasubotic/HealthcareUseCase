from pathlib import Path

import pytest

from eval.data.derm_filter import (
    find_matched_keywords,
    passes_derm_filter,
    preprocess_input,
)
from eval.data.harvest_chatdoctor import (
    harvest_chatdoctor,
    harvest_candidates,
    iter_fixture_records,
    write_candidates,
)

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "chatdoctor_harvest_input.jsonl"
)


def test_preprocess_input_truncates_long_text() -> None:
    long_text = "rash " * 400
    result = preprocess_input(long_text, max_length=100)
    assert result is not None
    assert len(result) <= 100


def test_preprocess_input_rejects_short_text() -> None:
    assert preprocess_input("too short") is None


def test_passes_derm_filter_matches_keywords() -> None:
    message = "I have a spreading rash and itchy skin on my back."
    assert passes_derm_filter(message)
    assert "rash" in find_matched_keywords(message)
    assert "itchy" in find_matched_keywords(message)


def test_passes_derm_filter_rejects_non_derm() -> None:
    assert not passes_derm_filter("What are your office hours for billing questions?")


def test_harvest_candidates_dedupes_and_samples(tmp_path: Path) -> None:
    candidates = harvest_chatdoctor(
        fixture_path=FIXTURE,
        limit=3,
        seed=7,
        max_scan=20,
        max_pool=10,
    )

    assert len(candidates) == 3
    messages = [candidate.message for candidate in candidates]
    assert len(messages) == len(set(message.casefold() for message in messages))
    assert all(candidate.source == "chatdoctor" for candidate in candidates)
    assert all(candidate.matched_keywords for candidate in candidates)


def test_harvest_candidates_respects_max_scan() -> None:
    records = list(iter_fixture_records(FIXTURE))
    candidates = harvest_candidates(records, limit=50, max_scan=2, max_pool=50)
    assert len(candidates) <= 2


def test_write_candidates_round_trip(tmp_path: Path) -> None:
    candidates = harvest_chatdoctor(fixture_path=FIXTURE, limit=2, seed=1)
    output = tmp_path / "out.jsonl"
    write_candidates(output, candidates)

    lines = output.read_text().strip().splitlines()
    assert len(lines) == len(candidates)
    assert '"source": "chatdoctor"' in lines[0]


@pytest.mark.integration
def test_harvest_from_huggingface() -> None:
    candidates = harvest_chatdoctor(
        backend="http",
        limit=5,
        seed=42,
        max_scan=5_000,
        max_pool=100,
    )
    assert 1 <= len(candidates) <= 5
    assert all(len(candidate.message) >= 20 for candidate in candidates)
