from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from eval.data.candidate_io import load_candidates
from eval.data.label_candidates import (
    DummyLabelerA,
    DummyLabelerB,
    LabelerPair,
    build_labeler_pair,
    label_candidates,
    label_one_candidate,
    should_auto_accept,
    split_results,
)
from eval.schemas import LabelingDecision, UnlabeledCandidate
from service.models import Classification

PREVIEW_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "eval"
    / "raw"
    / "chatdoctor_candidates.preview.jsonl"
)


def test_should_auto_accept_when_models_agree_with_confidence() -> None:
    a = LabelingDecision(label=Classification.URGENT, rationale="a", confidence=0.9)
    b = LabelingDecision(label=Classification.URGENT, rationale="b", confidence=0.85)
    assert should_auto_accept(a, b, min_confidence=0.7)


def test_should_not_auto_accept_on_disagreement() -> None:
    a = LabelingDecision(label=Classification.URGENT, rationale="a", confidence=0.9)
    b = LabelingDecision(label=Classification.ROUTINE, rationale="b", confidence=0.9)
    assert not should_auto_accept(a, b, min_confidence=0.7)


def test_should_not_auto_accept_on_low_confidence() -> None:
    a = LabelingDecision(label=Classification.ADMIN, rationale="a", confidence=0.5)
    b = LabelingDecision(label=Classification.ADMIN, rationale="b", confidence=0.95)
    assert not should_auto_accept(a, b, min_confidence=0.7)


@pytest.mark.asyncio
async def test_label_candidates_dummy_mode_produces_splits() -> None:
    candidates = load_candidates(PREVIEW_FIXTURE)
    pair = build_labeler_pair(
        mode="dummy",
        labeler_a_model="dummy-a",
        labeler_b_model="dummy-b",
    )

    results = await label_candidates(candidates, pair, min_confidence=0.7)
    auto_labeled, review_queue, full_results = split_results(results)

    assert len(results) == len(candidates)
    assert len(full_results) == len(candidates)
    assert len(auto_labeled) + len(review_queue) == len(candidates)
    assert all(row["label_method"] == "multi_model_agreement" for row in auto_labeled)
    assert all(row["status"] == "needs_review" for row in review_queue)


@pytest.mark.asyncio
async def test_label_one_candidate_auto_accepts_llm_agreement() -> None:
    candidate = UnlabeledCandidate(
        id="c-001",
        message="Painful blisters on my hands",
        source="chatdoctor",
    )
    labeler_a = AsyncMock()
    labeler_a.model_name = "gpt-4.1"
    labeler_a.label = AsyncMock(
        return_value=LabelingDecision(
            label=Classification.URGENT,
            rationale="blisters",
            confidence=0.92,
        )
    )
    labeler_b = AsyncMock()
    labeler_b.model_name = "gpt-5.2"
    labeler_b.label = AsyncMock(
        return_value=LabelingDecision(
            label=Classification.URGENT,
            rationale="blisters",
            confidence=0.88,
        )
    )
    pair = LabelerPair(
        labeler_a=labeler_a,
        labeler_b=labeler_b,
        labeler_a_model="gpt-4.1",
        labeler_b_model="gpt-5.2",
    )

    result = await label_one_candidate(candidate, pair, min_confidence=0.7)

    assert result.auto_accepted
    assert result.label == Classification.URGENT


@pytest.mark.asyncio
async def test_dummy_labelers_disagree_on_mixed_admin_clinical() -> None:
    message = "I have a billing question and need moisturizer advice for dry skin."
    decision_a = await DummyLabelerA().label(message)
    decision_b = await DummyLabelerB().label(message)
    assert decision_a.label == Classification.ADMIN
    assert decision_b.label == Classification.ROUTINE
