import pytest

from service.models import (
    Classification,
    DraftJudgeResult,
    DraftRecord,
    DraftStatus,
    StructuredFields,
)
from service.store.memory import InMemoryDraftStore


def _sample_draft(message: str = "test message") -> DraftRecord:
    return DraftRecord(
        input_message=message,
        classifications=[Classification.ROUTINE],
        structured_fields=StructuredFields(symptoms="headache"),
        abstraction_flag=False,
        draft_response="dummy draft",
        judge_result=DraftJudgeResult(approved=True, final_draft="dummy draft"),
        needs_human_review=False,
        status=DraftStatus.READY,
    )


@pytest.mark.asyncio
async def test_store_save_and_get() -> None:
    store = InMemoryDraftStore()
    draft = _sample_draft()

    saved = await store.save(draft)
    fetched = await store.get(saved.id)

    assert fetched is not None
    assert fetched.id == saved.id
    assert fetched.draft_response == "dummy draft"


@pytest.mark.asyncio
async def test_store_get_missing_returns_none() -> None:
    store = InMemoryDraftStore()
    draft = _sample_draft()
    saved = await store.save(draft)

    assert await store.get(saved.id) is not None

    from uuid import uuid4

    assert await store.get(uuid4()) is None
