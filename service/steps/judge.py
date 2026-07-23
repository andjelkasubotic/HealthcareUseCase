from typing import Protocol

from service.models import Classification, DraftJudgeResult, StructuredFields


class DraftJudge(Protocol):
    async def review(
        self,
        message: str,
        draft_response: str,
        classifications: list[Classification],
        fields: StructuredFields,
    ) -> DraftJudgeResult: ...


class DummyJudge:
    """Pass-through judge for dummy mode (no API calls)."""

    async def review(
        self,
        message: str,
        draft_response: str,
        classifications: list[Classification],
        fields: StructuredFields,
    ) -> DraftJudgeResult:
        return DraftJudgeResult(
            approved=True,
            final_draft=draft_response,
        )
