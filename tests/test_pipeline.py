import pytest

from service.kb.build import build_knowledge_base
from service.models import Classification, DraftJudgeResult, DraftStatus
from service.pipeline import DraftPipeline
from service.settings import AppSettings
from service.steps.drafter import DummyDrafter


class RewritingJudge:
    async def review(self, message, draft_response, classifications, fields):
        return DraftJudgeResult(
            approved=False,
            violations=["diagnosis_statement"],
            reason="Draft contained a diagnosis.",
            was_rewritten=True,
            original_draft=draft_response,
            final_draft=(
                "Thank you for your message. A clinician will review your "
                "concerns and follow up with you."
            ),
        )


@pytest.mark.asyncio
async def test_pipeline_returns_complete_draft_record() -> None:
    pipeline = DraftPipeline()
    draft = await pipeline.run("I have a routine question about follow-up care")

    assert draft.input_message == "I have a routine question about follow-up care"
    assert draft.classifications == [Classification.ROUTINE]
    assert draft.structured_fields.symptoms == "unspecified symptom"
    assert draft.abstraction_flag is False
    assert draft.judge_result.approved is True
    assert draft.needs_human_review is False
    assert draft.human_review_reason is None
    assert "routine" in draft.draft_response
    assert draft.status == DraftStatus.READY


@pytest.mark.asyncio
async def test_pipeline_sets_abstraction_flag_for_emergent() -> None:
    pipeline = DraftPipeline()
    draft = await pipeline.run("This is an emergency")

    assert draft.classifications == [Classification.EMERGENT]
    assert draft.abstraction_flag is True
    assert draft.needs_human_review is True
    assert draft.human_review_reason is not None
    assert "flagged for clinical review" in draft.draft_response


@pytest.mark.asyncio
async def test_pipeline_retrieves_faq_context_when_kb_present() -> None:
    settings = AppSettings(llm_mode="dummy", kb_faq_path="data/kb/dermatology_faq.txt")
    kb = await build_knowledge_base(settings)
    pipeline = DraftPipeline(knowledge_base=kb, settings=settings)

    draft = await pipeline.run("I need to reschedule my dermatology appointment")

    assert draft.retrieved_sources
    assert draft.retrieved_sources[0].id == "faq-002"


@pytest.mark.asyncio
async def test_pipeline_uses_rewritten_draft_and_flags_review() -> None:
    unsafe_drafter = DummyDrafter()

    async def unsafe_draft(*args, **kwargs) -> str:
        return "You have cellulitis. Take 500 mg antibiotics twice daily."

    unsafe_drafter.draft = unsafe_draft  # type: ignore[method-assign]

    pipeline = DraftPipeline(drafter=unsafe_drafter, judge=RewritingJudge())
    draft = await pipeline.run("I have a spreading rash")

    assert draft.judge_result.was_rewritten is True
    assert "clinician will review" in draft.draft_response
    assert "cellulitis" not in draft.draft_response.lower()
    assert draft.needs_human_review is True
    assert "auto-corrected" in draft.human_review_reason.lower()
