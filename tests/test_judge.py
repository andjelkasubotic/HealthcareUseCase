from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from service.llm.config import LLMConfig
from service.llm.protocol import LLMCompletionResult
from service.models import Classification, StructuredFields
from service.steps.judge import DummyJudge
from service.steps.llm_judge import JudgeCheckOutput, LlmJudge


@pytest.mark.asyncio
async def test_dummy_judge_passes_through_draft() -> None:
    judge = DummyJudge()
    draft = "Thank you for your message. A clinician will review and follow up."
    result = await judge.review(
        "I have a mild rash",
        draft,
        [Classification.ROUTINE],
        StructuredFields(),
    )
    assert result.approved is True
    assert result.was_rewritten is False
    assert result.final_draft == draft


@pytest.mark.asyncio
async def test_llm_judge_approves_without_rewrite() -> None:
    config = LLMConfig(
        provider="openai",
        model="gpt-5.2",
        api_key=SecretStr("test-key"),
    )
    judge = LlmJudge(config)
    draft = "Thank you. A clinician will review your message."

    with patch(
        "service.steps.llm_judge.complete_structured",
        new=AsyncMock(
            return_value=JudgeCheckOutput(approved=True, violations=[], reason="")
        ),
    ) as mock_check:
        result = await judge.review(
            "message",
            draft,
            [Classification.ROUTINE],
            StructuredFields(),
        )

    mock_check.assert_awaited_once()
    assert result.approved is True
    assert result.was_rewritten is False
    assert result.final_draft == draft


@pytest.mark.asyncio
async def test_llm_judge_rewrites_unsafe_draft() -> None:
    config = LLMConfig(
        provider="openai",
        model="gpt-5.2",
        api_key=SecretStr("test-key"),
    )
    judge = LlmJudge(config)
    unsafe_draft = "You have cellulitis. Take 500 mg antibiotics twice daily."
    safe_draft = "Thank you for your message. A clinician will review promptly."

    mock_client = AsyncMock()
    mock_client.acompletion = AsyncMock(
        return_value=LLMCompletionResult(content=safe_draft)
    )

    with patch(
        "service.steps.llm_judge.complete_structured",
        new=AsyncMock(
            return_value=JudgeCheckOutput(
                approved=False,
                violations=["diagnosis_statement", "specific_dosage"],
                reason="Draft diagnoses and prescribes medication.",
            )
        ),
    ), patch.object(judge, "_client", mock_client):
        result = await judge.review(
            "I have a spreading rash",
            unsafe_draft,
            [Classification.URGENT],
            StructuredFields(symptoms="rash"),
        )

    mock_client.acompletion.assert_awaited_once()
    assert result.approved is False
    assert result.was_rewritten is True
    assert result.original_draft == unsafe_draft
    assert result.final_draft == safe_draft
