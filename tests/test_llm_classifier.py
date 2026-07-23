from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from service.llm.config import LLMConfig
from service.models import Classification
from service.steps.llm_classifier import ClassificationOutput, LlmClassifier


@pytest.mark.asyncio
async def test_llm_classifier_parses_structured_response() -> None:
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=SecretStr("test-key"),
    )
    classifier = LlmClassifier(config)

    with patch(
        "service.steps.llm_classifier.complete_structured",
        new=AsyncMock(
            return_value=ClassificationOutput(classifications=[Classification.URGENT])
        ),
    ):
        result = await classifier.classify("I need urgent help")

    assert result == [Classification.URGENT]
