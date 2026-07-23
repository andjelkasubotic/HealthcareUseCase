from pydantic import BaseModel

from service.llm.config import LLMConfig
from service.llm.protocol import LLMClient
from service.llm.registry import get_llm_client
from service.llm.structured import complete_structured
from service.models import Classification
from service.steps.classifier import Classifier
from service.steps.prompts import CLASSIFY_SYSTEM_PROMPT


class ClassificationOutput(BaseModel):
    classifications: list[Classification]


class LlmClassifier:
    """Classify messages with an OpenAI model."""

    def __init__(self, config: LLMConfig) -> None:
        self._client: LLMClient = get_llm_client(config)

    async def classify(self, message: str) -> list[Classification]:
        result = await complete_structured(
            self._client,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            output_model=ClassificationOutput,
        )
        return result.classifications


def build_classifier(settings_mode: str, config: LLMConfig) -> Classifier:
    if settings_mode == "openai":
        return LlmClassifier(config)
    from service.steps.classifier import DummyClassifier

    return DummyClassifier()
