from service.llm.config import LLMConfig
from service.llm.protocol import LLMClient
from service.llm.registry import get_llm_client
from service.llm.structured import complete_structured
from service.models import StructuredFields
from service.steps.extractor import Extractor
from service.steps.prompts import EXTRACT_SYSTEM_PROMPT


class LlmExtractor:
    """Extract structured clinical fields with an OpenAI model."""

    def __init__(self, config: LLMConfig) -> None:
        self._client: LLMClient = get_llm_client(config)

    async def extract(self, message: str) -> StructuredFields:
        return await complete_structured(
            self._client,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            output_model=StructuredFields,
        )


def build_extractor(settings_mode: str, config: LLMConfig) -> Extractor:
    if settings_mode == "openai":
        return LlmExtractor(config)
    from service.steps.extractor import DummyExtractor

    return DummyExtractor()
