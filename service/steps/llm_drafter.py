from service.llm.config import LLMConfig
from service.llm.protocol import LLMClient
from service.llm.registry import get_llm_client
from service.models import Classification, StructuredFields
from service.steps.drafter import Drafter
from service.steps.prompts import DRAFT_SYSTEM_PROMPT, format_draft_user_prompt


class LlmDrafter:
    """Draft a response with an OpenAI model."""

    def __init__(self, config: LLMConfig) -> None:
        self._client: LLMClient = get_llm_client(config)

    async def draft(
        self,
        message: str,
        classifications: list[Classification],
        fields: StructuredFields,
        abstraction_flag: bool,
        faq_context: str = "",
    ) -> str:
        response = await self._client.acompletion(
            messages=[
                {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": format_draft_user_prompt(
                        message,
                        classifications,
                        fields,
                        abstraction_flag,
                        faq_context=faq_context,
                    ),
                },
            ],
        )
        return response.content.strip()


def build_drafter(settings_mode: str, config: LLMConfig) -> Drafter:
    if settings_mode == "openai":
        return LlmDrafter(config)
    from service.steps.drafter import DummyDrafter

    return DummyDrafter()
