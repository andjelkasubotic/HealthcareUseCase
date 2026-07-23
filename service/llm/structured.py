from typing import TypeVar

from pydantic import BaseModel

from service.llm.protocol import LLMClient

T = TypeVar("T", bound=BaseModel)


async def complete_structured(
    client: LLMClient,
    messages: list[dict[str, str]],
    output_model: type[T],
    **kwargs: object,
) -> T:
    """Return a Pydantic model via the provider's structured-output API (OpenAI json_schema)."""
    return await client.aparse(messages, output_model=output_model, **kwargs)
