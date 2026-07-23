from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from service.llm.config import LLMConfig
from service.llm.protocol import LLMCompletionResult

T = TypeVar("T", bound=BaseModel)


class OpenAILLMClient:
    """Async OpenAI chat client for a single model configuration."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.resolved_api_key(),
            timeout=config.timeout_seconds,
        )

    @classmethod
    def from_config(cls, config: LLMConfig) -> "OpenAILLMClient":
        return cls(config)

    @property
    def config(self) -> LLMConfig:
        return self._config

    async def acompletion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMCompletionResult:
        params = self._config.completion_kwargs(kwargs)
        response = await self._client.chat.completions.create(
            messages=messages,
            **params,
        )
        content = response.choices[0].message.content or ""
        return LLMCompletionResult(content=content)

    async def aparse(
        self,
        messages: list[dict[str, str]],
        output_model: type[T],
        **kwargs: Any,
    ) -> T:
        params = self._config.completion_kwargs(kwargs)
        response = await self._client.chat.completions.parse(
            messages=messages,
            response_format=output_model,
            **params,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned no parsed structured output.")
        return parsed
