from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from service.llm.config import LLMConfig

T = TypeVar("T", bound=BaseModel)


class LLMCompletionResult(BaseModel):
    content: str


@runtime_checkable
class LLMClient(Protocol):
    """Transport layer for a single provider configuration."""

    @property
    def config(self) -> LLMConfig: ...

    async def acompletion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMCompletionResult: ...

    async def aparse(
        self,
        messages: list[dict[str, str]],
        output_model: type[T],
        **kwargs: Any,
    ) -> T: ...
