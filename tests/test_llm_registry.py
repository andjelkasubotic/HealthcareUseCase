import pytest

from service.llm.config import LLMConfig
from service.llm.registry import clear_client_cache, get_llm_client
from service.llm.providers.openai import OpenAILLMClient
from pydantic import SecretStr


@pytest.fixture(autouse=True)
def clear_llm_client_cache() -> None:
    clear_client_cache()
    yield
    clear_client_cache()


def test_get_llm_client_returns_cached_openai_client() -> None:
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=SecretStr("test-key"),
    )

    first = get_llm_client(config)
    second = get_llm_client(config)

    assert isinstance(first, OpenAILLMClient)
    assert first is second


def test_get_llm_client_rejects_unknown_provider() -> None:
    config = LLMConfig.model_construct(
        provider="anthropic",
        model="gpt-4o-mini",
        api_key=SecretStr("test-key"),
        temperature=0.0,
        max_tokens=None,
        timeout_seconds=60.0,
    )

    with pytest.raises(ValueError, match="Unsupported provider"):
        get_llm_client(config)
