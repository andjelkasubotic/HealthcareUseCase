from service.llm.config import LLMConfig, ProviderName
from service.llm.protocol import LLMClient
from service.llm.providers.openai import OpenAILLMClient

_CLIENT_CACHE: dict[str, LLMClient] = {}


def get_llm_client(config: LLMConfig) -> LLMClient:
    """Return a cached client for the given configuration."""
    cache_key = config.cache_key()
    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = _build_client(config)
    return _CLIENT_CACHE[cache_key]


def clear_client_cache() -> None:
    _CLIENT_CACHE.clear()


def _build_client(config: LLMConfig) -> LLMClient:
    builders = {
        "openai": OpenAILLMClient.from_config,
    }
    builder = builders.get(config.provider)
    if builder is None:
        supported: list[ProviderName] = ["openai"]
        raise ValueError(
            f"Unsupported provider '{config.provider}'. Supported: {supported}"
        )
    return builder(config)
