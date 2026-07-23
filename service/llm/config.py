import hashlib
import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr


ProviderName = Literal["openai"]


class LLMConfig(BaseModel):
    """Connection settings for one LLM client instance."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName = "openai"
    model: str
    api_key: Optional[SecretStr] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)

    def cache_key(self) -> str:
        """Return a stable hash used to reuse one client per configuration."""
        payload = self.model_dump(mode="json", exclude_none=True)
        if payload.get("api_key") is not None:
            payload["api_key"] = "***"
        encoded = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def resolved_api_key(self) -> str:
        if self.api_key is None:
            raise ValueError("API key is required for LLM client configuration.")
        return self.api_key.get_secret_value()

    def completion_kwargs(
        self, overrides: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if overrides:
            params.update(overrides)
        return params
