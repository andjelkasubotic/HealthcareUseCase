import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

from service.llm.config import LLMConfig

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE)

LLMMode = Literal["dummy", "openai"]


class TaskLLMSettings(BaseModel):
    """Per-step LLM options (model + temperature) from .env; validated before API calls."""

    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class AppSettings(BaseModel):
    """Typed app configuration: bridges .env strings to objects used by pipeline_factory."""

    llm_mode: LLMMode = "dummy"
    openai_api_key: SecretStr | None = None
    classify: TaskLLMSettings = TaskLLMSettings(model="gpt-4.1-mini", temperature=0.0)
    extract: TaskLLMSettings = TaskLLMSettings(model="gpt-4.1", temperature=0.0)
    draft: TaskLLMSettings = TaskLLMSettings(model="gpt-5.1", temperature=0.3)
    judge: TaskLLMSettings = TaskLLMSettings(model="gpt-5.2", temperature=0.0)
    embed: TaskLLMSettings = TaskLLMSettings(model="text-embedding-3-small", temperature=0.0)
    kb_faq_path: str = "data/kb/dermatology_faq.txt"
    kb_top_k: int = Field(default=3, ge=1, le=10)

    def task_config(
        self, task: Literal["classify", "extract", "draft", "judge"]
    ) -> LLMConfig:
        """Build LLMConfig for one pipeline step (passed to build_classifier/extractor/drafter)."""
        task_settings = getattr(self, task)
        return LLMConfig(
            provider="openai",
            model=task_settings.model,
            api_key=self.openai_api_key,
            temperature=task_settings.temperature,
        )


@lru_cache
def get_settings() -> AppSettings:
    """Load app config from .env once (cached); used when building the pipeline at startup."""
    return AppSettings(
        llm_mode=_read_llm_mode(),
        openai_api_key=_read_secret("OPENAI_API_KEY"),
        classify=TaskLLMSettings(
            model=os.getenv("LLM_CLASSIFY_MODEL", "gpt-4.1-mini"),
            temperature=float(os.getenv("LLM_CLASSIFY_TEMPERATURE", "0")),
        ),
        extract=TaskLLMSettings(
            model=os.getenv("LLM_EXTRACT_MODEL", "gpt-4.1"),
            temperature=float(os.getenv("LLM_EXTRACT_TEMPERATURE", "0")),
        ),
        draft=TaskLLMSettings(
            model=os.getenv("LLM_DRAFT_MODEL", "gpt-5.1"),
            temperature=float(os.getenv("LLM_DRAFT_TEMPERATURE", "0.3")),
        ),
        judge=TaskLLMSettings(
            model=os.getenv("LLM_JUDGE_MODEL", "gpt-5.2"),
            temperature=float(os.getenv("LLM_JUDGE_TEMPERATURE", "0")),
        ),
        embed=TaskLLMSettings(
            model=os.getenv("LLM_EMBED_MODEL", "text-embedding-3-small"),
            temperature=0.0,
        ),
        kb_faq_path=os.getenv("KB_FAQ_PATH", "data/kb/dermatology_faq.txt"),
        kb_top_k=int(os.getenv("KB_TOP_K", "3")),
    )


def _read_llm_mode() -> LLMMode:
    """Read LLM_MODE from .env: dummy (no API) or openai (real classify/extract/draft)."""
    value = os.getenv("LLM_MODE", "dummy").lower()
    if value in {"dummy", "openai"}:
        return value  # type: ignore[return-value]
    raise ValueError("LLM_MODE must be 'dummy' or 'openai'.")


def _read_secret(name: str) -> SecretStr | None:
    value = os.getenv(name)
    if value:
        return SecretStr(value)
    return None
