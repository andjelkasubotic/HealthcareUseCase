from contextlib import asynccontextmanager

from service.kb.build import build_knowledge_base
from service.kb.retriever import FaissKnowledgeBase
from service.pipeline import DraftPipeline
from service.pipeline_factory import build_pipeline
from service.settings import get_settings
from service.store.memory import InMemoryDraftStore

_draft_store = InMemoryDraftStore()
_knowledge_base: FaissKnowledgeBase | None = None
_draft_pipeline: DraftPipeline | None = None


async def startup_app_resources() -> None:
    """Initialize shared singletons used by API dependency injection."""
    global _knowledge_base, _draft_pipeline

    settings = get_settings()
    # PROTOTYPE ONLY: rebuilds the FAISS index on every app startup.
    # Production should persist the index (e.g. disk/S3), key it by FAQ file hash,
    # and rebuild only when the FAQ source changes — not on every deploy/restart.
    _knowledge_base = await build_knowledge_base(settings)
    _draft_pipeline = build_pipeline(settings=settings, knowledge_base=_knowledge_base)


def get_draft_store() -> InMemoryDraftStore:
    return _draft_store


def get_draft_pipeline() -> DraftPipeline:
    if _draft_pipeline is None:
        raise RuntimeError(
            "Draft pipeline is not initialized. FastAPI lifespan startup must run first."
        )
    return _draft_pipeline


@asynccontextmanager
async def app_lifespan(_app):
    await startup_app_resources()
    yield
