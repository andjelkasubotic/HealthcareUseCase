import sys
from pathlib import Path

TEMP_CHECKS_ROOT = Path(__file__).resolve().parent.parent
if str(TEMP_CHECKS_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMP_CHECKS_ROOT))

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from service.dependencies import get_draft_pipeline, get_draft_store
from service.main import create_app
from service.pipeline import DraftPipeline
from service.store.memory import InMemoryDraftStore


@pytest.fixture
def isolated_store() -> InMemoryDraftStore:
    return InMemoryDraftStore()


@pytest.fixture
def isolated_pipeline() -> DraftPipeline:
    return DraftPipeline()


@pytest_asyncio.fixture
async def client(isolated_store: InMemoryDraftStore, isolated_pipeline: DraftPipeline):
    app = create_app()
    app.dependency_overrides[get_draft_store] = lambda: isolated_store
    app.dependency_overrides[get_draft_pipeline] = lambda: isolated_pipeline

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client

    app.dependency_overrides.clear()
