from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from eval.api_client import DraftApiClient
from eval.api_smoke import run_api_smoke
from eval.report import format_api_smoke_markdown, write_api_smoke_reports
from eval.schemas import ApiSmokeRunConfig, ApiSmokeRunResult
from service.dependencies import get_draft_pipeline, get_draft_store
from service.main import create_app
from service.pipeline import DraftPipeline
from service.store.memory import InMemoryDraftStore

SMOKE_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "eval"
    / "fixtures"
    / "smoke.jsonl"
)


@pytest_asyncio.fixture
async def api_client():
    store = InMemoryDraftStore()
    pipeline = DraftPipeline()
    app = create_app()
    app.dependency_overrides[get_draft_store] = lambda: store
    app.dependency_overrides[get_draft_pipeline] = lambda: pipeline

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield DraftApiClient(client)

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_client_round_trip(api_client: DraftApiClient) -> None:
    assert await api_client.check_health() == 200

    create_status, get_status, draft, error = await api_client.round_trip(
        "I have a routine question about follow-up care"
    )

    assert error is None
    assert create_status == 201
    assert get_status == 200
    assert draft is not None
    assert draft.input_message == "I have a routine question about follow-up care"


@pytest.mark.asyncio
async def test_run_api_smoke_on_fixture(api_client: DraftApiClient) -> None:
    result = await run_api_smoke(
        SMOKE_FIXTURE,
        api_client,
        base_url="http://test",
    )

    assert result.health_status == 200
    assert len(result.cases) == 8
    assert result.all_passed
    assert result.n_failed == 0


def test_smoke_fixture_loads() -> None:
    from eval.dataset import load_jsonl

    examples = load_jsonl(SMOKE_FIXTURE)
    assert len(examples) == 8


def test_write_api_smoke_reports(tmp_path) -> None:
    result = ApiSmokeRunResult(
        config=ApiSmokeRunConfig(
            dataset_path="data/eval/fixtures/smoke.jsonl",
            base_url="http://test",
        ),
        health_status=200,
        cases=[],
    )

    json_path, md_path = write_api_smoke_reports(result, tmp_path / "out")

    assert json_path.exists()
    assert md_path.exists()
    assert "API smoke test report" in format_api_smoke_markdown(result)
