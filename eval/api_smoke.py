import asyncio
from datetime import datetime, timezone
from pathlib import Path

from eval.api_client import DraftApiClient
from eval.dataset import load_jsonl
from eval.schemas import ApiSmokeCaseResult, ApiSmokeRunConfig, ApiSmokeRunResult


def build_api_smoke_config(
    dataset_path: Path | str,
    base_url: str,
) -> ApiSmokeRunConfig:
    return ApiSmokeRunConfig(
        dataset_path=str(Path(dataset_path)),
        base_url=base_url.rstrip("/"),
        started_at=datetime.now(timezone.utc),
    )


async def run_api_smoke(
    dataset_path: Path | str,
    client: DraftApiClient,
    base_url: str | None = None,
) -> ApiSmokeRunResult:
    examples = load_jsonl(dataset_path)
    if not examples:
        raise ValueError(f"No examples found in {dataset_path}")

    resolved_base_url = base_url or str(client._client.base_url)
    config = build_api_smoke_config(dataset_path, resolved_base_url)

    health_status = await client.check_health()
    cases: list[ApiSmokeCaseResult] = []

    for example in examples:
        if health_status != 200:
            cases.append(
                ApiSmokeCaseResult(
                    id=example.id,
                    message=example.message,
                    ok=False,
                    error=f"/health returned {health_status}",
                )
            )
            continue

        create_status, get_status, _draft, error = await client.round_trip(
            example.message
        )
        cases.append(
            ApiSmokeCaseResult(
                id=example.id,
                message=example.message,
                create_status=create_status,
                get_status=get_status if get_status else None,
                draft_id=str(_draft.id) if _draft is not None else None,
                ok=error is None,
                error=error,
            )
        )

    return ApiSmokeRunResult(
        config=config,
        health_status=health_status,
        cases=cases,
    )


def run_api_smoke_sync(
    dataset_path: Path | str,
    base_url: str,
) -> ApiSmokeRunResult:
    async def _run() -> ApiSmokeRunResult:
        client = DraftApiClient.from_base_url(base_url)
        try:
            return await run_api_smoke(dataset_path, client, base_url=base_url)
        finally:
            await client.aclose()

    return asyncio.run(_run())
