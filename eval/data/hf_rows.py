"""Stream dataset rows from the Hugging Face datasets-server API (no `datasets` / lzma)."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import httpx

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_PAGE_SIZE = 100
DEFAULT_REQUEST_DELAY_SECONDS = 0.75
DEFAULT_MAX_RETRIES = 8


def _auth_headers() -> dict[str, str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _retry_delay_seconds(
    response: httpx.Response, attempt: int, base_delay: float
) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(float(retry_after), base_delay)
        except ValueError:
            pass
    return base_delay * (2**attempt)


def fetch_hf_rows_page(
    client: httpx.Client,
    *,
    dataset_name: str,
    split: str,
    config: str,
    offset: int,
    length: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    """Fetch one page from datasets-server, retrying on rate limits and transient errors."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        response = client.get(
            DATASETS_SERVER_URL,
            params={
                "dataset": dataset_name,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
            headers=_auth_headers(),
        )

        if response.status_code in {429, 500, 502, 503, 504}:
            last_error = httpx.HTTPStatusError(
                f"HF datasets-server returned {response.status_code}",
                request=response.request,
                response=response,
            )
            if attempt < max_retries:
                delay = _retry_delay_seconds(response, attempt, base_delay_seconds)
                time.sleep(delay)
                continue
            response.raise_for_status()

        response.raise_for_status()
        return response.json()

    if last_error is not None:
        raise last_error
    raise RuntimeError("fetch_hf_rows_page failed without a response")


def iter_hf_rows(
    dataset_name: str,
    *,
    split: str = "train",
    config: str = "default",
    input_field: str = "input",
    max_scan: int | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout_seconds: float = 60.0,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Iterator[tuple[int, str]]:
    """Yield (row_idx, input_text) pairs from a HF dataset via the public rows API."""
    offset = 0
    scanned = 0

    with httpx.Client(timeout=timeout_seconds) as client:
        while True:
            if max_scan is not None and scanned >= max_scan:
                return

            length = page_size
            if max_scan is not None:
                length = min(page_size, max_scan - scanned)

            payload = fetch_hf_rows_page(
                client,
                dataset_name=dataset_name,
                split=split,
                config=config,
                offset=offset,
                length=length,
                max_retries=max_retries,
            )
            rows: list[dict[str, Any]] = payload.get("rows", [])
            if not rows:
                return

            for item in rows:
                if max_scan is not None and scanned >= max_scan:
                    return
                row = item.get("row", {})
                raw = row.get(input_field)
                scanned += 1
                if raw is None:
                    continue
                source_index = int(item.get("row_idx", offset))
                yield source_index, str(raw)

            offset += len(rows)

            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
