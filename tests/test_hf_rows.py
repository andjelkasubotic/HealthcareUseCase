from unittest.mock import MagicMock, patch

import httpx
import pytest

from eval.data.hf_rows import DATASETS_SERVER_URL, iter_hf_rows


def test_iter_hf_rows_retries_on_rate_limit() -> None:
    rate_limited = httpx.Response(
        429,
        request=httpx.Request("GET", DATASETS_SERVER_URL),
        headers={"Retry-After": "0"},
    )
    success = httpx.Response(
        200,
        request=httpx.Request("GET", DATASETS_SERVER_URL),
        json={"rows": [{"row_idx": 0, "row": {"input": "itchy rash"}}]},
    )
    empty = httpx.Response(
        200,
        request=httpx.Request("GET", DATASETS_SERVER_URL),
        json={"rows": []},
    )

    mock_client = MagicMock()
    mock_client.get.side_effect = [rate_limited, success, empty]
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with (
        patch("eval.data.hf_rows.httpx.Client", return_value=mock_client),
        patch("eval.data.hf_rows.time.sleep"),
    ):
        rows = list(
            iter_hf_rows(
                "lavita/ChatDoctor-HealthCareMagic-100k",
                max_scan=5,
                page_size=1,
                request_delay_seconds=0,
            )
        )

    assert rows == [(0, "itchy rash")]
    assert mock_client.get.call_count == 3


def test_iter_hf_rows_paginates_and_yields_input_field() -> None:
    page_one = MagicMock()
    page_one.raise_for_status = MagicMock()
    page_one.json.return_value = {
        "rows": [
            {"row_idx": 0, "row": {"input": "rash on arm", "output": "ignored"}},
            {"row_idx": 1, "row": {"input": "billing only", "output": "ignored"}},
        ]
    }
    page_two = MagicMock()
    page_two.raise_for_status = MagicMock()
    page_two.json.return_value = {"rows": []}

    mock_client = MagicMock()
    mock_client.get.side_effect = [page_one, page_two]
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with (
        patch("eval.data.hf_rows.httpx.Client", return_value=mock_client),
        patch("eval.data.hf_rows.time.sleep"),
    ):
        rows = list(
            iter_hf_rows(
                "lavita/ChatDoctor-HealthCareMagic-100k",
                max_scan=10,
                page_size=2,
                request_delay_seconds=0,
            )
        )

    assert rows == [(0, "rash on arm"), (1, "billing only")]
    assert mock_client.get.call_count == 2


@pytest.mark.integration
def test_iter_hf_rows_live_request() -> None:
    rows = list(
        iter_hf_rows(
            "lavita/ChatDoctor-HealthCareMagic-100k",
            max_scan=3,
            page_size=3,
            timeout_seconds=120.0,
            request_delay_seconds=1.0,
        )
    )
    assert len(rows) == 3
    assert all(isinstance(text, str) and text for _idx, text in rows)
