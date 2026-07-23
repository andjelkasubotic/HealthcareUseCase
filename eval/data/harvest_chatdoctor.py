from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from eval.data.derm_filter import (
    MAX_MESSAGE_LENGTH,
    MIN_MESSAGE_LENGTH,
    find_matched_keywords,
    passes_derm_filter,
    preprocess_input,
)
from eval.data.hf_rows import DEFAULT_REQUEST_DELAY_SECONDS, iter_hf_rows

DEFAULT_DATASET = "lavita/ChatDoctor-HealthCareMagic-100k"
DEFAULT_BACKEND = "http"
DEFAULT_OUTPUT = Path("data/eval/raw/chatdoctor_candidates.jsonl")
DEFAULT_LIMIT = 50
DEFAULT_SEED = 42
DEFAULT_MAX_SCAN = 100_000
DEFAULT_MAX_POOL = 500


@dataclass(frozen=True)
class ChatDoctorCandidate:
    id: str
    message: str
    source: str
    source_index: int
    matched_keywords: list[str]

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "message": self.message,
            "source": self.source,
            "source_index": self.source_index,
            "matched_keywords": self.matched_keywords,
        }


def iter_dataset_records(
    dataset_name: str,
    split: str = "train",
    input_field: str = "input",
    *,
    backend: str = DEFAULT_BACKEND,
    max_scan: int = DEFAULT_MAX_SCAN,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    max_retries: int = 8,
    page_size: int = 100,
) -> Iterator[tuple[int, str]]:
    if backend == "http":
        yield from iter_hf_rows(
            dataset_name,
            split=split,
            input_field=input_field,
            max_scan=max_scan,
            page_size=page_size,
            request_delay_seconds=request_delay_seconds,
            max_retries=max_retries,
        )
        return

    if backend == "datasets":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "backend='datasets' requires the `datasets` package. "
                "Use --backend http (default) or run: uv sync --group dev"
            ) from exc
        except ModuleNotFoundError as exc:
            if exc.name == "_lzma":
                raise RuntimeError(
                    "Your Python was built without lzma support, so backend='datasets' "
                    "cannot be used. Reinstall Python with xz (brew install xz) or use "
                    "--backend http (default)."
                ) from exc
            raise

        dataset = load_dataset(dataset_name, split=split, streaming=True)
        for index, row in enumerate(dataset):
            if index >= max_scan:
                break
            raw = row.get(input_field)
            if raw is None:
                continue
            yield index, str(raw)
        return

    raise ValueError(f"Unknown backend {backend!r}. Use 'http' or 'datasets'.")


def iter_fixture_records(
    fixture_path: Path, input_field: str = "input"
) -> Iterator[tuple[int, str]]:
    for index, line in enumerate(fixture_path.read_text().splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        raw = payload.get(input_field) or payload.get("message")
        if raw is None:
            continue
        yield index, str(raw)


def harvest_candidates(
    records: Iterable[tuple[int, str]],
    *,
    limit: int = DEFAULT_LIMIT,
    seed: int = DEFAULT_SEED,
    max_scan: int = DEFAULT_MAX_SCAN,
    max_pool: int = DEFAULT_MAX_POOL,
    min_length: int = MIN_MESSAGE_LENGTH,
    max_length: int = MAX_MESSAGE_LENGTH,
) -> list[ChatDoctorCandidate]:
    pool: list[ChatDoctorCandidate] = []
    seen_messages: set[str] = set()
    scanned = 0

    for source_index, raw_text in records:
        scanned += 1
        if scanned > max_scan:
            break

        message = preprocess_input(
            raw_text, min_length=min_length, max_length=max_length
        )
        if message is None or not passes_derm_filter(message):
            continue

        dedupe_key = message.casefold()
        if dedupe_key in seen_messages:
            continue
        seen_messages.add(dedupe_key)

        pool.append(
            ChatDoctorCandidate(
                id=f"chatdoctor-{len(pool) + 1:05d}",
                message=message,
                source="chatdoctor",
                source_index=source_index,
                matched_keywords=find_matched_keywords(message),
            )
        )
        if len(pool) >= max_pool:
            break

    rng = random.Random(seed)
    if len(pool) <= limit:
        selected = pool
    else:
        selected = rng.sample(pool, limit)

    selected.sort(key=lambda candidate: candidate.source_index)
    return _renumber_candidates(selected)


def _renumber_candidates(
    candidates: list[ChatDoctorCandidate],
) -> list[ChatDoctorCandidate]:
    return [
        ChatDoctorCandidate(
            id=f"chatdoctor-{idx:05d}",
            message=candidate.message,
            source=candidate.source,
            source_index=candidate.source_index,
            matched_keywords=candidate.matched_keywords,
        )
        for idx, candidate in enumerate(candidates, start=1)
    ]


def write_candidates(path: Path, candidates: list[ChatDoctorCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(candidate.to_json(), ensure_ascii=False) for candidate in candidates
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def harvest_chatdoctor(
    *,
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    fixture_path: Path | None = None,
    backend: str = DEFAULT_BACKEND,
    limit: int = DEFAULT_LIMIT,
    seed: int = DEFAULT_SEED,
    max_scan: int = DEFAULT_MAX_SCAN,
    max_pool: int = DEFAULT_MAX_POOL,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    max_retries: int = 8,
    page_size: int = 100,
) -> list[ChatDoctorCandidate]:
    if fixture_path is not None:
        records = iter_fixture_records(fixture_path)
    else:
        records = iter_dataset_records(
            dataset_name,
            split=split,
            backend=backend,
            max_scan=max_scan,
            request_delay_seconds=request_delay_seconds,
            max_retries=max_retries,
            page_size=page_size,
        )
    return harvest_candidates(
        records,
        limit=limit,
        seed=seed,
        max_scan=max_scan,
        max_pool=max_pool,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvest dermatology-filtered ChatDoctor patient messages (input field only)."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset id (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to scan (default: train).",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Offline JSONL fixture with `input` or `message` fields (skips Hugging Face download).",
    )
    parser.add_argument(
        "--backend",
        choices=["http", "datasets"],
        default=DEFAULT_BACKEND,
        help="How to read Hugging Face data: http (default, no lzma) or datasets library.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of candidates to write.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for sampling."
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=DEFAULT_MAX_SCAN,
        help="Stop after scanning this many source rows.",
    )
    parser.add_argument(
        "--max-pool",
        type=int,
        default=DEFAULT_MAX_POOL,
        help="Maximum filtered matches to collect before sampling.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help="Seconds to wait between HF API page requests (default: 0.75).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Retries per page on HTTP 429/5xx from HF datasets-server.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Rows per HF API request (default: 100).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        candidates = harvest_chatdoctor(
            dataset_name=args.dataset,
            split=args.split,
            fixture_path=args.fixture,
            backend=args.backend,
            limit=args.limit,
            seed=args.seed,
            max_scan=args.max_scan,
            max_pool=args.max_pool,
            request_delay_seconds=args.request_delay,
            max_retries=args.max_retries,
            page_size=args.page_size,
        )
    except Exception as exc:
        print(f"Harvest failed: {exc}", file=sys.stderr)
        return 1

    write_candidates(args.output, candidates)
    print(f"Wrote {len(candidates)} candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
