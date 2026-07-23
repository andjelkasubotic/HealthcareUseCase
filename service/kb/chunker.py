from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field


class FaqChunk(BaseModel):
    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


_SECTION_PATTERN = re.compile(
    r"^##\s+(?P<id>[\w-]+)\s+—\s+(?P<title>.+?)\s*$",
    re.MULTILINE,
)


def load_faq_chunks(path: Path | str) -> list[FaqChunk]:
    """Parse FAQ sections marked with '## faq-id — Title' headers."""
    raw = Path(path).read_text(encoding="utf-8")
    matches = list(_SECTION_PATTERN.finditer(raw))
    if not matches:
        raise ValueError(f"No FAQ sections found in {path}")

    chunks: list[FaqChunk] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        if not body:
            raise ValueError(f"FAQ section {match.group('id')!r} has empty body")
        chunks.append(
            FaqChunk(
                id=match.group("id"),
                title=match.group("title").strip(),
                text=body,
            )
        )
    return chunks
