from __future__ import annotations

import json
from pathlib import Path

from eval.schemas import UnlabeledCandidate


def load_candidates(path: Path | str) -> list[UnlabeledCandidate]:
    file_path = Path(path)
    candidates: list[UnlabeledCandidate] = []
    for line_number, line in enumerate(file_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
            candidates.append(UnlabeledCandidate.model_validate(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"{file_path}:{line_number}: invalid candidate row — {exc}"
            ) from exc
    return candidates


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
