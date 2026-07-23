from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from eval.data.candidate_io import load_candidates, write_jsonl
from eval.data.labeling_prompts import LABELING_PROMPT_VERSION, LABELING_SYSTEM_PROMPT
from eval.schemas import LabelingDecision, LabelingResult, UnlabeledCandidate
from service.llm.config import LLMConfig
from service.llm.registry import get_llm_client
from service.llm.structured import complete_structured
from service.models import Classification
from service.settings import get_settings

DEFAULT_INPUT = Path("data/eval/raw/chatdoctor_candidates.jsonl")
DEFAULT_AUTO_OUTPUT = Path("data/eval/auto_labeled.jsonl")
DEFAULT_REVIEW_OUTPUT = Path("data/eval/review_queue.jsonl")
DEFAULT_RESULTS_OUTPUT = Path("data/eval/labeling_results.jsonl")
DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_LABELER_A_MODEL = "gpt-4.1"
DEFAULT_LABELER_B_MODEL = "gpt-5.2"


class _LabelingResponse(BaseModel):
    label: Classification
    rationale: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Labeler(Protocol):
    model_name: str

    async def label(self, message: str) -> LabelingDecision: ...


@dataclass(frozen=True)
class LabelerPair:
    labeler_a: Labeler
    labeler_b: Labeler
    labeler_a_model: str
    labeler_b_model: str


class LlmLabeler:
    def __init__(self, config: LLMConfig, model_name: str) -> None:
        self.model_name = model_name
        self._config = config
        self._client = get_llm_client(config)

    async def label(self, message: str) -> LabelingDecision:
        result = await complete_structured(
            self._client,
            messages=[
                {"role": "system", "content": LABELING_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            output_model=_LabelingResponse,
        )
        return LabelingDecision(
            label=result.label,
            rationale=result.rationale,
            confidence=result.confidence,
        )


class DummyLabelerA:
    """Keyword labeler A — conservative on admin keywords."""

    model_name = "dummy-labeler-a"

    async def label(self, message: str) -> LabelingDecision:
        label = _dummy_label(message, prefer_admin=True)
        return LabelingDecision(
            label=label,
            rationale=f"Dummy labeler A selected {label.value}.",
            confidence=0.95,
        )


class DummyLabelerB:
    """Keyword labeler B — down-weights standalone admin cues."""

    model_name = "dummy-labeler-b"

    async def label(self, message: str) -> LabelingDecision:
        label = _dummy_label(message, prefer_admin=False)
        return LabelingDecision(
            label=label,
            rationale=f"Dummy labeler B selected {label.value}.",
            confidence=0.9,
        )


def _dummy_label(message: str, *, prefer_admin: bool) -> Classification:
    lowered = message.lower()
    if any(
        token in lowered
        for token in ("swelling", "shortness of breath", "fever", "emergency", "chills")
    ):
        return Classification.EMERGENT
    if any(
        token in lowered
        for token in ("mole", "blister", "cellulitis", "hives", "flare")
    ):
        return Classification.URGENT
    if prefer_admin and any(
        token in lowered for token in ("billing", "copay", "appointment")
    ):
        return Classification.ADMIN
    if any(token in lowered for token in ("moisturizer", "retinoid", "eczema", "itch")):
        return Classification.ROUTINE
    if any(
        token in lowered for token in ("billing", "copay", "appointment", "insurance")
    ):
        return Classification.ADMIN
    return Classification.ROUTINE


def should_auto_accept(
    decision_a: LabelingDecision,
    decision_b: LabelingDecision,
    *,
    min_confidence: float,
) -> bool:
    return (
        decision_a.label == decision_b.label
        and decision_a.confidence >= min_confidence
        and decision_b.confidence >= min_confidence
    )


def build_labeler_pair(
    *,
    mode: str,
    labeler_a_model: str,
    labeler_b_model: str,
) -> LabelerPair:
    if mode == "dummy":
        return LabelerPair(
            labeler_a=DummyLabelerA(),
            labeler_b=DummyLabelerB(),
            labeler_a_model=DummyLabelerA.model_name,
            labeler_b_model=DummyLabelerB.model_name,
        )

    settings = get_settings()
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required when labeling mode is openai.")

    api_key = settings.openai_api_key
    return LabelerPair(
        labeler_a=LlmLabeler(
            LLMConfig(
                provider="openai",
                model=labeler_a_model,
                api_key=api_key,
                temperature=0.0,
            ),
            labeler_a_model,
        ),
        labeler_b=LlmLabeler(
            LLMConfig(
                provider="openai",
                model=labeler_b_model,
                api_key=api_key,
                temperature=0.0,
            ),
            labeler_b_model,
        ),
        labeler_a_model=labeler_a_model,
        labeler_b_model=labeler_b_model,
    )


async def label_one_candidate(
    candidate: UnlabeledCandidate,
    pair: LabelerPair,
    *,
    min_confidence: float,
) -> LabelingResult:
    decision_a = await pair.labeler_a.label(candidate.message)
    decision_b = await pair.labeler_b.label(candidate.message)
    auto_accepted = should_auto_accept(
        decision_a, decision_b, min_confidence=min_confidence
    )

    return LabelingResult(
        id=candidate.id,
        message=candidate.message,
        source=candidate.source,
        labeler_a_model=pair.labeler_a_model,
        labeler_b_model=pair.labeler_b_model,
        label_a=decision_a.label,
        label_b=decision_b.label,
        rationale_a=decision_a.rationale,
        rationale_b=decision_b.rationale,
        confidence_a=decision_a.confidence,
        confidence_b=decision_b.confidence,
        agree=decision_a.label == decision_b.label,
        auto_accepted=auto_accepted,
        label=decision_a.label if auto_accepted else None,
        label_method="multi_model_agreement" if auto_accepted else None,
    )


async def label_candidates(
    candidates: list[UnlabeledCandidate],
    pair: LabelerPair,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[LabelingResult]:
    results: list[LabelingResult] = []
    for candidate in candidates:
        results.append(
            await label_one_candidate(candidate, pair, min_confidence=min_confidence)
        )
    return results


def split_results(
    results: list[LabelingResult],
) -> tuple[list[dict], list[dict], list[dict]]:
    auto_labeled: list[dict] = []
    review_queue: list[dict] = []
    full_results: list[dict] = []

    for result in results:
        payload = result.model_dump(mode="json")
        full_results.append(payload)
        if result.auto_accepted and result.label is not None:
            auto_labeled.append(
                {
                    "id": result.id,
                    "message": result.message,
                    "label": result.label.value,
                    "source": result.source,
                    "label_method": result.label_method,
                    "labeler_a_model": result.labeler_a_model,
                    "labeler_b_model": result.labeler_b_model,
                    "labeling_prompt_version": LABELING_PROMPT_VERSION,
                }
            )
        else:
            review_queue.append(
                {
                    "id": result.id,
                    "message": result.message,
                    "source": result.source,
                    "status": "needs_review",
                    "labeler_a_model": result.labeler_a_model,
                    "labeler_b_model": result.labeler_b_model,
                    "label_a": result.label_a.value,
                    "label_b": result.label_b.value,
                    "rationale_a": result.rationale_a,
                    "rationale_b": result.rationale_b,
                    "confidence_a": result.confidence_a,
                    "confidence_b": result.confidence_b,
                    "agree": result.agree,
                    "labeling_prompt_version": LABELING_PROMPT_VERSION,
                }
            )

    return auto_labeled, review_queue, full_results


def resolve_labeling_mode(cli_mode: str | None) -> str:
    if cli_mode is not None:
        return cli_mode
    return os.getenv("LABELING_MODE", get_settings().llm_mode)


def resolve_labeler_models() -> tuple[str, str]:
    return (
        os.getenv("LLM_LABELER_A_MODEL", DEFAULT_LABELER_A_MODEL),
        os.getenv("LLM_LABELER_B_MODEL", DEFAULT_LABELER_B_MODEL),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label candidate messages with two models; route disagreements to review queue."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--auto-output", type=Path, default=DEFAULT_AUTO_OUTPUT)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=["dummy", "openai"],
        default=None,
        help="Labeling backend (default: LABELING_MODE env or LLM_MODE).",
    )
    parser.add_argument(
        "--labeler-a-model",
        default=None,
        help=f"Model for labeler A (default: {DEFAULT_LABELER_A_MODEL}).",
    )
    parser.add_argument(
        "--labeler-b-model",
        default=None,
        help=f"Model for labeler B (default: {DEFAULT_LABELER_B_MODEL}).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum confidence from both labelers to auto-accept agreement.",
    )
    return parser.parse_args(argv)


async def run_labeling(args: argparse.Namespace) -> list[LabelingResult]:
    candidates = load_candidates(args.input)
    if not candidates:
        raise ValueError(f"No candidates found in {args.input}")

    mode = resolve_labeling_mode(args.mode)
    default_a, default_b = resolve_labeler_models()
    labeler_a_model = args.labeler_a_model or default_a
    labeler_b_model = args.labeler_b_model or default_b

    pair = build_labeler_pair(
        mode=mode,
        labeler_a_model=labeler_a_model,
        labeler_b_model=labeler_b_model,
    )
    return await label_candidates(
        candidates,
        pair,
        min_confidence=args.min_confidence,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        print(
            "Run harvest first:\n"
            "  uv run python -m eval.data.harvest_chatdoctor "
            "--output data/eval/raw/chatdoctor_candidates.jsonl",
            file=sys.stderr,
        )
        return 1

    try:
        results = asyncio.run(run_labeling(args))
    except Exception as exc:
        print(f"Labeling failed: {exc}", file=sys.stderr)
        return 1

    auto_labeled, review_queue, full_results = split_results(results)
    write_jsonl(args.auto_output, auto_labeled)
    write_jsonl(args.review_output, review_queue)
    write_jsonl(args.results_output, full_results)

    n_auto = len(auto_labeled)
    n_review = len(review_queue)
    print(
        f"Labeled {len(results)} candidates "
        f"(auto-accepted={n_auto}, review_queue={n_review})"
    )
    print(f"Wrote {args.auto_output}")
    print(f"Wrote {args.review_output}")
    print(f"Wrote {args.results_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
