from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from service.models import Classification


EvalMode = Literal["classify-only", "full-pipeline", "api"]


class EvalExample(BaseModel):
    """One labeled row in an eval JSONL file."""

    id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    label: Classification
    source: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    label_rationale: Optional[str] = None

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(tag) for tag in value]
        raise TypeError("tags must be a list of strings")

    @model_validator(mode="after")
    def _require_rationale_for_ambiguous(self) -> "EvalExample":
        if "ambiguous" in self.tags and not self.label_rationale:
            raise ValueError(
                f"Example {self.id!r}: label_rationale required when tags include 'ambiguous'"
            )
        return self


# Default per-class minimums for the full gold dataset (see data/labeling_protocol.md).
DEFAULT_CLASS_MINIMUMS: dict[Classification, int] = {
    Classification.EMERGENT: 10,
    Classification.URGENT: 8,
    Classification.ROUTINE: 8,
    Classification.ADMIN: 8,
}


class EvalPrediction(BaseModel):
    id: str
    label: Classification
    predicted: Classification
    predicted_all: list[Classification]
    correct: bool


class EvalRunConfig(BaseModel):
    mode: EvalMode = "classify-only"
    dataset_path: str
    llm_mode: str
    classify_model: str
    prompt_version: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvalRunResult(BaseModel):
    config: EvalRunConfig
    metrics: dict
    predictions: list[EvalPrediction]


class ApiSmokeRunConfig(BaseModel):
    mode: Literal["api"] = "api"
    dataset_path: str
    base_url: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ApiSmokeCaseResult(BaseModel):
    id: str
    message: str
    create_status: int | None = None
    get_status: int | None = None
    draft_id: str | None = None
    ok: bool
    error: str | None = None


class ApiSmokeRunResult(BaseModel):
    config: ApiSmokeRunConfig
    health_status: int | None = None
    cases: list[ApiSmokeCaseResult]

    @property
    def n_passed(self) -> int:
        return sum(1 for case in self.cases if case.ok)

    @property
    def n_failed(self) -> int:
        return sum(1 for case in self.cases if not case.ok)

    @property
    def all_passed(self) -> bool:
        return self.health_status == 200 and self.n_failed == 0


class UnlabeledCandidate(BaseModel):
    """Harvested message awaiting labeling."""

    id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    source_index: int | None = None
    matched_keywords: list[str] = Field(default_factory=list)


class LabelingDecision(BaseModel):
    label: Classification
    rationale: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class LabelingResult(BaseModel):
    id: str
    message: str
    source: str
    labeler_a_model: str
    labeler_b_model: str
    label_a: Classification
    label_b: Classification
    rationale_a: str
    rationale_b: str
    confidence_a: float
    confidence_b: float
    agree: bool
    auto_accepted: bool
    label: Classification | None = None
    label_method: str | None = None
