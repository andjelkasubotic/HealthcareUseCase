from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Classification(str, Enum):
    EMERGENT = "emergent"
    URGENT = "urgent"
    ROUTINE = "routine"
    ADMIN = "admin"


_CLASSIFICATION_PRIORITY: tuple[Classification, ...] = (
    Classification.EMERGENT,
    Classification.URGENT,
    Classification.ROUTINE,
    Classification.ADMIN,
)


def primary_classification(classifications: list[Classification]) -> Classification:
    """Return the highest-priority category for triage routing."""
    for category in _CLASSIFICATION_PRIORITY:
        if category in classifications:
            return category
    return classifications[0]


class StructuredFields(BaseModel):
    symptoms: Optional[str] = None
    duration: Optional[str] = None
    body_location: Optional[str] = None
    severity: Optional[str] = None
    onset: Optional[str] = None


class CreateDraftRequest(BaseModel):
    """POST /drafts request body. Invalid input returns HTTP 422 before the handler runs."""

    message: str = Field(..., min_length=1)


class DraftStatus(str, Enum):
    READY = "ready"
    FAILED = "failed"


class DraftJudgeResult(BaseModel):
    approved: bool
    violations: list[str] = Field(default_factory=list)
    reason: str = ""
    was_rewritten: bool = False
    original_draft: Optional[str] = None
    final_draft: str


class RetrievedSource(BaseModel):
    id: str
    title: str
    score: float


class DraftRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    input_message: str
    classifications: list[Classification] = Field(..., min_length=1)
    structured_fields: StructuredFields
    abstraction_flag: bool
    draft_response: str
    retrieved_sources: list[RetrievedSource] = Field(default_factory=list)
    judge_result: DraftJudgeResult
    needs_human_review: bool
    human_review_reason: Optional[str] = None
    status: DraftStatus = DraftStatus.READY
    created_at: datetime = Field(default_factory=datetime.utcnow)
