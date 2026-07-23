import asyncio
from typing import Dict, Optional
from uuid import UUID

from service.models import DraftRecord


class InMemoryDraftStore:
    """Thread-safe in-memory draft store for local development and tests."""

    def __init__(self) -> None:
        self._drafts: Dict[UUID, DraftRecord] = {}
        self._lock = asyncio.Lock()

    async def save(self, draft: DraftRecord) -> DraftRecord:
        async with self._lock:
            self._drafts[draft.id] = draft
            return draft

    async def get(self, draft_id: UUID) -> Optional[DraftRecord]:
        async with self._lock:
            return self._drafts.get(draft_id)
