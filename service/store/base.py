from typing import Optional, Protocol
from uuid import UUID

from service.models import DraftRecord


class DraftStore(Protocol):
    async def save(self, draft: DraftRecord) -> DraftRecord: ...

    async def get(self, draft_id: UUID) -> Optional[DraftRecord]: ...
