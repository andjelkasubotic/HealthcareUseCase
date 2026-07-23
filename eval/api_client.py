from uuid import UUID

import httpx

from service.models import DraftRecord


class DraftApiClient:
    """HTTP client for draft service smoke tests."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    @classmethod
    def from_base_url(cls, base_url: str, timeout: float = 60.0) -> "DraftApiClient":
        return cls(httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def check_health(self) -> int:
        response = await self._client.get("/health")
        return response.status_code

    async def create_draft(self, message: str) -> tuple[int, DraftRecord | None]:
        response = await self._client.post("/drafts", json={"message": message})
        if response.status_code != 201:
            return response.status_code, None
        return response.status_code, DraftRecord.model_validate(response.json())

    async def get_draft(self, draft_id: UUID | str) -> tuple[int, DraftRecord | None]:
        response = await self._client.get(f"/drafts/{draft_id}")
        if response.status_code != 200:
            return response.status_code, None
        return response.status_code, DraftRecord.model_validate(response.json())

    async def round_trip(
        self, message: str
    ) -> tuple[int, int, DraftRecord | None, str | None]:
        """POST /drafts then GET /drafts/{id}. Returns (create_status, get_status, draft, error)."""
        create_status, created = await self.create_draft(message)
        if create_status != 201 or created is None:
            return create_status, 0, None, f"POST /drafts returned {create_status}"

        get_status, fetched = await self.get_draft(created.id)
        if get_status != 200 or fetched is None:
            return (
                create_status,
                get_status,
                None,
                f"GET /drafts/{created.id} returned {get_status}",
            )

        if fetched.id != created.id:
            return (
                create_status,
                get_status,
                fetched,
                f"draft id mismatch: created {created.id}, fetched {fetched.id}",
            )

        if fetched.input_message != message:
            return (
                create_status,
                get_status,
                fetched,
                "round-trip input_message does not match request",
            )

        return create_status, get_status, fetched, None
