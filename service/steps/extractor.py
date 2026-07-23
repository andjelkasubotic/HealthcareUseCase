from typing import Protocol

from service.models import StructuredFields


class Extractor(Protocol):
    async def extract(self, message: str) -> StructuredFields: ...


class DummyExtractor:
    """Placeholder extractor — returns static fields until an LLM is wired."""

    async def extract(self, message: str) -> StructuredFields:
        return StructuredFields(
            symptoms="unspecified symptom",
            duration="unknown",
            body_location="unspecified",
            severity="mild",
            onset="unknown",
        )
