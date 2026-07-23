from typing import Protocol

from service.models import Classification


class Classifier(Protocol):
    async def classify(self, message: str) -> list[Classification]: ...


class DummyClassifier:
    """Placeholder classifier — keyword heuristics until a real model is wired."""

    async def classify(self, message: str) -> list[Classification]:
        lowered = message.lower()
        categories: list[Classification] = []
        if "emergency" in lowered or "emergent" in lowered:
            categories.append(Classification.EMERGENT)
        if "urgent" in lowered:
            categories.append(Classification.URGENT)
        if "admin" in lowered or "billing" in lowered:
            categories.append(Classification.ADMIN)
        if not categories:
            categories.append(Classification.ROUTINE)
        return categories
