from typing import Protocol

from service.models import Classification, StructuredFields, primary_classification


class Drafter(Protocol):
    async def draft(
        self,
        message: str,
        classifications: list[Classification],
        fields: StructuredFields,
        abstraction_flag: bool,
        faq_context: str = "",
    ) -> str: ...


class DummyDrafter:
    """Placeholder drafter — template response until an LLM is wired."""

    async def draft(
        self,
        message: str,
        classifications: list[Classification],
        fields: StructuredFields,
        abstraction_flag: bool,
        faq_context: str = "",
    ) -> str:
        abstraction_note = (
            " This case has been flagged for clinical review."
            if abstraction_flag
            else ""
        )
        primary = primary_classification(classifications)
        faq_note = ""
        if faq_context and "No FAQ context retrieved." not in faq_context:
            faq_note = " We referenced clinic FAQ guidance when preparing this draft."
        return (
            f"Thank you for your message. We have classified this as "
            f"{primary.value}. Based on the reported "
            f"{fields.symptoms} affecting {fields.body_location} for "
            f"{fields.duration}, we will follow up shortly.{faq_note}{abstraction_note}"
        )
