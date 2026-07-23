from service.models import Classification, StructuredFields


def compute_abstraction_flag(
    classifications: list[Classification],
    fields: StructuredFields,
) -> bool:
    """Return whether the case should be abstracted for triage review."""
    if Classification.EMERGENT in classifications:
        return True
    if Classification.URGENT in classifications and fields.severity == "severe":
        return True
    return False
