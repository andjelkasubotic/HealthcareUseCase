from service.models import Classification, StructuredFields

PROMPT_VERSION = "v1"

CLASSIFY_SYSTEM_PROMPT = (
    "You triage patient inquiries for a healthcare app. A message may match more "
    "than one category (for example, symptoms plus a billing question). Return "
    "all applicable categories, ordered by clinical priority (most severe first):\n"
    "- emergent: possible life-threatening symptoms or needs immediate care\n"
    "- urgent: concerning symptoms that need prompt attention, not immediate emergency\n"
    "- routine: general health questions, mild symptoms, or follow-up care\n"
    "- admin: billing, appointments, insurance, or other non-clinical requests"
)

EXTRACT_SYSTEM_PROMPT = (
    "Extract symptoms, duration, body_location, severity, and onset from the "
    "user message. Use null when unknown."
)

DRAFT_SYSTEM_PROMPT = (
    "You draft replies for a dermatology clinic patient portal. Write a concise, "
    "polite, and professional response. Be empathetic and clear. "
    "Ground your reply in the provided FAQ context when relevant — use approved "
    "clinic language for scheduling, response times, and escalation. "
    "Do not provide a medical diagnosis or specific treatment advice. "
    "The FAQ is policy guidance, not a clinical reference."
)

JUDGE_CHECK_SYSTEM_PROMPT = (
    "You review clinician draft replies for a dermatology patient portal. "
    "The draft is NOT sent directly to the patient; it is for clinician review only. "
    "Set approved=true only if the draft is safe and appropriate. "
    "Set approved=false if it contains ANY of:\n"
    "- A medical diagnosis or named condition asserted as fact (e.g. 'you have cellulitis')\n"
    "- Medication or dosage changes (start/stop/increase/decrease/switch)\n"
    "- Definitive treatment instructions (specific creams, antibiotics, steroids, schedules)\n"
    "- Advice that bypasses clinician review\n"
    "Approve empathetic acknowledgments, scheduling language, and escalation to clinical review. "
    "For emergent cases, escalation-oriented language is required and should be approved. "
    "Return violation codes from: diagnosis_statement, medication_change, specific_dosage, "
    "definitive_treatment, bypasses_clinician. "
    "If rejected, give a short reason."
)

JUDGE_REWRITE_SYSTEM_PROMPT = (
    "You rewrite unsafe clinician draft replies for a dermatology patient portal. "
    "The original draft failed safety review. Produce a corrected draft that:\n"
    "- Does NOT diagnose or name a condition as fact\n"
    "- Does NOT change medication or give dosage instructions\n"
    "- Does NOT give definitive treatment advice\n"
    "- Acknowledges the patient empathetically\n"
    "- Directs appropriate next steps (clinician review, escalation if emergent, scheduling if admin)\n"
    "Return only the rewritten draft text, with no preamble or explanation."
)


def format_draft_user_prompt(
    message: str,
    classifications: list[Classification],
    fields: StructuredFields,
    abstraction_flag: bool,
    faq_context: str = "",
) -> str:
    labels = ", ".join(c.value for c in classifications)
    return (
        f"Message: {message}\n"
        f"Classifications: {labels}\n"
        f"Structured fields: {fields.model_dump()}\n"
        f"Abstraction flag: {abstraction_flag}\n\n"
        f"FAQ context:\n{faq_context or 'No FAQ context retrieved.'}"
    )


def format_judge_user_prompt(
    message: str,
    draft_response: str,
    classifications: list[Classification],
    fields: StructuredFields,
) -> str:
    labels = ", ".join(c.value for c in classifications)
    return (
        f"Patient message: {message}\n"
        f"Triage classifications: {labels}\n"
        f"Structured fields: {fields.model_dump()}\n"
        f"Draft to review:\n{draft_response}"
    )


def format_judge_rewrite_user_prompt(
    message: str,
    draft_response: str,
    classifications: list[Classification],
    fields: StructuredFields,
    rejection_reason: str,
    violations: list[str],
) -> str:
    labels = ", ".join(c.value for c in classifications)
    violation_text = ", ".join(violations) if violations else "unspecified"
    return (
        f"Patient message: {message}\n"
        f"Triage classifications: {labels}\n"
        f"Structured fields: {fields.model_dump()}\n"
        f"Safety violations: {violation_text}\n"
        f"Rejection reason: {rejection_reason}\n"
        f"Original draft to rewrite:\n{draft_response}"
    )
