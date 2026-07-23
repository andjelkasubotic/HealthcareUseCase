# Labeling protocol — dermatology triage eval dataset

> **Full methodology and runbook:** see [`docs/eval_dataset_methodology.md`](../docs/eval_dataset_methodology.md) for how examples were collected, labeled, merged, and evaluated.

This document defines how gold labels are assigned for the healthcare dermatology
triage prototype. Update the **Disagreement log** at the end as labeling proceeds.

## Scope

- **Task:** assign one **primary triage label** per patient portal message.
- **Domain:** dermatology clinic (portal messages, not live chat).
- **Output:** JSONL rows consumed by `eval/dataset.py` (see `eval/schemas.py`).

## Class definitions

| Label | Definition | Dermatology examples |
|-------|------------|----------------------|
| `emergent` | Possible life-threatening or same-day emergency care. Err toward emergent when in doubt. | Anaphylaxis, angioedema, Stevens–Johnson/TEN suspicion, rapidly spreading rash with fever, post-op wound with pus + fever, mucosal involvement with systemic symptoms |
| `urgent` | Clinically concerning; needs prompt (often 24–72h) clinician review, not immediate ER. | Changing mole, severe uncontrolled flare, widespread painful rash without systemic red flags, suspected cellulitis without sepsis, new bullae |
| `routine` | Non-urgent clinical question or mild symptoms manageable in routine visit. | Mild eczema itch, moisturizer question, routine follow-up, stable chronic condition |
| `admin` | No clinical triage needed; scheduling, billing, insurance, records. | Reschedule, copay question, portal login — **unless** a clinically urgent/emergent complaint is also present |

## Mixed-intent messages

Patients often combine clinical and admin content in one message.

1. **Gold label = highest clinical priority** present (emergent > urgent > routine > admin).
2. The system may predict **multiple** `classifications`; eval compares **primary** labels via `primary_classification()`.
3. Document ambiguous cases in `label_rationale` (required for `ambiguous` tag).

## Labeler workflow

1. **Independent labeling:** two models (e.g. gpt-4.1 and gpt-5.2), neither the system-under-test.
2. **Agreement:** both agree → provisional label.
3. **Disagreement:** queue for human adjudication; record outcome in Disagreement log.
4. **Freeze:** once `labeled.jsonl` is versioned, changes require a dataset version bump.

## Leakage and quality controls

- Do not use ChatDoctor `output` field as labels (model-generated, not gold).
- Hold out at least 10% of gold rows for final reporting; do not tune prompts on hold-out labels.
- Synthetic perturbations require human approval before merge.
- No real PHI; handcrafted and synthetic messages only in committed fixtures until harvest is reviewed.

## Per-class minimums (full gold set)

Target **50–100** total rows (aim ~80):

| Class | Minimum |
|-------|---------|
| emergent | 10 |
| urgent | 8 |
| routine | 8 |
| admin | 8 |

Fixtures (`seed.jsonl`, `handcrafted.jsonl`) may be smaller; `build_dataset.py` enforces minimums on `labeled.jsonl`.

## Row schema

```json
{
  "id": "hc-001",
  "message": "Patient portal text…",
  "label": "urgent",
  "source": "handcrafted",
  "tags": ["adversarial"],
  "label_rationale": "Required when tags include ambiguous"
}
```

## Disagreement log

| id | labeler_a | labeler_b | adjudicated | notes |
|----|-----------|-----------|-------------|-------|
| chatdoctor-00004 | routine | urgent | urgent | Genital/spreading rash; chronic but needs prompt derm eval |
| chatdoctor-00005 | routine | urgent | routine | BC mood side effects; non-derm, no acute safety flags |
| chatdoctor-00009 | routine | urgent | urgent | New painless genital lesions |
| chatdoctor-00012 | routine | urgent | urgent | Infant HFM with severe pain |
| chatdoctor-00014 | urgent | emergent | urgent | Rectal bleeding + perianal rash; no sepsis described |
| chatdoctor-00021 | urgent | emergent | urgent | Worsening post-op purulent drainage |
| chatdoctor-00022 | routine | urgent | routine | Scheduled pulmonary biopsy anxiety |
| chatdoctor-00023 | routine | urgent | urgent | Severe psoriasis flare failing therapy |
| chatdoctor-00025 | routine | urgent | routine | Oral gum lesion; dental routing |
| chatdoctor-00026 | urgent | emergent | emergent | Acute bilateral renal symptoms |
| chatdoctor-00034 | routine | urgent | routine | Late period, negative HPT |
| chatdoctor-00035 | routine | urgent | routine | Benign-appearing twitching |
| chatdoctor-00041 | routine | urgent | urgent | Large anogenital lesion |
| chatdoctor-00042 | routine | routine | routine | Agreed; low confidence blocked auto-accept |
| chatdoctor-00046 | routine | urgent | urgent | Acute widespread pruritic eruption |
| chatdoctor-00049 | routine | urgent | urgent | Painful phimosis-like symptoms dominate admin questions |
