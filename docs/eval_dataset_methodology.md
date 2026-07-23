# Evaluation dataset — methodology and runbook

This document describes **how the dermatology triage evaluation set was built**, the **labeling methodology**, and **how to reproduce** collection, labeling, merging, and evaluation.

**Gold file:** `data/eval/labeled.jsonl` (75 examples, version 1 — July 2026)

Related references:
- `data/labeling_protocol.md` — class definitions and disagreement log
- `docs/eval_decisions.md` — architecture decisions and ablation log
- `eval/README.md` — eval harness command reference

---

## 1. Purpose

We evaluate the **primary triage classification** of patient portal messages for a dermatology clinic prototype. Each example has one gold label:

| Label | Meaning |
|-------|---------|
| `emergent` | Possible life-threatening or same-day emergency |
| `urgent` | Clinically concerning; needs prompt review (often 24–72h) |
| `routine` | Mild symptoms or general follow-up questions |
| `admin` | Scheduling, billing, insurance, records only |

The live classifier may return **multiple** categories; metrics compare the **primary** predicted label (highest clinical priority) to the single gold label.

**Headline metric:** **EMERGENT recall** (missing an emergency is costlier than a false alarm).

---

## 2. Dataset composition (v1)

| Source | Rows | Label method | Role |
|--------|------|--------------|------|
| ChatDoctor harvest → auto-labeled | 34 | Two-model agreement (`gpt-4.1` + `gpt-5.2`) | Bulk real patient language |
| ChatDoctor harvest → human adjudicated | 16 | Human review of labeler disagreements | Quality control on hard cases |
| Handcrafted adversarial + ambiguous | 25 | Agent-drafted, human-reviewed | Edge cases, mixed intent, negation traps |
| **Total** | **75** | | |

### Per-class counts (v1)

| Class | Count | Target minimum |
|-------|-------|----------------|
| emergent | 10 | 10 ✓ |
| urgent | 36 | 8 ✓ |
| routine | 25 | 8 ✓ |
| admin | 4 | 8 ✗ |

**Known gap:** `admin` is under the target minimum because ChatDoctor dermatology filtering yields mostly clinical messages. Add 4+ admin-only portal messages before treating the set as fully balanced.

### Source files merged into gold

```
data/eval/auto_labeled.jsonl          (34 rows)
data/eval/review_adjudicated.jsonl    (16 rows)
data/eval/handcrafted.jsonl           (25 rows)
        ↓  build_dataset.py
data/eval/labeled.jsonl               (75 rows — gold)
```

---

## 3. Collection methodology

### 3.1 ChatDoctor harvest

**Source:** [lavita/ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) on Hugging Face.

**Field used:** `input` (patient message) **only**. The doctor `output` field is never used as a label (it is model-generated, not gold).

**Filter pipeline** (`eval/data/harvest_chatdoctor.py` + `eval/data/derm_filter.py`):

1. Stream rows via Hugging Face **datasets-server HTTP API** (default `--backend http`; no `lzma` / `datasets` required).
2. Normalize whitespace; drop messages shorter than 20 chars; truncate at 1500 chars.
3. Keep rows matching dermatology keywords (e.g. rash, skin, mole, eczema, hives, blister, itch).
4. Deduplicate by normalized message text.
5. Collect up to 500 matches while scanning up to 100,000 source rows.
6. Random sample **50** candidates (seed `42`) for labeling.

**Output:** `data/eval/raw/chatdoctor_candidates.jsonl`

Each candidate row:

```json
{
  "id": "chatdoctor-00001",
  "message": "...",
  "source": "chatdoctor",
  "source_index": 1234,
  "matched_keywords": ["rash", "itchy"]
}
```

### 3.2 Handcrafted examples

**File:** `data/eval/handcrafted.jsonl` (25 rows)

| Subset | Count | Purpose |
|--------|-------|---------|
| Adversarial | 15 | Buried red flags, negation traps, poor grammar, admin + clinical mix |
| Ambiguous | 10 | Borderline cases with required `label_rationale` |

Authored to mirror dermatology clinic portal patterns from the use case. Labels reviewed against `data/labeling_protocol.md`.

### 3.3 Development fixtures (not in gold)

| File | Rows | Use |
|------|------|-----|
| `data/eval/fixtures/seed.jsonl` | 12 | Early harness development |
| `data/eval/fixtures/smoke.jsonl` | 8 | API smoke tests |

---

## 4. Labeling methodology

### 4.1 Principles

1. **One primary gold label** per message; clinical risk dominates in mixed-intent messages.
2. **Two independent labeler models** — neither is the system-under-test.
3. **Human adjudication** on disagreement or low-confidence agreement.
4. **Separate labeling prompt** from production classify prompt (`eval/data/labeling_prompts.py`, version `v1`).

### 4.2 Models used

| Role | Model | Notes |
|------|-------|-------|
| Labeler A | `gpt-4.1` | `LLM_LABELER_A_MODEL` env override |
| Labeler B | `gpt-5.2` | `LLM_LABELER_B_MODEL` env override |
| System under test | `gpt-4.1-mini` | `LLM_CLASSIFY_MODEL` in `.env` |

Labeler models must differ from the classifier under evaluation to reduce labeling leakage.

### 4.3 Auto-labeling (`eval/data/label_candidates.py`)

For each harvested candidate:

1. Both labelers return `{ label, rationale, confidence }` via structured output.
2. **Auto-accept** if labels match **and** both confidences ≥ `0.7`.
3. Otherwise → `review_queue.jsonl` for human adjudication.

**Outputs:**

| File | Contents |
|------|----------|
| `data/eval/auto_labeled.jsonl` | 34 agreed labels |
| `data/eval/review_queue.jsonl` | 16 disagreements (raw labeler outputs) |
| `data/eval/labeling_results.jsonl` | Full audit trail for all 50 candidates |

### 4.4 Human adjudication

All 16 review-queue rows were adjudicated into `data/eval/review_adjudicated.jsonl` using the protocol in `data/labeling_protocol.md`. Decisions are logged in the **Disagreement log** table in that file.

Example adjudicated row:

```json
{
  "id": "chatdoctor-00014",
  "message": "...",
  "label": "urgent",
  "source": "chatdoctor",
  "tags": ["adjudicated"],
  "label_method": "human",
  "label_rationale": "Perianal/vulvar rash plus rectal bleeding..."
}
```

### 4.5 Gold merge (`eval/data/build_dataset.py`)

Merges the three labeled sources, checks unique IDs, validates per-class minimums, writes `data/eval/labeled.jsonl`.

Gold row schema (required fields for eval):

```json
{
  "id": "chatdoctor-00001",
  "message": "patient portal text",
  "label": "urgent",
  "source": "chatdoctor",
  "tags": [],
  "label_rationale": null
}
```

---

## 5. How to reproduce (full pipeline)

All commands from the `temp_checks` directory:

```bash
cd temp_checks
uv sync
```

### Environment

```bash
# .env
LLM_MODE=openai
OPENAI_API_KEY=sk-...
LLM_CLASSIFY_MODEL=gpt-4.1-mini   # system under test

# Optional for harvest rate limits
export HF_TOKEN=hf_...
```

### Step 1 — Harvest candidates

```bash
uv run python -m eval.data.harvest_chatdoctor \
  --limit 50 \
  --seed 42 \
  --output data/eval/raw/chatdoctor_candidates.jsonl \
  --request-delay 1.5 \
  --max-retries 10
```

Offline test (no Hugging Face):

```bash
uv run python -m eval.data.harvest_chatdoctor \
  --fixture tests/fixtures/chatdoctor_harvest_input.jsonl \
  --limit 5
```

### Step 2 — Multi-model labeling

```bash
uv run python -m eval.data.label_candidates \
  --input data/eval/raw/chatdoctor_candidates.jsonl \
  --mode openai
```

Dummy mode (no API):

```bash
LABELING_MODE=dummy uv run python -m eval.data.label_candidates \
  --input data/eval/raw/chatdoctor_candidates.preview.jsonl
```

### Step 3 — Adjudicate review queue

Manually review `data/eval/review_queue.jsonl` and write decisions to `data/eval/review_adjudicated.jsonl` (one row per disagreement with final `label` and `label_rationale`).

### Step 4 — Merge gold set

```bash
uv run python -m eval.data.build_dataset \
  --output data/eval/labeled.jsonl
```

Use `--strict` to fail if per-class minimums are not met.

### Step 5 — Run classification eval

```bash
uv run python -m eval.run \
  --dataset data/eval/labeled.jsonl \
  --mode classify-only \
  --output-dir eval/results/gold-v1
```

Reports: `report.json` + `report.md` with pinned config (model, prompt version, timestamp).

### Step 6 — API smoke test (optional)

Terminal 1:

```bash
uv run uvicorn service.main:app --reload --app-dir .
```

Terminal 2:

```bash
uv run python -m eval.run --mode api
```

---

## 6. Evaluation methodology

### What is measured

- **Primary comparison:** gold `label` vs `primary_classification(predicted classifications)`
- **Metrics:** per-class precision/recall/F1, macro F1, confusion matrix
- **Safety metric:** EMERGENT recall

### What is not measured yet

- Extraction field accuracy
- Draft quality / guardrails
- Full-pipeline end-to-end (`--mode full-pipeline` — planned)

### Eval implementation

| Component | Path |
|-----------|------|
| Dataset loader | `eval/dataset.py` |
| Metrics | `eval/metrics.py` |
| Runner | `eval/runner.py` |
| CLI | `eval/run.py` |
| Production classify prompt | `service/steps/prompts.py` (`PROMPT_VERSION=v1`) |

Eval calls `build_classifier()` directly (not HTTP) for classification metrics.

---

## 7. Baseline results (gold v1)

**Run:** `eval/results/gold-v1/`  
**Config:** `gpt-4.1-mini`, `LLM_MODE=openai`, `prompt_version=v1`, 75 examples

| Metric | Value |
|--------|-------|
| **EMERGENT recall** | **1.00** (10/10) |
| Macro F1 | 0.67 |
| Accuracy | 0.67 |
| emergent F1 | 0.77 |
| urgent F1 | 0.69 |
| routine F1 | 0.56 |
| admin F1 | 0.67 |

**Confusion matrix (rows=true, cols=predicted):**

| true \\ pred | emergent | urgent | routine | admin |
|--------------|----------|--------|---------|-------|
| emergent | 10 | 0 | 0 | 0 |
| urgent | 6 | 26 | 4 | 0 |
| routine | 0 | 13 | 12 | 0 |
| admin | 0 | 0 | 2 | 2 |

**Interpretation:** Emergent cases are never missed (recall 1.0), but 6 urgent cases are over-triaged to emergent. Routine recall is moderate (0.48); many routine messages are classified as urgent. Admin support is thin (n=4).

---

## 8. Limitations and next steps

1. **Admin class under-represented** — add 4+ admin-only portal messages to meet minimums.
2. **ChatDoctor noise** — harvested messages are general HealthCareMagic queries, not all dermatology-specific; keyword filter is approximate.
3. **No hold-out split yet** — full 75 rows used for reporting; reserve a hold-out before prompt tuning.
4. **Single eval run** — plan calls for 3-run variance and ablations (see `docs/eval_decisions.md`).
5. **Labeler ≠ clinician** — gold labels are LLM + human adjudication, not board-certified clinician annotation.

---

## 9. File inventory

```
data/
  labeling_protocol.md              # Class rules + disagreement log
  eval/
    labeled.jsonl                   # ★ Gold evaluation set (75 rows)
    auto_labeled.jsonl              # ChatDoctor auto-accepted labels
    review_queue.jsonl              # Raw disagreements from labelers
    review_adjudicated.jsonl        # Human-finalized disagreements
    labeling_results.jsonl          # Full labeling audit trail
    handcrafted.jsonl               # Adversarial + ambiguous examples
    raw/
      chatdoctor_candidates.jsonl   # Harvested unlabeled pool (50)
    fixtures/
      seed.jsonl                    # Dev fixture (12)
      smoke.jsonl                   # API smoke fixture (8)

eval/
  README.md                         # Command quick reference
  run.py                            # Eval CLI
  data/
    harvest_chatdoctor.py
    label_candidates.py
    build_dataset.py
    labeling_prompts.py
    derm_filter.py
    hf_rows.py

docs/
  eval_decisions.md                 # ADR log
  eval_dataset_methodology.md       # ★ This document

eval/results/gold-v1/
  report.json
  report.md
```

---

## 10. Tests

```bash
uv run pytest tests/test_eval_*.py tests/test_harvest_chatdoctor.py tests/test_label_candidates.py tests/test_build_dataset.py -v
```

---

*Document version: 1.0 — matches gold dataset v1 (`labeled.jsonl`, 75 rows, July 2026).*
