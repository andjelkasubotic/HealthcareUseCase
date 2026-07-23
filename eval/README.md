# Evaluation harness

Classify patient portal messages against a labeled JSONL dataset and write metrics reports.

## Prerequisites

```bash
cd temp_checks
uv sync
```

Set `LLM_MODE` in `.env`:

- `dummy` — keyword classifier, no API calls (good for plumbing tests)
- `openai` — real LLM classification (`OPENAI_API_KEY` required)

## Harvest ChatDoctor candidates (slice 3)

Downloads `lavita/ChatDoctor-HealthCareMagic-100k` and filters the **`input` field only**
(never `output`) for dermatology keywords. Uses the Hugging Face datasets-server API
by default (`--backend http`) so **no `datasets` package or lzma** is required.

```bash
uv sync
uv run python -m eval.data.harvest_chatdoctor
```

Defaults: 50 candidates → `data/eval/raw/chatdoctor_candidates.jsonl`.

If Hugging Face rate-limits you (HTTP 429), the harvester retries automatically with
backoff. For more headroom, set a token and slow requests slightly:

```bash
export HF_TOKEN=hf_...   # optional; higher rate limits
uv run python -m eval.data.harvest_chatdoctor \
  --request-delay 1.5 \
  --max-retries 10
```

Useful flags:

```bash
uv run python -m eval.data.harvest_chatdoctor \
  --limit 50 \
  --seed 42 \
  --max-scan 100000 \
  --output data/eval/raw/chatdoctor_candidates.jsonl
```

Offline / CI fixture mode (no Hugging Face download):

```bash
uv run python -m eval.data.harvest_chatdoctor \
  --fixture tests/fixtures/chatdoctor_harvest_input.jsonl \
  --limit 5 \
  --output data/eval/raw/chatdoctor_candidates.preview.jsonl
```

Each row includes `id`, `message`, `source`, `source_index`, and `matched_keywords`.
Labels are added in slice 4 (`label_candidates.py`) → `review_queue.jsonl`.

## Multi-model labeling (slice 4)

Label harvested candidates with **two labeler models** (defaults: `gpt-4.1` and `gpt-5.2`).
Uses a dedicated labeling prompt (`eval/data/labeling_prompts.py`), separate from production
classify prompts. Labeler models must differ from the system-under-test model.

```bash
# Dummy mode (no API)
LABELING_MODE=dummy uv run python -m eval.data.label_candidates \
  --input data/eval/raw/chatdoctor_candidates.preview.jsonl

# OpenAI mode (requires OPENAI_API_KEY)
uv run python -m eval.data.label_candidates \
  --input data/eval/raw/chatdoctor_candidates.jsonl \
  --mode openai
```

Outputs:

| File | Contents |
|------|----------|
| `data/eval/auto_labeled.jsonl` | Both labelers agree with sufficient confidence |
| `data/eval/review_queue.jsonl` | Disagreements or low-confidence agreement for human review |
| `data/eval/labeling_results.jsonl` | Full per-candidate labeling record |

Env overrides: `LLM_LABELER_A_MODEL`, `LLM_LABELER_B_MODEL`, `LABELING_MODE`.

## Build gold dataset (slice 5)

After adjudicating `review_queue.jsonl` into `review_adjudicated.jsonl`, merge all sources:

```bash
uv run python -m eval.data.build_dataset
```

Merges `auto_labeled.jsonl` + `review_adjudicated.jsonl` + `handcrafted.jsonl` → `data/eval/labeled.jsonl`.

Use `--strict` to fail if per-class minimums are not met.

## Run classify-only eval (default)

```bash
uv run python -m eval.run
```

Defaults to `data/eval/fixtures/seed.jsonl` (12 examples).

Custom dataset and output directory:

```bash
uv run python -m eval.run \
  --dataset data/eval/handcrafted.jsonl \
  --output-dir eval/results/manual-run
```

## Run API smoke test

Requires a running draft service:

```bash
uv run uvicorn service.main:app --reload --app-dir .
```

In another terminal:

```bash
uv run python -m eval.run --mode api
```

Defaults to `data/eval/fixtures/smoke.jsonl` (8 examples). Checks `GET /health`, `POST /drafts` (201), and `GET /drafts/{id}` round-trip.

Custom base URL:

```bash
uv run python -m eval.run --mode api --base-url http://127.0.0.1:8000
```

Writes `api_smoke.json` and `api_smoke.md` under the output directory. Exit code is `1` if any case fails.

## Output

Each run writes:

- `report.json` — full config, metrics, per-example predictions
- `report.md` — human-readable summary with EMERGENT recall, confusion matrix, misclassifications

Pinned config fields: `llm_mode`, classify model, `prompt_version` (from `service/steps/prompts.py`), dataset path, UTC timestamp.

## Modes

| Mode | Status | Description |
|------|--------|-------------|
| `classify-only` | **implemented** | Primary-label metrics via `build_classifier()` |
| `full-pipeline` | planned | classify + extract + draft |
| `api` | **implemented** | HTTP smoke test against running server |

## Tests

```bash
uv run pytest tests/test_eval_*.py -v
```

## Related docs

- **`docs/eval_dataset_methodology.md`** — full dataset collection, labeling methodology, reproduction steps, and baseline results
- `data/labeling_protocol.md` — gold label rules and disagreement log
- `docs/eval_decisions.md` — architecture decisions and ablation log
