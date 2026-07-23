# Draft Service — Instructions

How to install, configure, run, and test the dermatology triage draft service locally.

**Architecture, workflow, and design decisions:** [`docs/project_overview.md`](docs/project_overview.md)  
**Eval dataset and metrics:** [`eval/README.md`](eval/README.md)

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Python 3.10–3.12

---

## Setup

```bash
uv sync
cp .env.example .env   # then edit .env
```

**Modes**

| `LLM_MODE` | Behavior |
|---|---|
| `dummy` | Keyword/template steps + local KB embeddings — no API key |
| `openai` | Real LLM calls — set `OPENAI_API_KEY` in `.env` |

See [`.env.example`](.env.example) for model and KB settings (`LLM_*`, `KB_*`).

If `uv run` fails after moving the project folder, recreate the venv:

```bash
rm -rf .venv && uv sync
```

---

## Run the API

```bash
uv run uvicorn service.main:app --reload --app-dir .
```

- Server: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`
- Health: `GET /health`

**Endpoints**

- `POST /drafts` — `{ "message": "..." }` → runs classify → extract → KB → draft → judge
- `GET /drafts/{id}` — fetch a saved draft

**Example**

```bash
curl -s -X POST http://127.0.0.1:8000/drafts \
  -H "Content-Type: application/json" \
  -d '{"message": "I need to reschedule my appointment"}' \
  | python3 -m json.tool
```

---

## Tests

```bash
uv run pytest -v
```

Skip HuggingFace network tests if offline:

```bash
uv run pytest -v --ignore=tests/test_harvest_chatdoctor.py --ignore=tests/test_hf_rows.py
```

---

## Evaluation

Classification metrics (no server required):

```bash
uv run python -m eval.run \
  --dataset data/eval/labeled.jsonl \
  --output-dir eval/results/local-run
```

API smoke test (server must be running):

```bash
uv run python -m eval.run --mode api
```

---

## Docker

```bash
docker build -t draft-service .
docker run --rm -p 8000:8000 --env-file .env draft-service
```

`.env` is not baked into the image — pass it with `--env-file` (or explicit `-e` flags).

---

## Related docs

| Doc | Contents |
|---|---|
| [`docs/project_overview.md`](docs/project_overview.md) | Structure, pipeline, decisions |
| [`eval/README.md`](eval/README.md) | Dataset harvest, labeling, eval CLI |
| [`docs/eval_dataset_methodology.md`](docs/eval_dataset_methodology.md) | Gold set methodology and baseline results |
