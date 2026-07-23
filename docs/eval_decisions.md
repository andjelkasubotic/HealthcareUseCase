# Eval decisions (ADR log)

Decisions for the dermatology triage evaluation harness. Add an entry per ablation run.

## 1. Eval entry point

**Decision:** Primary metrics use direct `DraftPipeline` invocation (`classify-only`, `full-pipeline`). HTTP is limited to `--mode api` smoke tests on a small fixture set.

**Rationale:** Pipeline calls are faster, deterministic for unit tests, and avoid server lifecycle in CI. API mode validates wiring only.

## 2. Dataset sources

**Decision:** ChatDoctor dermatology-filtered inputs + handcrafted adversarial/ambiguous rows + optional human-approved synthetic perturbations. No PHI in committed data.

**Rationale:** ChatDoctor scale; handcrafted covers failure modes and mixed intent.

## 3. Labeling

**Decision:** Two labeler models (neither system-under-test) with human adjudication on disagreement.

**Rationale:** Reduces single-model bias; documented in `data/labeling_protocol.md`.

## 4. Leakage controls

**Decision:** Hold-out subset for final reporting; no prompt tuning on gold labels; never use ChatDoctor `output` as labels.

## 5. Primary metric

**Decision:** **EMERGENT recall** is the safety-critical headline metric (asymmetric cost of missed emergencies).

**Supporting metrics:** Per-class P/R/F1, macro F1, confusion matrix.

## 6. Target size

**Decision:** 50–100 total examples (aim ~80). Per-class minimums: emergent ≥10, others ≥8 each.

## 7. Rejected alternatives

| Alternative | Why rejected |
|-------------|--------------|
| HTTP-only eval for all examples | Too slow; couples metrics to server |
| Full 100-example API smoke | Unnecessary for classification metrics |
| ChatDoctor `output` as gold | Model-generated, not clinician labels |



