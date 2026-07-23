# Classification eval report

## Config

| Setting | Value |
|---------|-------|
| Mode | `classify-only` |
| Dataset | `data/eval/fixtures/seed.jsonl` |
| LLM mode | `openai` |
| Classify model | `gpt-4.1-mini` |
| Prompt version | `v1` |
| Started at | `2026-07-18T15:47:07.401912+00:00` |
| Examples | 12 |

## Headline metrics

- **EMERGENT recall:** 0.667
- **Macro F1:** 0.914
- **Macro precision:** 0.938
- **Macro recall:** 0.917

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| emergent | 1.000 | 0.667 | 0.800 | 3 |
| urgent | 0.750 | 1.000 | 0.857 | 3 |
| routine | 1.000 | 1.000 | 1.000 | 3 |
| admin | 1.000 | 1.000 | 1.000 | 3 |

## Confusion matrix

| true \ pred | emergent | urgent | routine | admin |
|---|---|---|---|---|
| emergent | 2 | 1 | 0 | 0 |
| urgent | 0 | 3 | 0 | 0 |
| routine | 0 | 0 | 3 | 0 |
| admin | 0 | 0 | 0 | 3 |

## Misclassifications

- `seed-002`: gold `emergent` → `urgent` (all: urgent)

## Sklearn classification report

```
              precision    recall  f1-score   support

    emergent       1.00      0.67      0.80         3
      urgent       0.75      1.00      0.86         3
     routine       1.00      1.00      1.00         3
       admin       1.00      1.00      1.00         3

    accuracy                           0.92        12
   macro avg       0.94      0.92      0.91        12
weighted avg       0.94      0.92      0.91        12
```
