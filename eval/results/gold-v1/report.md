# Classification eval report

## Config

| Setting | Value |
|---------|-------|
| Mode | `classify-only` |
| Dataset | `data/eval/labeled.jsonl` |
| LLM mode | `openai` |
| Classify model | `gpt-4.1-mini` |
| Prompt version | `v1` |
| Started at | `2026-07-18T17:10:48.415050+00:00` |
| Examples | 75 |

## Headline metrics

- **EMERGENT recall:** 1.000
- **Macro F1:** 0.672
- **Macro precision:** 0.740
- **Macro recall:** 0.676

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| emergent | 0.625 | 1.000 | 0.769 | 10 |
| urgent | 0.667 | 0.722 | 0.693 | 36 |
| routine | 0.667 | 0.480 | 0.558 | 25 |
| admin | 1.000 | 0.500 | 0.667 | 4 |

## Confusion matrix

| true \ pred | emergent | urgent | routine | admin |
|---|---|---|---|---|
| emergent | 10 | 0 | 0 | 0 |
| urgent | 6 | 26 | 4 | 0 |
| routine | 0 | 13 | 12 | 0 |
| admin | 0 | 0 | 2 | 2 |

## Misclassifications

- `chatdoctor-00003`: gold `routine` → `urgent` (all: urgent, routine)
- `chatdoctor-00005`: gold `routine` → `urgent` (all: urgent, routine)
- `chatdoctor-00010`: gold `routine` → `urgent` (all: urgent, routine)
- `chatdoctor-00014`: gold `urgent` → `emergent` (all: emergent, urgent)
- `chatdoctor-00018`: gold `urgent` → `routine` (all: routine)
- `chatdoctor-00021`: gold `urgent` → `emergent` (all: emergent)
- `chatdoctor-00022`: gold `routine` → `urgent` (all: urgent, routine)
- `chatdoctor-00023`: gold `urgent` → `routine` (all: routine)
- `chatdoctor-00025`: gold `routine` → `urgent` (all: urgent)
- `chatdoctor-00027`: gold `urgent` → `routine` (all: routine)
- `chatdoctor-00030`: gold `routine` → `urgent` (all: urgent)
- `chatdoctor-00034`: gold `routine` → `urgent` (all: urgent)
- `chatdoctor-00035`: gold `routine` → `urgent` (all: urgent)
- `chatdoctor-00037`: gold `routine` → `urgent` (all: urgent, routine, admin)
- `chatdoctor-00039`: gold `urgent` → `emergent` (all: emergent)
- `chatdoctor-00040`: gold `routine` → `urgent` (all: urgent, routine)
- `chatdoctor-00041`: gold `urgent` → `routine` (all: routine)
- `chatdoctor-00042`: gold `routine` → `urgent` (all: urgent)
- `chatdoctor-00045`: gold `routine` → `urgent` (all: urgent, routine)
- `chatdoctor-00048`: gold `urgent` → `emergent` (all: emergent, urgent)
- `hc-005`: gold `urgent` → `emergent` (all: emergent)
- `hc-014`: gold `urgent` → `emergent` (all: emergent, admin)
- `hc-019`: gold `admin` → `routine` (all: routine, admin)
- `hc-022`: gold `routine` → `urgent` (all: urgent)
- `hc-023`: gold `admin` → `routine` (all: routine, admin)

## Sklearn classification report

```
              precision    recall  f1-score   support

    emergent       0.62      1.00      0.77        10
      urgent       0.67      0.72      0.69        36
     routine       0.67      0.48      0.56        25
       admin       1.00      0.50      0.67         4

    accuracy                           0.67        75
   macro avg       0.74      0.68      0.67        75
weighted avg       0.68      0.67      0.66        75
```
