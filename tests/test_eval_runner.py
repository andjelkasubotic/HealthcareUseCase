import pytest

from eval.report import format_report_markdown, write_reports
from eval.runner import run_classify_eval
from eval.schemas import EvalExample, EvalPrediction, EvalRunConfig, EvalRunResult
from service.models import Classification
from service.settings import AppSettings


class SequenceClassifier:
    def __init__(self, labels: list[list[Classification]]) -> None:
        self._labels = iter(labels)

    async def classify(self, message: str) -> list[Classification]:
        return next(self._labels)


@pytest.mark.asyncio
async def test_run_classify_eval_computes_metrics() -> None:
    examples = [
        EvalExample(
            id="t-001",
            message="emergency swelling",
            label=Classification.EMERGENT,
            source="test",
        ),
        EvalExample(
            id="t-002",
            message="billing question",
            label=Classification.ADMIN,
            source="test",
        ),
    ]
    classifier = SequenceClassifier(
        [
            [Classification.EMERGENT],
            [Classification.ROUTINE],
        ]
    )

    from eval.runner import classify_examples

    predictions = await classify_examples(examples, classifier)

    assert predictions[0].correct is True
    assert predictions[1].correct is False
    assert predictions[1].predicted == Classification.ROUTINE


@pytest.mark.asyncio
async def test_run_classify_eval_from_fixture(tmp_path) -> None:
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text(
        '{"id":"t-001","message":"routine check","label":"routine","source":"test"}\n'
    )

    settings = AppSettings(llm_mode="dummy")
    result = await run_classify_eval(dataset, settings=settings)

    assert result.metrics["n_examples"] == 1
    assert result.config.mode == "classify-only"
    assert result.config.prompt_version == "v1"


def test_write_reports_creates_json_and_markdown(tmp_path) -> None:
    result = EvalRunResult(
        config=EvalRunConfig(
            mode="classify-only",
            dataset_path="data/eval/fixtures/seed.jsonl",
            llm_mode="dummy",
            classify_model="gpt-4.1-mini",
            prompt_version="v1",
        ),
        metrics={
            "n_examples": 1,
            "emergent_recall": 1.0,
            "macro": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "per_class": {
                label.value: {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "support": 0,
                }
                for label in Classification
            },
            "confusion_matrix": {
                "labels": [c.value for c in Classification],
                "matrix": [[0, 0, 0, 0] for _ in Classification],
            },
            "classification_report": "ok",
        },
        predictions=[],
    )

    json_path, md_path = write_reports(result, tmp_path / "out")

    assert json_path.exists()
    assert md_path.exists()
    markdown = md_path.read_text()
    assert "EMERGENT recall" in markdown
    assert "`v1`" in markdown


def test_format_report_markdown_lists_misclassifications() -> None:
    result = EvalRunResult(
        config=EvalRunConfig(
            mode="classify-only",
            dataset_path="x.jsonl",
            llm_mode="dummy",
            classify_model="gpt-4.1-mini",
            prompt_version="v1",
        ),
        metrics={
            "n_examples": 1,
            "emergent_recall": 0.0,
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "per_class": {
                label.value: {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0,
                }
                for label in Classification
            },
            "confusion_matrix": {
                "labels": [c.value for c in Classification],
                "matrix": [[0, 0, 0, 0] for _ in Classification],
            },
            "classification_report": "ok",
        },
        predictions=[
            EvalPrediction(
                id="bad-1",
                label=Classification.EMERGENT,
                predicted=Classification.ROUTINE,
                predicted_all=[Classification.ROUTINE],
                correct=False,
            )
        ],
    )

    markdown = format_report_markdown(result)
    assert "Misclassifications" in markdown
    assert "bad-1" in markdown
