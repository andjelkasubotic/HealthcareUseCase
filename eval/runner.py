import asyncio
from datetime import datetime, timezone
from pathlib import Path

from service.models import Classification, primary_classification
from service.settings import AppSettings, get_settings
from service.steps.classifier import Classifier
from service.steps.llm_classifier import build_classifier
from service.steps.prompts import PROMPT_VERSION

from eval.dataset import load_jsonl
from eval.metrics import compute_classification_metrics
from eval.schemas import (
    EvalExample,
    EvalMode,
    EvalPrediction,
    EvalRunConfig,
    EvalRunResult,
)


def build_eval_config(
    dataset_path: Path | str,
    mode: EvalMode = "classify-only",
    settings: AppSettings | None = None,
) -> EvalRunConfig:
    app_settings = settings or get_settings()
    return EvalRunConfig(
        mode=mode,
        dataset_path=str(Path(dataset_path)),
        llm_mode=app_settings.llm_mode,
        classify_model=app_settings.classify.model,
        prompt_version=PROMPT_VERSION,
        started_at=datetime.now(timezone.utc),
    )


def build_eval_classifier(settings: AppSettings | None = None) -> Classifier:
    app_settings = settings or get_settings()
    if app_settings.llm_mode == "openai" and app_settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required when LLM_MODE=openai.")
    return build_classifier(app_settings.llm_mode, app_settings.task_config("classify"))


async def classify_examples(
    examples: list[EvalExample],
    classifier: Classifier,
) -> list[EvalPrediction]:
    predictions: list[EvalPrediction] = []
    for example in examples:
        predicted_all = await classifier.classify(example.message)
        predicted = primary_classification(predicted_all)
        predictions.append(
            EvalPrediction(
                id=example.id,
                label=example.label,
                predicted=predicted,
                predicted_all=predicted_all,
                correct=predicted == example.label,
            )
        )
    return predictions


async def run_classify_eval(
    dataset_path: Path | str,
    classifier: Classifier | None = None,
    settings: AppSettings | None = None,
) -> EvalRunResult:
    examples = load_jsonl(dataset_path)
    if not examples:
        raise ValueError(f"No examples found in {dataset_path}")

    app_settings = settings or get_settings()
    active_classifier = classifier or build_eval_classifier(app_settings)
    config = build_eval_config(
        dataset_path, mode="classify-only", settings=app_settings
    )

    predictions = await classify_examples(examples, active_classifier)
    y_true = [prediction.label for prediction in predictions]
    y_pred = [prediction.predicted for prediction in predictions]
    metrics = compute_classification_metrics(y_true, y_pred)

    return EvalRunResult(
        config=config,
        metrics=metrics,
        predictions=predictions,
    )


def run_classify_eval_sync(
    dataset_path: Path | str,
    classifier: Classifier | None = None,
    settings: AppSettings | None = None,
) -> EvalRunResult:
    return asyncio.run(
        run_classify_eval(
            dataset_path,
            classifier=classifier,
            settings=settings,
        )
    )
