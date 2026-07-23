import json
from datetime import datetime, timezone
from pathlib import Path

from eval.schemas import ApiSmokeRunResult, EvalRunResult


def default_output_dir(base: Path | None = None) -> Path:
    root = base or Path("eval/results")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / timestamp


def write_reports(result: EvalRunResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"

    payload = result.model_dump(mode="json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(format_report_markdown(result))

    return json_path, md_path


def format_report_markdown(result: EvalRunResult) -> str:
    config = result.config
    metrics = result.metrics
    macro = metrics["macro"]
    per_class = metrics["per_class"]
    matrix = metrics["confusion_matrix"]

    lines = [
        "# Classification eval report",
        "",
        "## Config",
        "",
        f"| Setting | Value |",
        f"|---------|-------|",
        f"| Mode | `{config.mode}` |",
        f"| Dataset | `{config.dataset_path}` |",
        f"| LLM mode | `{config.llm_mode}` |",
        f"| Classify model | `{config.classify_model}` |",
        f"| Prompt version | `{config.prompt_version}` |",
        f"| Started at | `{config.started_at.isoformat()}` |",
        f"| Examples | {metrics['n_examples']} |",
        "",
        "## Headline metrics",
        "",
        f"- **EMERGENT recall:** {metrics['emergent_recall']:.3f}",
        f"- **Macro F1:** {macro['f1']:.3f}",
        f"- **Macro precision:** {macro['precision']:.3f}",
        f"- **Macro recall:** {macro['recall']:.3f}",
        "",
        "## Per-class metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]

    for label, values in per_class.items():
        lines.append(
            f"| {label} | {values['precision']:.3f} | {values['recall']:.3f} "
            f"| {values['f1']:.3f} | {values['support']} |"
        )

    lines.extend(["", "## Confusion matrix", "", _format_confusion_matrix(matrix), ""])

    incorrect = [
        prediction for prediction in result.predictions if not prediction.correct
    ]
    if incorrect:
        lines.extend(["## Misclassifications", ""])
        for prediction in incorrect:
            predicted_all = ", ".join(c.value for c in prediction.predicted_all)
            lines.append(
                f"- `{prediction.id}`: gold `{prediction.label.value}` → "
                f"`{prediction.predicted.value}` (all: {predicted_all})"
            )
        lines.append("")

    lines.extend(
        [
            "## Sklearn classification report",
            "",
            "```",
            metrics["classification_report"].rstrip(),
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def _format_confusion_matrix(matrix: dict) -> str:
    labels: list[str] = matrix["labels"]
    rows: list[list[int]] = matrix["matrix"]

    header = "| true \\ pred | " + " | ".join(labels) + " |"
    separator = "|" + "|".join(["---"] * (len(labels) + 1)) + "|"
    body = [
        f"| {labels[row_idx]} | " + " | ".join(str(cell) for cell in row) + " |"
        for row_idx, row in enumerate(rows)
    ]
    return "\n".join([header, separator, *body])


def write_api_smoke_reports(
    result: ApiSmokeRunResult, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "api_smoke.json"
    md_path = output_dir / "api_smoke.md"

    payload = result.model_dump(mode="json")
    payload["summary"] = {
        "n_passed": result.n_passed,
        "n_failed": result.n_failed,
        "all_passed": result.all_passed,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(format_api_smoke_markdown(result))

    return json_path, md_path


def format_api_smoke_markdown(result: ApiSmokeRunResult) -> str:
    config = result.config
    lines = [
        "# API smoke test report",
        "",
        "## Config",
        "",
        "| Setting | Value |",
        "|---------|-------|",
        f"| Mode | `{config.mode}` |",
        f"| Dataset | `{config.dataset_path}` |",
        f"| Base URL | `{config.base_url}` |",
        f"| Started at | `{config.started_at.isoformat()}` |",
        f"| Health status | `{result.health_status}` |",
        "",
        "## Summary",
        "",
        f"- **Passed:** {result.n_passed}/{len(result.cases)}",
        f"- **Failed:** {result.n_failed}",
        f"- **All passed:** {result.all_passed}",
        "",
        "## Cases",
        "",
        "| ID | POST | GET | OK | Error |",
        "|----|------|-----|----|-------|",
    ]

    for case in result.cases:
        lines.append(
            f"| {case.id} | {case.create_status or '-'} | {case.get_status or '-'} "
            f"| {'yes' if case.ok else 'no'} | {case.error or ''} |"
        )

    lines.append("")
    return "\n".join(lines)
