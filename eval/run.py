import argparse
import sys
from pathlib import Path

from eval.api_smoke import run_api_smoke_sync
from eval.report import default_output_dir, write_api_smoke_reports, write_reports
from eval.runner import run_classify_eval_sync
from eval.schemas import EvalMode
from service.settings import get_settings
from service.steps.prompts import PROMPT_VERSION

DEFAULT_CLASSIFY_DATASET = Path("data/eval/fixtures/seed.jsonl")
DEFAULT_API_DATASET = Path("data/eval/fixtures/smoke.jsonl")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run triage classification evaluation."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to labeled JSONL dataset (default depends on --mode).",
    )
    parser.add_argument(
        "--mode",
        choices=["classify-only", "full-pipeline", "api"],
        default="classify-only",
        help="Eval mode: classify-only (metrics) or api (HTTP smoke test).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Draft service base URL for --mode api (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for reports (default: eval/results/<timestamp>).",
    )
    return parser.parse_args(argv)


def resolve_dataset(mode: EvalMode, dataset: Path | None) -> Path:
    if dataset is not None:
        return dataset
    if mode == "api":
        return DEFAULT_API_DATASET
    return DEFAULT_CLASSIFY_DATASET


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode: EvalMode = args.mode
    dataset = resolve_dataset(mode, args.dataset)

    if not dataset.exists():
        print(f"Dataset not found: {dataset}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or default_output_dir()

    if mode == "classify-only":
        settings = get_settings()
        print(
            f"Running classify-only eval on {dataset} "
            f"(llm_mode={settings.llm_mode}, model={settings.classify.model}, "
            f"prompt_version={PROMPT_VERSION})"
        )

        result = run_classify_eval_sync(dataset, settings=settings)
        json_path, md_path = write_reports(result, output_dir)

        print(f"EMERGENT recall: {result.metrics['emergent_recall']:.3f}")
        print(f"Macro F1: {result.metrics['macro']['f1']:.3f}")
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")
        return 0

    if mode == "api":
        print(f"Running API smoke test on {dataset} (base_url={args.base_url})")

        result = run_api_smoke_sync(dataset, base_url=args.base_url)
        json_path, md_path = write_api_smoke_reports(result, output_dir)

        print(f"Health: {result.health_status}")
        print(f"Passed: {result.n_passed}/{len(result.cases)}")
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")
        return 0 if result.all_passed else 1

    print(
        f"Mode {mode!r} is not implemented yet. Use --mode classify-only or --mode api.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
