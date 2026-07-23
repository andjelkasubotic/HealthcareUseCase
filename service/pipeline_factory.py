from service.kb.retriever import FaissKnowledgeBase
from service.pipeline import DraftPipeline
from service.settings import AppSettings, get_settings
from service.steps.llm_classifier import build_classifier
from service.steps.llm_drafter import build_drafter
from service.steps.llm_extractor import build_extractor
from service.steps.llm_judge import build_judge


def build_pipeline(
    settings: AppSettings | None = None,
    knowledge_base: FaissKnowledgeBase | None = None,
) -> DraftPipeline:
    app_settings = settings or get_settings()
    if app_settings.llm_mode == "openai" and app_settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required when LLM_MODE=openai.")

    return DraftPipeline(
        classifier=build_classifier(
            app_settings.llm_mode,
            app_settings.task_config("classify"),
        ),
        extractor=build_extractor(
            app_settings.llm_mode,
            app_settings.task_config("extract"),
        ),
        drafter=build_drafter(
            app_settings.llm_mode,
            app_settings.task_config("draft"),
        ),
        judge=build_judge(
            app_settings.llm_mode,
            app_settings.task_config("judge"),
        ),
        knowledge_base=knowledge_base,
        settings=app_settings,
    )
