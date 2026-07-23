from service.kb.retriever import FaissKnowledgeBase
from service.models import Classification, DraftJudgeResult, DraftRecord, DraftStatus, RetrievedSource
from service.settings import AppSettings, get_settings
from service.steps.abstraction import compute_abstraction_flag
from service.steps.classifier import Classifier, DummyClassifier
from service.steps.drafter import Drafter, DummyDrafter
from service.steps.extractor import DummyExtractor, Extractor
from service.steps.judge import DraftJudge, DummyJudge


def _build_human_review(
    abstraction_flag: bool,
    judge_result: DraftJudgeResult,
) -> tuple[bool, str | None]:
    reasons: list[str] = []
    if abstraction_flag:
        reasons.append("High-priority triage — clinician review required.")
    if judge_result.was_rewritten:
        reasons.append("Draft was auto-corrected for safety; clinician review required.")
    elif not judge_result.approved:
        reason = judge_result.reason.strip() or "Draft failed safety review."
        reasons.append(reason)
    if not reasons:
        return False, None
    return True, " ".join(reasons)


class DraftPipeline:
    def __init__(
        self,
        classifier: Classifier | None = None,
        extractor: Extractor | None = None,
        drafter: Drafter | None = None,
        judge: DraftJudge | None = None,
        knowledge_base: FaissKnowledgeBase | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self._classifier = classifier or DummyClassifier()
        self._extractor = extractor or DummyExtractor()
        self._drafter = drafter or DummyDrafter()
        self._judge = judge or DummyJudge()
        self._knowledge_base = knowledge_base
        self._settings = settings or get_settings()

    async def run(self, message: str) -> DraftRecord:
        classifications = await self._classifier.classify(message)
        structured_fields = await self._extractor.extract(message)
        abstraction_flag = compute_abstraction_flag(classifications, structured_fields)

        retrieved_sources: list[RetrievedSource] = []
        faq_context = ""
        if self._knowledge_base is not None:
            retrieved_sources = await self._knowledge_base.retrieve(
                message,
                top_k=self._settings.kb_top_k,
            )
            faq_context = self._knowledge_base.format_context(retrieved_sources)

        draft_response = await self._drafter.draft(
            message,
            classifications,
            structured_fields,
            abstraction_flag,
            faq_context=faq_context,
        )
        judge_result = await self._judge.review(
            message,
            draft_response,
            classifications,
            structured_fields,
        )
        needs_human_review, human_review_reason = _build_human_review(
            abstraction_flag,
            judge_result,
        )
        return DraftRecord(
            input_message=message,
            classifications=classifications,
            structured_fields=structured_fields,
            abstraction_flag=abstraction_flag,
            draft_response=judge_result.final_draft,
            retrieved_sources=retrieved_sources,
            judge_result=judge_result,
            needs_human_review=needs_human_review,
            human_review_reason=human_review_reason,
            status=DraftStatus.READY,
        )
