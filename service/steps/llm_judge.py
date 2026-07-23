from pydantic import BaseModel, Field

from service.llm.config import LLMConfig
from service.llm.protocol import LLMClient
from service.llm.registry import get_llm_client
from service.llm.structured import complete_structured
from service.models import Classification, DraftJudgeResult, StructuredFields
from service.steps.judge import DraftJudge, DummyJudge
from service.steps.prompts import (
    JUDGE_CHECK_SYSTEM_PROMPT,
    JUDGE_REWRITE_SYSTEM_PROMPT,
    format_judge_rewrite_user_prompt,
    format_judge_user_prompt,
)


class JudgeCheckOutput(BaseModel):
    approved: bool
    violations: list[str] = Field(default_factory=list)
    reason: str = ""


class LlmJudge:
    """Two-step LLM review: check draft safety, then rewrite if needed."""

    def __init__(self, config: LLMConfig) -> None:
        self._client: LLMClient = get_llm_client(config)

    async def review(
        self,
        message: str,
        draft_response: str,
        classifications: list[Classification],
        fields: StructuredFields,
    ) -> DraftJudgeResult:
        check = await complete_structured(
            self._client,
            messages=[
                {"role": "system", "content": JUDGE_CHECK_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": format_judge_user_prompt(
                        message, draft_response, classifications, fields
                    ),
                },
            ],
            output_model=JudgeCheckOutput,
        )

        if check.approved:
            return DraftJudgeResult(
                approved=True,
                violations=check.violations,
                reason=check.reason,
                final_draft=draft_response,
            )

        rewrite = await self._client.acompletion(
            messages=[
                {"role": "system", "content": JUDGE_REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": format_judge_rewrite_user_prompt(
                        message,
                        draft_response,
                        classifications,
                        fields,
                        check.reason,
                        check.violations,
                    ),
                },
            ],
        )

        return DraftJudgeResult(
            approved=False,
            violations=check.violations,
            reason=check.reason,
            was_rewritten=True,
            original_draft=draft_response,
            final_draft=rewrite.content.strip(),
        )


def build_judge(settings_mode: str, config: LLMConfig) -> DraftJudge:
    if settings_mode == "openai":
        return LlmJudge(config)
    return DummyJudge()
