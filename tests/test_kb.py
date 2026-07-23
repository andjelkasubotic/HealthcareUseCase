import pytest

from service.kb.build import build_knowledge_base
from service.kb.chunker import load_faq_chunks
from service.settings import AppSettings


FAQ_PATH = "data/kb/dermatology_faq.txt"


def test_load_faq_chunks_parses_sections() -> None:
    chunks = load_faq_chunks(FAQ_PATH)
    assert len(chunks) >= 10
    assert chunks[0].id == "faq-001"
    assert "portal" in chunks[0].title.lower()


@pytest.mark.asyncio
async def test_build_knowledge_base_and_retrieve_routine_rash() -> None:
    settings = AppSettings(llm_mode="dummy", kb_faq_path=FAQ_PATH, kb_top_k=3)
    kb = await build_knowledge_base(settings)

    assert kb.chunk_count >= 10
    results = await kb.retrieve("I have a mild itchy rash on my arm", top_k=3)
    assert len(results) == 3
    assert all(result.score > 0 for result in results)
    assert all(result.id.startswith("faq-") for result in results)


@pytest.mark.asyncio
async def test_build_knowledge_base_and_retrieve_scheduling() -> None:
    settings = AppSettings(llm_mode="dummy", kb_faq_path=FAQ_PATH, kb_top_k=2)
    kb = await build_knowledge_base(settings)

    results = await kb.retrieve("I need to reschedule my appointment", top_k=2)
    assert results
    assert results[0].id == "faq-002"
