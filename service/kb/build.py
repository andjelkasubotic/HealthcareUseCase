from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from openai import AsyncOpenAI

from service.kb.chunker import FaqChunk, load_faq_chunks
from service.kb.embeddings import DummyEmbedder, Embedder, OpenAIEmbedder
from service.kb.retriever import FaissKnowledgeBase
from service.models import RetrievedSource
from service.settings import AppSettings


async def build_knowledge_base(settings: AppSettings) -> FaissKnowledgeBase:
    """Build a fresh in-memory FAISS index from the FAQ file.

    PROTOTYPE ONLY: this rebuilds the full index on every call (typically once per
    app startup). In production you should persist the index to disk (or object
    storage), version it with the FAQ file hash, and rebuild only when the source
    FAQ changes — not on every deploy or process restart.
    """
    faq_path = Path(settings.kb_faq_path)
    chunks = load_faq_chunks(faq_path)
    embedder = _build_embedder(settings)
    vectors = await embedder.embed_texts([_chunk_document(chunk) for chunk in chunks])

    index = faiss.IndexFlatIP(embedder.dimension)
    index.add(vectors)
    return FaissKnowledgeBase(chunks=chunks, index=index, embedder=embedder)


def _chunk_document(chunk: FaqChunk) -> str:
    return f"{chunk.title}\n{chunk.text}"


def _build_embedder(settings: AppSettings) -> Embedder:
    if settings.llm_mode == "openai":
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required to embed FAQ chunks.")
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        return OpenAIEmbedder(client=client, model=settings.embed.model)
    return DummyEmbedder(dimension=384)


def build_knowledge_base_sync(settings: AppSettings) -> FaissKnowledgeBase:
    import asyncio

    return asyncio.run(build_knowledge_base(settings))
