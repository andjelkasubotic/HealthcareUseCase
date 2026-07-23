from __future__ import annotations

from service.kb.chunker import FaqChunk
from service.kb.embeddings import Embedder
from service.models import RetrievedSource


class FaissKnowledgeBase:
    """In-memory FAISS index over FAQ chunks for draft grounding."""

    def __init__(
        self,
        chunks: list[FaqChunk],
        index,
        embedder: Embedder,
    ) -> None:
        self._chunks = chunks
        self._index = index
        self._embedder = embedder
        self._chunk_by_id = {chunk.id: chunk for chunk in chunks}

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedSource]:
        if not query.strip() or self._index.ntotal == 0:
            return []

        query_vector = await self._embedder.embed_texts([query])
        scores, indices = self._index.search(query_vector, min(top_k, self._index.ntotal))

        results: list[RetrievedSource] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                RetrievedSource(
                    id=chunk.id,
                    title=chunk.title,
                    score=float(score),
                )
            )
        return results

    def format_context(self, sources: list[RetrievedSource]) -> str:
        if not sources:
            return "No FAQ context retrieved."

        lines = ["Approved clinic FAQ excerpts (use for tone and policy, not diagnosis):"]
        for source in sources:
            chunk = self._chunk_by_id[source.id]
            lines.append(f"[{chunk.id}] {chunk.title}\n{chunk.text}")
        return "\n\n".join(lines)
