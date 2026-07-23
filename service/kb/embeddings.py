from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    async def embed_texts(self, texts: list[str]) -> np.ndarray: ...


class DummyEmbedder:
    """Deterministic bag-of-words-style vectors for local/dev use (no API)."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.vstack([self._embed_one(text) for text in texts]).astype(np.float32)
        return _normalize_rows(vectors)

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in text.lower().split():
            vector[hash(token) % self._dimension] += 1.0
        return vector


class OpenAIEmbedder:
    """OpenAI embedding API for production-quality retrieval."""

    def __init__(self, client, model: str, dimension: int = 1536) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        vectors = np.vstack(
            [np.array(item.embedding, dtype=np.float32) for item in response.data]
        )
        return _normalize_rows(vectors)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms
