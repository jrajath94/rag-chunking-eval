"""Semantic chunker: cut between sentences where similarity drops."""

from __future__ import annotations

import numpy as np

from ..text import split_sentences
from .base import Chunk, Chunker


class SemanticChunker(Chunker):
    """Split on sentences, cut where adjacent cosine similarity < threshold.

    embed_fn is a callable(list[str]) -> np.ndarray, injected so tests can
    use a stub. When None, sentence_transformers is imported lazily and
    model_name is loaded on first use, so unit tests pay no import cost.
    """

    def __init__(
        self,
        embed_fn=None,
        threshold: float = 0.5,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.model_name = model_name
        self.name = f"semantic-{threshold}"
        self._model = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.embed_fn is not None:
            return np.asarray(self.embed_fn(texts), dtype=float)
        if self._model is None:
            # Lazy import: keeps module import light for unit tests.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(texts), dtype=float)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def chunk_document(self, doc_id: str, text: str) -> list[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        # Sentence-index boundaries; each chunk covers sentences[a:b].
        bounds = [0]
        if len(sentences) > 1:
            embs = self._embed([s for _, _, s in sentences])
            for i in range(len(embs) - 1):
                if self._cosine(embs[i], embs[i + 1]) < self.threshold:
                    bounds.append(i + 1)
        bounds.append(len(sentences))
        chunks: list[Chunk] = []
        for ci, (a, b) in enumerate(zip(bounds, bounds[1:])):
            start = sentences[a][0]
            end = sentences[b - 1][1]
            chunks.append(
                Chunk(
                    chunk_id=self._make_id(doc_id, ci),
                    doc_id=doc_id,
                    text=text[start:end],
                    char_start=start,
                    char_end=end,
                    sentence_ids=tuple(f"{doc_id}:s{i}" for i in range(a, b)),
                )
            )
        return chunks
