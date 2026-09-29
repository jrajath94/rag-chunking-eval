"""Fixed-size character-window chunker with optional overlap."""

from __future__ import annotations

from ..text import split_sentences
from .base import Chunk, Chunker


def _sentence_ids(doc_id: str, sentences, start: int, end: int) -> tuple[str, ...]:
    # Include a sentence only when its span lies fully inside the chunk.
    return tuple(
        f"{doc_id}:s{i}"
        for i, (s, e, _) in enumerate(sentences)
        if s >= start and e <= end
    )


class FixedCharChunker(Chunker):
    """Pure character windows.

    Starts at 0, then chunk_size, chunk_size - overlap, and so on. The last
    chunk may be shorter than chunk_size; empty chunks are never emitted.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if chunk_size - overlap <= 0:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._step = chunk_size - overlap
        self.name = f"fixed-{chunk_size}-{overlap}"

    def chunk_document(self, doc_id: str, text: str) -> list[Chunk]:
        sentences = split_sentences(text)
        chunks: list[Chunk] = []
        for i, start in enumerate(range(0, len(text), self._step)):
            end = min(start + self.chunk_size, len(text))
            if end <= start:
                break
            chunks.append(
                Chunk(
                    chunk_id=self._make_id(doc_id, i),
                    doc_id=doc_id,
                    text=text[start:end],
                    char_start=start,
                    char_end=end,
                    sentence_ids=_sentence_ids(doc_id, sentences, start, end),
                )
            )
        return chunks
