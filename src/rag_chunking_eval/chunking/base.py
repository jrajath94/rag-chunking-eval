"""Shared contract every chunker implements.

A Chunk is a slice of one document with character offsets, the ids of the
sentences it covers (so the evaluator can map chunks back to sentence-level
gold annotations no matter which strategy produced them), and a metadata
dict for strategy-specific links (e.g. hierarchical parent ids).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    sentence_ids: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sentence_ids = tuple(self.sentence_ids)


class Chunker(ABC):
    """Base class. Subclasses set `name` and implement chunk_document."""

    name: str = "base"

    @abstractmethod
    def chunk_document(self, doc_id: str, text: str) -> list[Chunk]:
        """Split one document into chunks. Must be deterministic."""
        ...

    def chunk_corpus(self, docs: dict[str, str]) -> list[Chunk]:
        """Chunk every document in a {doc_id: text} mapping, in order."""
        out: list[Chunk] = []
        for doc_id, text in docs.items():
            out.extend(self.chunk_document(doc_id, text))
        return out

    def _make_id(self, doc_id: str, i: int) -> str:
        return f"{doc_id}:::{self.name}:::{i:04d}"
