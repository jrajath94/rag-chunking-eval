"""Pure metric functions for the chunking evaluation.

All functions are deterministic and model-free. They operate on Chunk
objects (via chunk.sentence_ids) and on NLI judge outputs.
"""

from __future__ import annotations

from rag_chunking_eval.chunking.base import Chunk


def covered_sentence_ids(retrieved: list[Chunk]) -> set[str]:
    """Union of sentence ids covered by the retrieved chunks."""
    covered: set[str] = set()
    for chunk in retrieved:
        covered.update(chunk.sentence_ids)
    return covered


def recall_at_k(retrieved: list[Chunk], gold_ids: list[str]) -> float:
    """Fraction of gold sentence ids covered by the retrieved chunks.

    Empty gold list is defined as 0.0: with nothing to find, no retrieval
    can earn recall.
    """
    if not gold_ids:
        return 0.0
    covered = covered_sentence_ids(retrieved)
    return len(set(gold_ids) & covered) / len(gold_ids)


def faithfulness_score(supported: list[bool]) -> float | None:
    """Mean of per-claim support flags.

    Returns None for an empty claim list so the caller can exclude
    unjudgeable answers from the mean instead of scoring them as zero.
    """
    if not supported:
        return None
    return sum(supported) / len(supported)
