"""Unit tests for evaluation metrics.

All fixtures are hand-made Chunks. No models, no corpus, no network.
"""

from rag_chunking_eval.chunking.base import Chunk
from rag_chunking_eval.evaluation import metrics


def _chunk(chunk_id, sentence_ids):
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        text="some text",
        char_start=0,
        char_end=9,
        sentence_ids=sentence_ids,
    )


def test_recall_perfect():
    chunks = [_chunk("c1", ["d1:s0", "d1:s1"])]
    assert metrics.recall_at_k(chunks, ["d1:s0", "d1:s1"]) == 1.0


def test_recall_partial():
    chunks = [_chunk("c1", ["d1:s0"])]
    assert metrics.recall_at_k(chunks, ["d1:s0", "d1:s1"]) == 0.5


def test_recall_zero():
    chunks = [_chunk("c1", ["d1:s5"])]
    assert metrics.recall_at_k(chunks, ["d1:s0", "d1:s1"]) == 0.0


def test_coverage_union_across_chunks():
    chunks = [_chunk("c1", ["d1:s0"]), _chunk("c2", ["d1:s0", "d1:s2"])]
    assert metrics.covered_sentence_ids(chunks) == {"d1:s0", "d1:s2"}


def test_faithfulness_score_mean_math():
    assert metrics.faithfulness_score([True, True, False]) == 2 / 3
    assert metrics.faithfulness_score([True]) == 1.0
    assert metrics.faithfulness_score([False, False]) == 0.0


def test_faithfulness_score_empty_is_none():
    # Caller must exclude unjudgeable answers from the mean.
    assert metrics.faithfulness_score([]) is None


def test_recall_empty_gold_is_zero():
    # No gold sentences means nothing to recall; defined as 0.0.
    chunks = [_chunk("c1", ["d1:s0"])]
    assert metrics.recall_at_k(chunks, []) == 0.0
