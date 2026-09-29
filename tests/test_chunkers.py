"""Tests for the three chunking strategies.

Red-green: these tests were written before the implementations and must
fail on import until fixed.py, semantic.py, and hierarchical.py exist.
No network or model downloads; semantic tests inject a stub embed_fn.
"""

import re

import numpy as np
import pytest

from rag_chunking_eval.chunking.fixed import FixedCharChunker
from rag_chunking_eval.chunking.hierarchical import HierarchicalChunker
from rag_chunking_eval.chunking.semantic import SemanticChunker
from rag_chunking_eval.text import split_sentences

DOC = "d1"

# Short doc with exactly 3 sentences, used for sentence_ids coverage.
THREE_SENT = "The cat sat on the mat. The dog barked loudly. Birds flew south."

# Longer doc with real sentences, used for metadata checks.
PARA = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
    "Sphinx of black quartz, judge my vow."
)


def sentence_index(sentence_id: str) -> int:
    return int(sentence_id.split(":s")[1])


def constant_embed(texts):
    """Stub embed_fn: every sentence gets the same vector."""
    return np.ones((len(texts), 4))


def clustered_embed(texts):
    """Stub embed_fn: two tight clusters, one boundary between 2 and 3."""
    vecs = [
        [1.0, 0.0],
        [1.0, 0.05],
        [0.0, 1.0],
        [0.05, 1.0],
    ]
    return np.array([vecs[i % 4] for i in range(len(texts))])


def snapshot(chunks):
    return [
        (c.chunk_id, c.doc_id, c.text, c.char_start, c.char_end, c.sentence_ids)
        for c in chunks
    ]


# ---------------------------------------------------------------- fixed

class TestFixedOverlapMath:
    def test_stride_and_coverage(self):
        text = "x" * 1000
        chunks = FixedCharChunker(chunk_size=100, overlap=20).chunk_document(DOC, text)
        assert chunks[0].char_start == 0
        for prev, cur in zip(chunks, chunks[1:]):
            assert cur.char_start == prev.char_start + 80
        # union covers [0, 1000) with no gaps
        covered = [False] * 1000
        for c in chunks:
            for i in range(c.char_start, c.char_end):
                covered[i] = True
        assert all(covered)
        for prev, cur in zip(chunks, chunks[1:]):
            assert cur.char_start <= prev.char_end  # no gap

    def test_name(self):
        assert FixedCharChunker(100, 20).name == "fixed-100-20"


class TestFixedBoundaries:
    def test_empty_text(self):
        assert FixedCharChunker(100, 0).chunk_document(DOC, "") == []

    def test_short_text_single_chunk(self):
        chunks = FixedCharChunker(100, 0).chunk_document(DOC, "abc")
        assert len(chunks) == 1
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 3
        assert chunks[0].text == "abc"

    def test_exactly_two_chunks(self):
        chunks = FixedCharChunker(100, 0).chunk_document(DOC, "x" * 200)
        assert len(chunks) == 2
        assert (chunks[0].char_start, chunks[0].char_end) == (0, 100)
        assert (chunks[1].char_start, chunks[1].char_end) == (100, 200)

    def test_no_empty_chunks_on_ragged_end(self):
        chunks = FixedCharChunker(100, 30).chunk_document(DOC, "x" * 250)
        assert all(c.char_end > c.char_start for c in chunks)
        assert chunks[-1].char_end == 250

    def test_bad_overlap_rejected(self):
        with pytest.raises(ValueError):
            FixedCharChunker(100, 100)


# ------------------------------------------------------------- metadata

@pytest.mark.parametrize(
    "chunker",
    [
        FixedCharChunker(60, 10),
        SemanticChunker(embed_fn=constant_embed),
        HierarchicalChunker(child_chars=60, parent_chars=300),
    ],
)
class TestMetadata:
    def test_chunk_ids_doc_id_and_slices(self, chunker):
        chunks = chunker.chunk_document(DOC, PARA)
        assert chunks, "expected at least one chunk"
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"{DOC}:::{chunker.name}:::{i:04d}"
            assert c.doc_id == DOC
        # text slice check holds for fixed/semantic and hierarchical children;
        # hierarchical parents join child texts with "\n" by design.
        for c in chunks:
            if c.metadata.get("level") != "parent":
                assert c.text == PARA[c.char_start : c.char_end]

    def test_sentence_ids_inside_span(self, chunker):
        chunks = chunker.chunk_document(DOC, PARA)
        sentences = split_sentences(PARA)
        assert sentences
        seen = set()
        for c in chunks:
            # A window may straddle sentence boundaries and legitimately carry
            # no ids; what matters is every id appears in at least one chunk
            # and each listed id lies fully inside its chunk span.
            for sid in c.sentence_ids:
                seen.add(sid)
                s_start, s_end, _ = sentences[sentence_index(sid)]
                assert s_start >= c.char_start
                assert s_end <= c.char_end
        expected = {f"{DOC}:s{i}" for i in range(len(sentences))}
        # Fixed windows can straddle a sentence in every window, so full
        # coverage is not guaranteed; listed ids must still be valid.
        assert seen, "expected some sentence ids on a sentence-bearing doc"
        assert seen <= expected


# ---------------------------------------------------------- determinism

@pytest.mark.parametrize(
    "chunker",
    [
        FixedCharChunker(60, 10),
        SemanticChunker(embed_fn=clustered_embed),
        HierarchicalChunker(child_chars=60, parent_chars=300),
    ],
)
def test_determinism(chunker):
    assert snapshot(chunker.chunk_document(DOC, PARA)) == snapshot(
        chunker.chunk_document(DOC, PARA)
    )


# -------------------------------------------------------------- semantic

class TestSemanticChunker:
    def test_one_boundary_between_clusters(self):
        text = "Alpha one. Beta two. Gamma three. Delta four."
        chunks = SemanticChunker(embed_fn=clustered_embed, threshold=0.5).chunk_document(
            DOC, text
        )
        assert len(chunks) == 2
        first, second = chunks
        assert set(first.sentence_ids) == {f"{DOC}:s0", f"{DOC}:s1"}
        assert set(second.sentence_ids) == {f"{DOC}:s2", f"{DOC}:s3"}
        assert first.char_end <= second.char_start

    def test_identical_embeddings_single_chunk(self):
        text = "Alpha one. Beta two. Gamma three. Delta four."
        chunks = SemanticChunker(embed_fn=constant_embed, threshold=0.5).chunk_document(
            DOC, text
        )
        assert len(chunks) == 1
        assert set(chunks[0].sentence_ids) == {f"{DOC}:s{i}" for i in range(4)}

    def test_single_sentence_doc(self):
        chunks = SemanticChunker(embed_fn=constant_embed).chunk_document(
            DOC, "Just one sentence here."
        )
        assert len(chunks) == 1

    def test_empty_text(self):
        assert SemanticChunker(embed_fn=constant_embed).chunk_document(DOC, "") == []

    def test_name(self):
        assert SemanticChunker(embed_fn=constant_embed, threshold=0.5).name == "semantic-0.5"

    def test_lazy_import_no_embed_fn(self):
        # Constructing without embed_fn must not import sentence_transformers.
        chunker = SemanticChunker()
        assert "sentence_transformers" not in __import__("sys").modules


# ---------------------------------------------------------- hierarchical

class TestHierarchicalChunker:
    def _split(self, chunks):
        children = [c for c in chunks if c.metadata.get("level") == "child"]
        parents = [c for c in chunks if c.metadata.get("level") == "parent"]
        return children, parents

    def test_children_tile_no_gaps(self):
        chunks = HierarchicalChunker(child_chars=60, parent_chars=300).chunk_document(
            DOC, PARA
        )
        children, _ = self._split(chunks)
        assert children
        assert children[0].char_start == 0
        assert children[-1].char_end == len(PARA)
        for prev, cur in zip(children, children[1:]):
            assert cur.char_start == prev.char_end

    def test_child_parent_links(self):
        chunks = HierarchicalChunker(child_chars=60, parent_chars=300).chunk_document(
            DOC, PARA
        )
        children, parents = self._split(chunks)
        parent_ids = {p.chunk_id for p in parents}
        assert parents
        by_id = {c.chunk_id: c for c in chunks}
        for child in children:
            assert child.metadata["level"] == "child"
            assert child.metadata["parent_id"] in parent_ids
        for parent in parents:
            assert parent.metadata["level"] == "parent"
            child_ids = parent.metadata["child_ids"]
            assert len(child_ids) >= 1
            for cid in child_ids:
                child = by_id[cid]
                assert child.metadata["parent_id"] == parent.chunk_id
                # parent span covers child span
                assert parent.char_start <= child.char_start
                assert parent.char_end >= child.char_end
            # parent sentence ids are the union of its children's
            union = set()
            for cid in child_ids:
                union.update(by_id[cid].sentence_ids)
            assert set(parent.sentence_ids) == union

    def test_children_first_then_parents(self):
        chunks = HierarchicalChunker(child_chars=60, parent_chars=300).chunk_document(
            DOC, PARA
        )
        children, parents = self._split(chunks)
        assert [c.chunk_id for c in chunks] == [
            c.chunk_id for c in children + parents
        ]
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"{DOC}:::hierarchical-60-300:::{i:04d}"

    def test_parent_text_joins_children(self):
        chunks = HierarchicalChunker(child_chars=60, parent_chars=300).chunk_document(
            DOC, PARA
        )
        children, parents = self._split(chunks)
        by_id = {c.chunk_id: c for c in chunks}
        for parent in parents:
            expected = "\n".join(by_id[cid].text for cid in parent.metadata["child_ids"])
            assert parent.text == expected

    def test_overlong_sentence_own_child(self):
        text = "Short one. " + "w" * 500 + ". " + "Short two."
        chunks = HierarchicalChunker(child_chars=60, parent_chars=1000).chunk_document(
            DOC, text
        )
        children, _ = self._split(chunks)
        solo = [c for c in children if len(c.sentence_ids) == 1]
        assert any(c.char_end - c.char_start > 60 for c in solo)

    def test_empty_text(self):
        assert HierarchicalChunker().chunk_document(DOC, "") == []

    def test_name(self):
        assert HierarchicalChunker(250, 1000).name == "hierarchical-250-1000"


# ------------------------------------------- sentence_ids end to end

@pytest.mark.parametrize(
    "chunker",
    [
        FixedCharChunker(50, 10),
        SemanticChunker(embed_fn=constant_embed),
        HierarchicalChunker(child_chars=50, parent_chars=500),
    ],
)
def test_every_sentence_covered(chunker):
    chunks = chunker.chunk_document(DOC, THREE_SENT)
    seen = set()
    for c in chunks:
        seen.update(c.sentence_ids)
    assert seen == {f"{DOC}:s{i}" for i in range(3)}
