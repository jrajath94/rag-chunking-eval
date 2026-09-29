"""Property-style QA for the chunking pipeline (Project 3).

Model-free and deterministic: no network, no downloads. The semantic
chunker gets a stub embed_fn (md5-seeded per-text random vectors); the
runner tests get a stub keyword embed_fn (hashed bag-of-words). Random
texts come from a fixed seed so failures are reproducible.
"""

import hashlib
import random
import re

import numpy as np
import pytest

from rag_chunking_eval.chunking.fixed import FixedCharChunker
from rag_chunking_eval.chunking.hierarchical import HierarchicalChunker
from rag_chunking_eval.chunking.semantic import SemanticChunker
from rag_chunking_eval.evaluation.runner import ExperimentRunner
from rag_chunking_eval.text import split_sentences

SEED = 20260928

WORD_POOL = [
    "river", "stone", "market", "harbor", "candle", "forest", "train",
    "window", "paper", "signal", "garden", "bridge", "coffee", "mountain",
    "letter", "cloud", "engine", "meadow", "clock", "orchard",
]


def make_random_texts(seed: int = SEED) -> list[str]:
    """20 seeded texts: 14 length-graded plus 6 hostile edge cases."""
    rng = random.Random(seed)
    targets = [0, 1, 5, 50, 120, 250, 500, 750, 1000, 1300, 1600, 2000, 2500, 3000]
    texts = []
    for target in targets:
        parts = []
        total = 0
        while total < target:
            n_words = rng.randint(2, 9)
            words = [rng.choice(WORD_POOL) for _ in range(n_words)]
            words[0] = words[0].capitalize()
            sent = " ".join(words) + "."
            parts.append(sent)
            total += len(sent) + 1
        texts.append(" ".join(parts)[:target])
    texts.extend(
        [
            "",                      # empty
            "   \n\t  ",             # whitespace only
            "no punctuation anywhere just words drifting",  # no sentences
            "x" * 4000,              # one over-long sentence
            "  leading space. Second sentence here.",  # leading whitespace
            "lowercase start. Another One. and again.",   # ragged boundaries
        ]
    )
    return texts


def keyword_embed(texts):
    """Deterministic hashed bag-of-words stub: cosine ~ word overlap."""
    dim = 256
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9_]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return out


def per_text_embed(texts):
    """Deterministic per-text random vectors for the semantic chunker."""
    vecs = []
    for t in texts:
        seed = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % (2**32)
        r = random.Random(seed)
        vecs.append([r.random() for _ in range(8)])
    return np.array(vecs, dtype=float)


def snapshot(chunks):
    return [
        (
            c.chunk_id, c.doc_id, c.text, c.char_start, c.char_end,
            c.sentence_ids, tuple(sorted(c.metadata.items(), key=str)),
        )
        for c in chunks
    ]


RANDOM_TEXTS = make_random_texts()
FIXED = FixedCharChunker(chunk_size=500, overlap=0)
HIER = HierarchicalChunker(child_chars=250, parent_chars=1000)
SEM = SemanticChunker(embed_fn=per_text_embed, threshold=0.5)
ALL_CHUNKERS = [FIXED, HIER, SEM]


def doc_id_for(i: int) -> str:
    return f"prop-doc-{i:02d}"


# ---------------------------------------------------------------- fixed

class TestFixedProperties:
    def test_coverage_contiguity_ids_determinism(self):
        assert FIXED.name == "fixed-500-0"
        for i, text in enumerate(RANDOM_TEXTS):
            doc_id = doc_id_for(i)
            chunks = FIXED.chunk_document(doc_id, text)
            # every char offset covered by at least one chunk
            covered = [False] * len(text)
            for c in chunks:
                assert c.char_end > c.char_start, "degenerate chunk"
                assert c.text == text[c.char_start : c.char_end]
                for off in range(c.char_start, c.char_end):
                    covered[off] = True
            assert all(covered), f"gap in coverage for text {i}"
            # contiguous, no gaps between consecutive spans
            starts = sorted(c.char_start for c in chunks)
            if starts:
                assert starts[0] == 0
                for prev, cur in zip(chunks, chunks[1:]):
                    assert cur.char_start <= prev.char_end
            # chunk_id unique within the doc
            ids = [c.chunk_id for c in chunks]
            assert len(set(ids)) == len(ids)
            # chunking twice is identical
            assert snapshot(FIXED.chunk_document(doc_id, text)) == snapshot(chunks)


# ---------------------------------------------------------------- hierarchical

class TestHierarchicalProperties:
    def test_children_tile_and_parent_links(self):
        assert HIER.name == "hierarchical-250-1000"
        for i, text in enumerate(RANDOM_TEXTS):
            doc_id = doc_id_for(i)
            chunks = HIER.chunk_document(doc_id, text)
            children = [c for c in chunks if c.metadata.get("level") == "child"]
            parents = [c for c in chunks if c.metadata.get("level") == "parent"]
            if not text.strip():
                assert chunks == []
                continue
            assert children, "non-empty text must produce children"
            # children emitted before parents
            assert chunks.index(children[-1]) < chunks.index(parents[0])
            # children partition [0, len) with no gaps and no overlaps
            spans = sorted((c.char_start, c.char_end) for c in children)
            assert spans[0][0] == 0
            assert spans[-1][1] == len(text)
            for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
                assert s1 == e0, "children must tile contiguously"
            # child ids unique
            assert len({c.chunk_id for c in children}) == len(children)
            by_id = {c.chunk_id: c for c in chunks}
            for child in children:
                pid = child.metadata["parent_id"]
                assert pid in by_id, "child parent_id must point at an emitted chunk"
                parent = by_id[pid]
                assert parent.metadata.get("level") == "parent"
                assert child.chunk_id in parent.metadata["child_ids"]
                # parent char span covers the child span
                assert parent.char_start <= child.char_start
                assert parent.char_end >= child.char_end
            for parent in parents:
                kid_ids = parent.metadata["child_ids"]
                assert kid_ids, "parent must hold at least one child"
                kids = [by_id[k] for k in kid_ids]
                # parent sentence_ids == union of children's sentence_ids
                union = {s for k in kids for s in k.sentence_ids}
                assert set(parent.sentence_ids) == union
                # parent text is the newline join of child texts
                assert parent.text == "\n".join(k.text for k in kids)
            # deterministic
            assert snapshot(HIER.chunk_document(doc_id, text)) == snapshot(chunks)


# ---------------------------------------------------------------- runner determinism

class TestRunnerDeterminism:
    def _tiny_setup(self):
        docs = {
            f"qdoc-{j}": RANDOM_TEXTS[k]
            for j, k in enumerate([5, 8, 12])
        }
        questions = []
        for j, (doc_id, text) in enumerate(docs.items()):
            sents = split_sentences(text)
            assert sents, "chosen texts must contain sentences"
            n = min(2, len(sents))
            questions.append(
                {
                    "question_id": f"prop-q{j}",
                    "doc_id": doc_id,
                    "question": sents[0][2],
                    "gold_sentence_ids": [f"{doc_id}:s{t}" for t in range(n)],
                }
            )
        chunkers = [FixedCharChunker(500, 0), HierarchicalChunker(250, 1000)]
        return docs, questions, chunkers

    def test_run_twice_identical(self):
        docs, questions, chunkers = self._tiny_setup()
        run1 = ExperimentRunner(keyword_embed).run(questions, chunkers, docs)
        run2 = ExperimentRunner(keyword_embed).run(questions, chunkers, docs)
        assert len(run1) == 2 * len(questions)
        assert run1 == run2
        for r1, r2 in zip(run1, run2):
            assert r1["recall"] == r2["recall"]
            assert r1["retrieved_ids"] == r2["retrieved_ids"]


# ---------------------------------------------------------------- retrieval sanity

class TestRetrievalSanity:
    def _disjoint_docs(self):
        # every sentence's words are globally unique: a verbatim query
        # sentence can only match its own doc's chunks
        docs = {}
        names = ["docA", "docB", "docC"]
        for d, name in enumerate(names):
            sents = [
                f"Alpha{d}_{i} Beta{d}_{i} Gamma{d}_{i} Delta{d}_{i}."
                for i in range(12)
            ]
            docs[name] = " ".join(sents)
        return docs, "Alpha0_3 Beta0_3 Gamma0_3 Delta0_3."

    def test_verbatim_sentence_ranks_own_doc_first(self):
        docs, query = self._disjoint_docs()
        for chunker in [FixedCharChunker(500, 0), HierarchicalChunker(250, 1000)]:
            runner = ExperimentRunner(keyword_embed, k=3)
            runner.build_index(chunker.chunk_corpus(docs))
            hits = runner.retrieve(query, top_k=3)
            assert hits, "must retrieve something"
            assert hits[0].doc_id == "docA", (
                f"{chunker.name}: verbatim sentence query must rank docA first, "
                f"got {hits[0].doc_id}"
            )


# ---------------------------------------------------------------- sentence-id validity

class TestSentenceIdValidity:
    def test_sentence_ids_well_formed_and_inside_chunk(self):
        for chunker in ALL_CHUNKERS:
            for i, text in enumerate(RANDOM_TEXTS):
                doc_id = doc_id_for(i)
                sentences = split_sentences(text)
                n = len(sentences)
                for c in chunker.chunk_document(doc_id, text):
                    assert c.doc_id == doc_id
                    for sid in c.sentence_ids:
                        assert sid.startswith(f"{doc_id}:s"), f"bad id {sid}"
                        idx = int(sid.split(":s")[1])
                        assert 0 <= idx < n, f"id {sid} out of range for {n} sentences"
                        s, e, _ = sentences[idx]
                        assert c.char_start <= s and e <= c.char_end, (
                            f"sentence {sid} span [{s},{e}) not inside "
                            f"chunk [{c.char_start},{c.char_end})"
                        )
