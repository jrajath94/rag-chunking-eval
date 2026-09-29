"""Hierarchical chunker: small children tiled under larger parents.

Retrieval contract: the evaluator retrieves at the child level (small,
precise spans), then expands each hit to its parent's full text for
generation. Children are emitted first, then parents, with sequential ids.
Child metadata carries the parent_id; parent metadata carries child_ids;
parent text is the children's texts joined with a newline.
"""

from __future__ import annotations

from ..text import split_sentences
from .base import Chunk, Chunker


class HierarchicalChunker(Chunker):
    """Two-level chunking.

    Children: greedy sentence packing, each child at most child_chars
    (one over-long sentence still becomes its own child). Children tile
    the text contiguously with no gaps.
    Parents: greedy grouping of consecutive children, each parent at most
    parent_chars and holding at least one child.
    """

    def __init__(self, child_chars: int = 250, parent_chars: int = 1000) -> None:
        if child_chars <= 0:
            raise ValueError("child_chars must be positive")
        if parent_chars <= 0:
            raise ValueError("parent_chars must be positive")
        self.child_chars = child_chars
        self.parent_chars = parent_chars
        self.name = f"hierarchical-{child_chars}-{parent_chars}"

    def chunk_document(self, doc_id: str, text: str) -> list[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        # Children: (char_start, char_end, [sentence indexes]).
        children: list[tuple[int, int, list[int]]] = []
        cur_start = 0
        cur_sents: list[int] = []
        for i, (_, e, _) in enumerate(sentences):
            if cur_sents and e - cur_start > self.child_chars:
                children.append((cur_start, sentences[cur_sents[-1]][1], cur_sents))
                cur_start = sentences[cur_sents[-1]][1]
                cur_sents = []
            cur_sents.append(i)
        if cur_sents:
            children.append((cur_start, len(text), cur_sents))

        # Parents: greedy groups of consecutive child indexes.
        groups: list[list[int]] = []
        cur: list[int] = []
        for j, (cs, ce, _) in enumerate(children):
            if cur and ce - children[cur[0]][0] > self.parent_chars:
                groups.append(cur)
                cur = []
            cur.append(j)
        if cur:
            groups.append(cur)

        n_children = len(children)
        parent_ids = [self._make_id(doc_id, n_children + k) for k in range(len(groups))]
        child_parent = {j: parent_ids[k] for k, grp in enumerate(groups) for j in grp}

        chunks: list[Chunk] = []
        for j, (cs, ce, sents) in enumerate(children):
            chunks.append(
                Chunk(
                    chunk_id=self._make_id(doc_id, j),
                    doc_id=doc_id,
                    text=text[cs:ce],
                    char_start=cs,
                    char_end=ce,
                    sentence_ids=tuple(f"{doc_id}:s{i}" for i in sents),
                    metadata={"level": "child", "parent_id": child_parent[j]},
                )
            )
        for k, grp in enumerate(groups):
            child_ids = [self._make_id(doc_id, j) for j in grp]
            sids = [f"{doc_id}:s{i}" for j in grp for i in children[j][2]]
            chunks.append(
                Chunk(
                    chunk_id=parent_ids[k],
                    doc_id=doc_id,
                    text="\n".join(text[children[j][0] : children[j][1]] for j in grp),
                    char_start=children[grp[0]][0],
                    char_end=children[grp[-1]][1],
                    sentence_ids=tuple(sids),
                    metadata={"level": "parent", "child_ids": child_ids},
                )
            )
        return chunks
