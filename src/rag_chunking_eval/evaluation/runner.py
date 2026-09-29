"""Experiment runner: chunk -> retrieve -> expand -> generate -> judge.

ExperimentRunner takes injectable callables (embed_fn, generate_fn,
nli_judge) so tests never touch real models. The real run wires in
sentence-transformers embeddings, an LLM generator, and HFNLIJudge.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import yaml

from rag_chunking_eval.chunking.base import Chunk, Chunker
from rag_chunking_eval.evaluation import metrics
from rag_chunking_eval.evaluation.judges import split_claims


def load_strategies(path: str | Path, embed_fn=None) -> list[Chunker]:
    """Instantiate chunkers from a YAML config.

    Each entry needs: name (the strategy identity used in results and chunk
    ids), class (dotted import path), params (dict). If embed_fn is given it
    is passed to any chunker constructor that accepts an `embed_fn` keyword
    (e.g. the semantic chunker); constructors without that parameter are
    built from params alone. Returns chunkers in file order.
    """
    import inspect

    with open(path) as f:
        config = yaml.safe_load(f)
    chunkers: list[Chunker] = []
    for entry in config["strategies"]:
        module_name, _, class_name = entry["class"].rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        params = dict(entry.get("params", {}))
        if embed_fn is not None and "embed_fn" in inspect.signature(cls.__init__).parameters:
            params["embed_fn"] = embed_fn
        chunker = cls(**params)
        # The config name is the strategy identity used in results and chunk ids.
        chunker.name = entry["name"]
        chunkers.append(chunker)
    return chunkers


class ExperimentRunner:
    """Runs retrieval + faithfulness experiments over chunking strategies.

    embed_fn: (list[str]) -> np.ndarray, rows expected L2-normalized but the
      runner normalizes defensively anyway.
    generate_fn: (question_dict, context_texts) -> str, or None to skip.
    nli_judge: object with score(premise, claims) -> list[bool], or None.
    k: number of chunks retrieved per question.
    """

    def __init__(self, embed_fn, generate_fn=None, nli_judge=None, k: int = 5):
        self.embed_fn = embed_fn
        self.generate_fn = generate_fn
        self.nli_judge = nli_judge
        self.k = k
        self._matrix: np.ndarray | None = None
        self._index_chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}

    def build_index(self, chunks: list[Chunk]) -> None:
        """Embed chunks and store the retrieval matrix.

        Hierarchical strategies index only child-level chunks
        (metadata level == "child"); parent chunks are kept for expansion.
        """
        self._by_id = {c.chunk_id: c for c in chunks}
        if any(c.metadata.get("level") == "child" for c in chunks):
            self._index_chunks = [c for c in chunks if c.metadata.get("level") == "child"]
        else:
            self._index_chunks = list(chunks)
        self._parent_of = {
            c.chunk_id: c.metadata["parent_id"]
            for c in chunks
            if c.metadata.get("parent_id") in self._by_id
        }
        matrix = np.asarray(self.embed_fn([c.text for c in self._index_chunks]), dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        """Brute-force cosine retrieval. Ties break on chunk_id for determinism."""
        q = np.asarray(self.embed_fn([query])[0], dtype=np.float64)
        norm = np.linalg.norm(q)
        q = q / norm if norm else q
        scores = self._matrix @ q
        ranked = sorted(
            range(len(self._index_chunks)),
            key=lambda i: (-scores[i], self._index_chunks[i].chunk_id),
        )
        return [self._index_chunks[i] for i in ranked[:top_k]]

    def expand(self, chunks: list[Chunk]) -> list[Chunk]:
        """Replace hierarchical children with their parent chunk.

        Dedupes by chunk id, keeps rank order. Non-hierarchical chunks have
        no parent link and pass through unchanged.
        """
        seen: set[str] = set()
        out: list[Chunk] = []
        for chunk in chunks:
            parent_id = self._parent_of.get(chunk.chunk_id)
            target = self._by_id[parent_id] if parent_id else chunk
            if target.chunk_id not in seen:
                seen.add(target.chunk_id)
                out.append(target)
        return out

    def run_question(self, question: dict, top_k: int | None = None) -> dict:
        """Retrieve, expand, score recall, and optionally generate + judge."""
        k = self.k if top_k is None else top_k
        retrieved = self.retrieve(question["question"], k)
        expanded = self.expand(retrieved)
        contexts = [c.text for c in expanded]
        record = {
            "question_id": question["question_id"],
            "retrieved_ids": [c.chunk_id for c in expanded],
            "recall": metrics.recall_at_k(expanded, question["gold_sentence_ids"]),
            "answer": None,
            "n_claims": 0,
            "n_supported": 0,
            "faithfulness": None,
        }
        if self.generate_fn is not None:
            answer = self.generate_fn(question, contexts)
            record["answer"] = answer
            claims = split_claims(answer)
            record["n_claims"] = len(claims)
            if claims and self.nli_judge is not None:
                premise = " ".join(contexts)
                supported = self.nli_judge.score(premise, claims)
                record["n_supported"] = sum(supported)
                record["faithfulness"] = metrics.faithfulness_score(supported)
        return record

    def run(self, questions: list[dict], chunkers: list[Chunker], docs: dict[str, str]) -> list[dict]:
        """Run every question under every strategy. Deterministic order."""
        results: list[dict] = []
        for chunker in chunkers:
            chunks = chunker.chunk_corpus(docs)
            self.build_index(chunks)
            for question in questions:
                record = self.run_question(question)
                record["strategy"] = chunker.name
                results.append(record)
        return results

    @staticmethod
    def summarize(results: list[dict]) -> list[dict]:
        """Per-strategy summary: n, mean recall, mean faithfulness (over
        judged questions only), and count of empty answers."""
        by_strategy: dict[str, list[dict]] = {}
        for record in results:
            by_strategy.setdefault(record["strategy"], []).append(record)
        summary = []
        for strategy, rows in by_strategy.items():
            faith = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
            summary.append(
                {
                    "strategy": strategy,
                    "n": len(rows),
                    "mean_recall": sum(r["recall"] for r in rows) / len(rows),
                    "mean_faithfulness": (sum(faith) / len(faith)) if faith else None,
                    "n_empty_answers": sum(1 for r in rows if not r["answer"]),
                }
            )
        return summary
