"""SQuAD v2.0 dev slice loader.

Downloads the SQuAD v2.0 dev set once into .data/ (raw data stays out of
git), then derives a deterministic slice of answerable questions with
sentence-level gold ids: the gold sentence is the one containing the
answer span's character offset.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from rag_chunking_eval.text import split_sentences

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def data_dir() -> Path:
    """Directory holding the raw downloaded SQuAD file."""
    return Path(__file__).resolve().parents[3] / ".data"


def download_squad_dev(dest: Path | None = None) -> Path:
    """Fetch dev-v2.0.json into .data/ if not already present."""
    d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    dest = dest or (d / "dev-v2.0.json")
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"downloading SQuAD v2.0 dev set -> {dest}", flush=True)
    with urllib.request.urlopen(SQUAD_URL, timeout=120) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    return dest


def build_squad_slice(n_questions: int = 60) -> tuple[dict[str, str], list[dict]]:
    """Build a deterministic slice of answerable dev questions.

    Iterates articles in order, keeps only questions with is_impossible ==
    False and a non-empty answers list. Contexts are deduped by exact text
    into docs named squad-doc-000, .... gold_sentence_ids holds the id of
    the sentence containing answers[0]["answer_start"].
    """
    path = download_squad_dev()
    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    docs: dict[str, str] = {}
    doc_for_context: dict[str, str] = {}
    questions: list[dict] = []

    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if len(questions) >= n_questions:
                    break
                if qa.get("is_impossible", False):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                if context not in doc_for_context:
                    doc_id = f"squad-doc-{len(docs):03d}"
                    docs[doc_id] = context
                    doc_for_context[context] = doc_id
                doc_id = doc_for_context[context]
                gold_sentence_ids = _gold_sentences(context, answers[0]["answer_start"], doc_id)
                questions.append(
                    {
                        "question_id": qa["id"],
                        "doc_id": doc_id,
                        "question": qa["question"],
                        "answer": answers[0]["text"],
                        "gold_sentence_ids": gold_sentence_ids,
                    }
                )
            if len(questions) >= n_questions:
                break
        if len(questions) >= n_questions:
            break
    return docs, questions


def _gold_sentences(context: str, answer_start: int, doc_id: str) -> list[str]:
    """Sentence ids covering the answer span's start offset."""
    out = []
    for i, (s, e, _text) in enumerate(split_sentences(context)):
        if s <= answer_start < e:
            out.append(f"{doc_id}:s{i}")
            break
    return out


def write_squad_slice_json(questions: list[dict], n_docs: int, out_path: Path) -> None:
    """Write results/squad_slice.json: question dicts plus slice metadata."""
    payload = {
        "questions": questions,
        "n_docs": n_docs,
        "source": "SQuAD v2.0 dev",
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    out = Path(__file__).resolve().parents[3] / "results" / "squad_slice.json"
    docs, questions = build_squad_slice(60)
    write_squad_slice_json(questions, len(docs), out)
    print(f"wrote {len(questions)} questions from {len(docs)} docs -> {out}")


if __name__ == "__main__":
    sys.exit(main())
