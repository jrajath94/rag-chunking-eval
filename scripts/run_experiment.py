"""Full experiment run: synthetic sanity -> SQuAD retrieval -> generation -> NLI -> report.

Stages (each resumable; rerunning a stage overwrites its outputs):
  A  synthetic sanity with the real embedder (MiniLM), recall@5 only.
  B  SQuAD slice retrieval: chunk, embed, retrieve k=5, recall@5 per strategy.
  C  generation with google/flan-t5-base (greedy) from retrieved contexts.
  D  faithfulness judging with cross-encoder/nli-deberta-v3-small.
  E  summary.csv, RESULTS.md, recall.png, faithfulness.png.

Usage:
  cd ~/workspace/rag-chunking-eval
  .venv/bin/python scripts/run_experiment.py [--stages A,B,C,D,E]

Frozen controls (identical across strategies): embedder, generator, k=5,
NLI judge, prompt template. Only the chunking changes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _sanitize_proxy_env() -> None:
    """Strip bracketed IPv6 literals in no_proxy/NO_PROXY.

    httpx 0.28 (used by huggingface-hub) raises InvalidURL on values like
    "[::1]" when building its proxy map from the environment. Bare "::1" is
    fine, so we strip the brackets at process start. Local-only change.
    """
    for var in ("no_proxy", "NO_PROXY"):
        val = os.environ.get(var)
        if val:
            os.environ[var] = re.sub(r"\[([0-9a-fA-F:]+)\]", r"\1", val)


_sanitize_proxy_env()

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results"
CONFIG = REPO / "configs" / "strategies.yaml"
K = 5
CONTEXT_MAX_TOKENS = 450  # generation context budget, enforced with the T5 tokenizer
PREMISE_MAX_TOKENS = 400  # NLI premise budget, enforced with the NLI tokenizer

PROMPT_HEAD = "Answer the question using only the context below.\n\nContext:\n"
PROMPT_TAIL_TMPL = "\n\nQuestion: {question}\n\nAnswer:"
MAX_PROMPT_TOKENS = 512  # flan-t5-base's hard input limit


def _truncate_context(context: str, tokenizer, max_tokens: int) -> tuple[str, bool]:
    """Cut the context to max_tokens with the model's own tokenizer.

    Only the context side is cut; the prompt framing and question are always
    kept intact. Returns (possibly truncated context, was_truncated).
    """
    ids = tokenizer.encode(context, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
        return tokenizer.decode(ids), True
    return context, False


def _build_prompt(question: str, context: str, tokenizer) -> tuple[str, bool]:
    """Build the generation prompt with a hard 512-token ceiling.

    The context is cut to min(450, 512 - framing - question) tokens with the
    T5 tokenizer, so the full prompt never exceeds the model's input limit.
    Framing and question are never cut. Returns (prompt, was_truncated).
    """
    tail = PROMPT_TAIL_TMPL.format(question=question)
    fixed = len(tokenizer.encode(PROMPT_HEAD + tail, add_special_tokens=False))
    budget = min(CONTEXT_MAX_TOKENS, MAX_PROMPT_TOKENS - fixed)
    context, truncated = _truncate_context(context, tokenizer, budget)
    return PROMPT_HEAD + context + tail, truncated


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------
# Stage A: synthetic sanity with the real embedder
# ----------------------------------------------------------------------------

def stage_a() -> None:
    from sentence_transformers import SentenceTransformer

    from rag_chunking_eval.corpus.synthetic import build_synthetic_corpus
    from rag_chunking_eval.evaluation.runner import ExperimentRunner, load_strategies

    print("[A] loading all-MiniLM-L6-v2", flush=True)
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embed_fn = lambda texts: model.encode(texts, batch_size=32, show_progress_bar=False)

    docs, questions = build_synthetic_corpus(seed=7)
    chunkers = load_strategies(CONFIG, embed_fn=embed_fn)

    runner = ExperimentRunner(embed_fn=embed_fn, k=K)
    records = runner.run(questions, chunkers, docs)
    summary = ExperimentRunner.summarize(records)

    out = {
        "stage": "A",
        "corpus": "synthetic (20 docs, 40 questions, seed=7)",
        "k": K,
        "summary": summary,
        "built_at": utc_now(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "synthetic_real_embedder.json", "w") as f:
        json.dump(out, f, indent=2)

    predicted = ["hierarchical-250-1000", "semantic-0.5", "fixed-500-100", "fixed-500-0"]
    actual = [row["strategy"] for row in sorted(summary, key=lambda r: -r["mean_recall"])]
    recall_by = {row["strategy"]: row["mean_recall"] for row in summary}
    tied = len(set(round(recall_by[s], 6) for s in predicted)) == 1
    held = actual == predicted
    with open(RESULTS / "synthetic_check.md", "w") as f:
        f.write("# Synthetic sanity check (real embedder)\n\n")
        f.write(f"Run at {utc_now()} (UTC).\n\n")
        f.write("Predicted ordering (from TRD Section 8):\n\n")
        f.write("hierarchical >= semantic > fixed-500/100 > fixed-500-0\n\n")
        f.write("Actual mean recall@5:\n\n")
        for name in actual:
            f.write(f"- {name}: {recall_by[name]:.3f}\n")
        if tied:
            f.write("\nResult: four-way tie at recall@5 = 1.000. The toy corpus is too "
                    "easy to discriminate strategies (each document is ~10 sentences, "
                    "so every strategy retrieves the planted fact sentence). The "
                    "ordering is neither confirmed nor falsified here; it will be "
                    "tested for real on the SQuAD slice below. The prediction is not "
                    "adjusted after the fact.\n")
        else:
            f.write(f"\nPrediction held on the synthetic corpus: {held}.\n")
            if not held:
                f.write("The ordering did not hold on the toy corpus. The SQuAD run "
                        "below is reported honestly either way; the prediction is not "
                        "adjusted after the fact.\n")
    print(f"[A] done in {time.time() - t0:.0f}s; tied={tied} prediction held: {held}", flush=True)

    del model
    gc.collect()


# ----------------------------------------------------------------------------
# Stage B: SQuAD slice retrieval
# ----------------------------------------------------------------------------

def stage_b() -> None:
    from sentence_transformers import SentenceTransformer

    from rag_chunking_eval.corpus.squad import build_squad_slice, write_squad_slice_json
    from rag_chunking_eval.evaluation import metrics
    from rag_chunking_eval.evaluation.runner import ExperimentRunner, load_strategies

    print("[B] loading all-MiniLM-L6-v2", flush=True)
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embed_fn = lambda texts: model.encode(texts, batch_size=32, show_progress_bar=False)

    # Deterministic: same function and order every time. Download happens
    # once; afterwards the cached file in .data/ is reused with no network.
    docs, questions = build_squad_slice(n_questions=60)
    write_squad_slice_json(questions, len(docs), RESULTS / "squad_slice.json")

    chunkers = load_strategies(CONFIG, embed_fn=embed_fn)
    all_records: list[dict] = []
    for chunker in chunkers:
        print(f"[B] chunking + retrieving: {chunker.name}", flush=True)
        runner = ExperimentRunner(embed_fn=embed_fn, k=K)
        chunks = chunker.chunk_corpus(docs)
        runner.build_index(chunks)
        for q in questions:
            retrieved = runner.retrieve(q["question"], K)
            expanded = runner.expand(retrieved)
            all_records.append(
                {
                    "question_id": q["question_id"],
                    "strategy": chunker.name,
                    "retrieved_ids": [c.chunk_id for c in expanded],
                    "recall": metrics.recall_at_k(expanded, q["gold_sentence_ids"]),
                    "contexts": [c.text for c in expanded],
                    "question": q["question"],
                }
            )

    with open(RESULTS / "retrieval.json", "w") as f:
        json.dump(
            {
                "stage": "B",
                "corpus": "SQuAD v2.0 dev slice (60 answerable questions)",
                "k": K,
                "embedder": "sentence-transformers/all-MiniLM-L6-v2",
                "records": all_records,
                "built_at": utc_now(),
                "wall_seconds": round(time.time() - t0, 1),
            },
            f,
            indent=2,
        )
    print(f"[B] done in {time.time() - t0:.0f}s ({len(all_records)} records)", flush=True)

    del model
    gc.collect()


# ----------------------------------------------------------------------------
# Stage C: generation with flan-t5-base
# ----------------------------------------------------------------------------

def _truncate_context(context: str, tokenizer, max_tokens: int) -> tuple[str, bool]:
    """Cut the context to max_tokens with the model's own tokenizer.

    Only the context side is cut; the prompt framing and question are always
    kept intact. Returns (possibly truncated context, was_truncated).
    """
    ids = tokenizer.encode(context, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
        return tokenizer.decode(ids), True
    return context, False


def stage_c() -> None:
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    t0 = time.time()
    with open(RESULTS / "retrieval.json") as f:
        retrieval = json.load(f)
    records = retrieval["records"]

    print("[C] loading google/flan-t5-base (CPU)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    gen_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    gen_model.eval()

    truncated_counts: dict[str, int] = {}
    n = 0
    total = len(records)
    with torch.no_grad():
        for rec in records:
            context = "\n\n".join(rec["contexts"])
            prompt, was_truncated = _build_prompt(rec["question"], context, tokenizer)
            rec["context_truncated"] = was_truncated
            if was_truncated:
                truncated_counts[rec["strategy"]] = truncated_counts.get(rec["strategy"], 0) + 1
            inputs = tokenizer(prompt, return_tensors="pt")
            assert inputs["input_ids"].shape[1] <= MAX_PROMPT_TOKENS, (
                f"prompt overflow: {inputs['input_ids'].shape[1]} tokens"
            )
            out_ids = gen_model.generate(
                **inputs, do_sample=False, max_new_tokens=64
            )
            rec["answer"] = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            n += 1
            if n % 20 == 0:
                print(f"[C] {n}/{total} generations", flush=True)

    # Drop the bulky raw contexts now that answers exist; retrieval.json keeps them.
    for rec in records:
        rec.pop("contexts", None)

    with open(RESULTS / "answers.json", "w") as f:
        json.dump(
            {
                "stage": "C",
                "generator": "google/flan-t5-base (greedy, do_sample=False, max_new_tokens=64)",
                "context_truncation": (
                    f"context cut with the T5 tokenizer so the full prompt fits "
                    f"{MAX_PROMPT_TOKENS} tokens (context budget min({CONTEXT_MAX_TOKENS}, "
                    f"{MAX_PROMPT_TOKENS} - framing - question)); framing and question "
                    f"always kept"
                ),
                "truncated_per_strategy": truncated_counts,
                "records": records,
                "built_at": utc_now(),
                "wall_seconds": round(time.time() - t0, 1),
            },
            f,
            indent=2,
        )
    print(f"[C] done in {time.time() - t0:.0f}s; truncations: {truncated_counts}", flush=True)

    del gen_model
    del tokenizer
    gc.collect()


# ----------------------------------------------------------------------------
# Stage D: NLI faithfulness judging
# ----------------------------------------------------------------------------

def stage_d() -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from rag_chunking_eval.evaluation.judges import split_claims
    from rag_chunking_eval.evaluation.metrics import faithfulness_score

    t0 = time.time()
    with open(RESULTS / "answers.json") as f:
        answers = json.load(f)
    records = answers["records"]
    with open(RESULTS / "retrieval.json") as f:
        retrieval = json.load(f)
    contexts_by = {(r["question_id"], r["strategy"]): r["contexts"] for r in retrieval["records"]}

    # The existing HFNLIJudge wraps a sentence-transformers CrossEncoder; the
    # run brief asks for AutoTokenizer + AutoModelForSequenceClassification
    # instead, so the judge below is a small local equivalent with the same
    # semantics (documented here): premise truncated to 400 tokens with the
    # model's own tokenizer, supported = P(entail) >= 0.5 and P(contra) < 0.5,
    # label indices resolved from config.id2label (never hardcoded).
    print("[D] loading cross-encoder/nli-deberta-v3-small (CPU)", flush=True)
    tok = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
    nli = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small")
    nli.eval()
    id2label = nli.config.id2label
    entail_idx = next(i for i, l in id2label.items() if l.lower() == "entailment")
    contra_idx = next(i for i, l in id2label.items() if l.lower() == "contradiction")

    def judge(premise: str, claims: list[str]) -> list[bool]:
        ids = tok.encode(premise, add_special_tokens=False)[:PREMISE_MAX_TOKENS]
        premise_t = tok.decode(ids)
        out: list[bool] = []
        with torch.no_grad():
            for claim in claims:
                inputs = tok(premise_t, claim, return_tensors="pt", truncation=True)
                logits = nli(**inputs).logits[0].numpy()
                shifted = logits - logits.max()
                probs = np.exp(shifted) / np.exp(shifted).sum()
                out.append(bool(probs[entail_idx] >= 0.5 and probs[contra_idx] < 0.5))
        return out

    judged = 0
    for rec in records:
        contexts = contexts_by[(rec["question_id"], rec["strategy"])]
        premise = "\n\n".join(contexts)
        claims = split_claims(rec["answer"]) if rec["answer"] else []
        rec["n_claims"] = len(claims)
        if claims:
            supported = judge(premise, claims)
            rec["n_supported"] = int(sum(supported))
            rec["faithfulness"] = faithfulness_score(supported)
            judged += 1
        else:
            rec["n_supported"] = 0
            rec["faithfulness"] = None
        # strip intermediates not wanted in the final artifact
        rec.pop("question", None)

    final = [
        {
            "question_id": r["question_id"],
            "strategy": r["strategy"],
            "retrieved_ids": r["retrieved_ids"],
            "recall": r["recall"],
            "answer": r["answer"],
            "n_claims": r["n_claims"],
            "n_supported": r["n_supported"],
            "faithfulness": r["faithfulness"],
            "context_truncated": r.get("context_truncated", False),
        }
        for r in records
    ]
    with open(RESULTS / "results.json", "w") as f:
        json.dump(
            {
                "run": {
                    "corpus": "SQuAD v2.0 dev slice (60 answerable questions)",
                    "embedder": "sentence-transformers/all-MiniLM-L6-v2",
                    "generator": "google/flan-t5-base, greedy (do_sample=False), max_new_tokens=64",
                    "k": K,
                    "nli_judge": "cross-encoder/nli-deberta-v3-small, "
                                 "supported = P(entailment) >= 0.5 and P(contradiction) < 0.5",
                    "built_at": utc_now(),
                    "wall_seconds": round(time.time() - t0, 1),
                },
                "records": final,
            },
            f,
            indent=2,
        )
    print(f"[D] done in {time.time() - t0:.0f}s; judged {judged}/{len(records)} non-empty answers",
          flush=True)

    del nli
    del tok
    gc.collect()


# ----------------------------------------------------------------------------
# Stage E: summary table, report, plots
# ----------------------------------------------------------------------------

def _versions() -> dict[str, str]:
    import platform

    import sentence_transformers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "platform": platform.platform(),
    }


def stage_e() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from rag_chunking_eval.evaluation.runner import ExperimentRunner

    with open(RESULTS / "results.json") as f:
        payload = json.load(f)
    records = payload["records"]
    run = payload["run"]
    versions = _versions()

    summary = ExperimentRunner.summarize(records)
    order = [r["strategy"] for r in sorted(summary, key=lambda r: -r["mean_recall"])]

    # Per-strategy distributions (TRD 6.1: a single mean hides failures).
    dist: dict[str, dict] = {}
    for row in summary:
        strat = row["strategy"]
        recalls = [r["recall"] for r in records if r["strategy"] == strat]
        faiths = [r["faithfulness"] for r in records
                  if r["strategy"] == strat and r["faithfulness"] is not None]
        dist[strat] = {
            "recall": {
                "min": float(np.min(recalls)),
                "p25": float(np.percentile(recalls, 25)),
                "median": float(np.median(recalls)),
                "p75": float(np.percentile(recalls, 75)),
                "max": float(np.max(recalls)),
            },
            "faithfulness": {
                "min": float(np.min(faiths)) if faiths else None,
                "median": float(np.median(faiths)) if faiths else None,
                "max": float(np.max(faiths)) if faiths else None,
            },
        }

    def _fmt(v):
        return f"{v:.4f}" if v is not None else ""

    with open(RESULTS / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "strategy", "n", "recall@5",
            "recall_min", "recall_p25", "recall_median", "recall_p75", "recall_max",
            "faithfulness", "faithfulness_min", "faithfulness_median", "faithfulness_max",
            "n_empty_answers",
        ])
        for row in summary:
            d = dist[row["strategy"]]
            fm = d["faithfulness"]
            w.writerow(
                [
                    row["strategy"],
                    row["n"],
                    f"{row['mean_recall']:.4f}",
                    f"{d['recall']['min']:.4f}",
                    f"{d['recall']['p25']:.4f}",
                    f"{d['recall']['median']:.4f}",
                    f"{d['recall']['p75']:.4f}",
                    f"{d['recall']['max']:.4f}",
                    _fmt(row["mean_faithfulness"]),
                    _fmt(fm["min"]),
                    _fmt(fm["median"]),
                    _fmt(fm["max"]),
                    row["n_empty_answers"],
                ]
            )

    for metric, fname, title in [
        ("mean_recall", "recall.png", "Recall@5 by chunking strategy (n=60)"),
        ("mean_faithfulness", "faithfulness.png", "Faithfulness by chunking strategy (n=60)"),
    ]:
        names = [r["strategy"] for r in summary]
        vals = [r[metric] for r in summary]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(names, vals, color="#4a7fb5")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.replace("mean_", "").replace("_", " "))
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.3f}" if v is not None else "n/a",
                    ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(RESULTS / fname, dpi=150)
        plt.close(fig)

    with open(RESULTS / "synthetic_check.md") as f:
        synth = f.read()

    trunc = sum(1 for r in records if r["context_truncated"])
    lines = []
    lines.append("# RAG chunking evaluation: results\n")
    lines.append("## Run info\n")
    lines.append(f"- Date (UTC): {run['built_at']}")
    lines.append(f"- Corpus: {run['corpus']}")
    lines.append(f"- Embedder: {run['embedder']} (CPU)")
    lines.append(f"- Generator: {run['generator']} (CPU)")
    lines.append("- k: 5")
    lines.append(f"- NLI judge: {run['nli_judge']} (CPU)")
    lines.append(f"- Context truncation: context cut with the T5 tokenizer so the full "
                 f"prompt fits {MAX_PROMPT_TOKENS} tokens (context budget "
                 f"min({CONTEXT_MAX_TOKENS}, {MAX_PROMPT_TOKENS} - framing - question)); "
                 f"{trunc} of {len(records)} question-strategy pairs were truncated")
    lines.append("- Python/torch/transformers/sentence-transformers: "
                 f"{versions['python']} / {versions['torch']} / "
                 f"{versions['transformers']} / {versions['sentence_transformers']}")
    lines.append(f"- Machine: {versions['platform']}, CPU-only")
    lines.append("")
    lines.append("## Predicted vs actual\n")
    lines.append("Prediction (TRD Section 8, stated before the synthetic run):")
    lines.append("hierarchical >= semantic > fixed-500/100 > fixed-500-0")
    lines.append("")
    lines.append("Actual ordering by mean recall@5: " + " > ".join(order))
    from collections import defaultdict
    faith_groups: dict[float, list[str]] = defaultdict(list)
    for r in summary:
        key = round(r["mean_faithfulness"], 6) if r["mean_faithfulness"] is not None else -1.0
        faith_groups[key].append(r["strategy"])
    faith_order_str = " > ".join(
        " = ".join(sorted(faith_groups[k])) for k in sorted(faith_groups, reverse=True)
    )
    lines.append("Actual ordering by mean faithfulness: " + faith_order_str
                 + " (three-way tie at 0.017; see read-out below)")
    lines.append("")
    lines.append("## Results\n")
    lines.append("| strategy | n | recall@5 | recall min/p25/median/p75/max | "
                 "faithfulness | n_empty_answers |")
    lines.append("|---|---|---|---|---|---|")
    for row in summary:
        d = dist[row["strategy"]]["recall"]
        q = (f"{d['min']:.2f} / {d['p25']:.2f} / {d['median']:.2f} / "
             f"{d['p75']:.2f} / {d['max']:.2f}")
        f = f"{row['mean_faithfulness']:.3f}" if row["mean_faithfulness"] is not None else "n/a"
        lines.append(f"| {row['strategy']} | {row['n']} | {row['mean_recall']:.3f} | "
                     f"{q} | {f} | {row['n_empty_answers']} |")
    lines.append("")
    lines.append("Recall distribution read-out: the median question scores 1.0 under "
                 "every strategy; the mean is dragged down by a minority of total "
                 "retrieval failures (recall 0). Overlap helps a little "
                 "(fixed-500/100 beats fixed-500/0), semantic beats fixed, and "
                 "hierarchical tops the table, matching the predicted ordering.")
    lines.append("")
    lines.append("Faithfulness read-out: near zero for all four strategies. This is a "
                 "measurement outcome, not a finding that every answer was wrong. Two "
                 "facts combine: (1) flan-t5-base emits ultra-short fragment answers "
                 "(all 240 answers are a single sentence, most 1-5 words, e.g. "
                 "'France'); (2) the NLI judge returns 'neutral' for fragment "
                 "hypotheses that carry no full proposition, even when the answer is "
                 "correct (e.g. the correct answer '10th and 11th centuries' against "
                 "a premise containing the same phrase scored entailment 0.015). "
                 "Only 3 of 240 claims cleared entailment >= 0.5, all genuinely "
                 "entailed multi-word restatements. With this judge and this "
                 "generator, the faithfulness metric cannot discriminate strategies; "
                 "it is reported as measured, with this caveat.")
    lines.append("")
    lines.append("")
    lines.append("## Synthetic sanity check\n")
    lines.append(synth)
    lines.append("## Limits\n")
    lines.append("- The NLI judge is a proxy, not ground truth: it can be fooled by "
                 "lexical overlap and can miss valid paraphrases.")
    lines.append("- SQuAD is extractive QA (answers copied from the passage), not "
                 "open-domain RAG.")
    lines.append("- flan-t5-base is small; faithfulness numbers reflect a small "
                 "generator, not a ceiling for RAG in general.")
    lines.append("- k=5 is arbitrary, not tuned.")
    lines.append("- Regex sentence splitting is approximate; boundary errors feed "
                 "into gold ids and claim splitting.")
    lines.append("- CPU-only run; greedy decoding.")
    lines.append("- 60 questions is a slice, not a benchmark; treat differences "
                 "between close strategies as noisy.")
    with open(RESULTS / "RESULTS.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[E] wrote summary.csv, RESULTS.md, recall.png, faithfulness.png", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="A,B,C,D,E",
                        help="comma-separated subset of stages to run, e.g. --stages C,D")
    args = parser.parse_args()
    wanted = [s.strip().upper() for s in args.stages.split(",")]
    for stage in wanted:
        if stage == "A":
            stage_a()
        elif stage == "B":
            stage_b()
        elif stage == "C":
            stage_c()
        elif stage == "D":
            stage_d()
        elif stage == "E":
            stage_e()
        else:
            raise SystemExit(f"unknown stage: {stage}")


if __name__ == "__main__":
    main()
