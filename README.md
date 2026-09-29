# rag-chunking-eval

A small harness that measures how the chunking strategy changes RAG quality.
It chunks the same documents four ways, retrieves with one fixed embedder,
generates answers with one fixed generator, and scores recall of the gold
evidence plus faithfulness of each answer to its context.

Built a RAG chunking evaluation harness; measured recall and faithfulness
across chunking strategies on SQuAD v2 (60-question dev slice).

## Method (frozen controls)

Only the chunking changes between strategies. Everything else is fixed:

- Embedder: `sentence-transformers/all-MiniLM-L6-v2`, local CPU.
- Generator: `google/flan-t5-base`, greedy decoding, local CPU.
- k = 5 retrieved chunks per question.
- NLI judge: `cross-encoder/nli-deberta-v3-small`.
- Same prompt template for every strategy.
- Gold evidence is sentence-level and strategy independent; recall@5 is the
  fraction of a question's gold sentences covered by the top-5 expanded chunks.

## Results (SQuAD v2 dev slice, n=60)

| strategy | n | recall@5 | faithfulness | n_empty_answers |
|---|---|---|---|---|
| fixed-500-0 | 60 | 0.800 | 0.017 | 0 |
| fixed-500-100 | 60 | 0.817 | 0.017 | 0 |
| semantic-0.5 | 60 | 0.867 | 0.017 | 0 |
| hierarchical-250-1000 | 60 | 0.917 | 0.000 | 0 |

Recall quartiles (min/p25/median/p75/max) are 0.00 / 1.00 / 1.00 / 1.00 / 1.00
for every strategy: the median question scores perfectly, and the mean is
dragged down by a minority of total retrieval failures.

The faithfulness numbers need a caveat: the generator emits ultra-short
fragment answers and the NLI judge returns "neutral" for fragments, so the
metric cannot discriminate strategies here. Full discussion in
results/RESULTS.md.

Prediction before the run (from docs/TRD.md): hierarchical >= semantic >
fixed-500/100 > fixed-500/0. It held on recall. See results/RESULTS.md for
the full read-out, including why faithfulness could not discriminate.

## Limits

- The NLI judge is a proxy, not ground truth. It can be fooled by word
  overlap and can miss valid paraphrases.
- SQuAD is extractive QA (answers are copied from the passage), not
  open-domain RAG where answers are written fresh.
- flan-t5-base is small. Faithfulness numbers show what a small generator
  does with retrieved context, not a ceiling for RAG in general.
- k=5 is arbitrary, not tuned. Results may not carry to other k.
- Sentence splitting is regex based and approximate. Boundary errors feed
  into gold ids and claim splitting.
- 60 questions is a slice, not a benchmark. Treat close scores as noisy.
- CPU-only run, greedy decoding.

## Repo layout

```
rag-chunking-eval/
  src/rag_chunking_eval/
    chunking/      # fixed, semantic, hierarchical chunkers
    corpus/        # synthetic builder, SQuAD slice loader
    evaluation/    # ExperimentRunner, metrics, NLI judges
    text.py        # regex sentence splitting with offsets
  configs/strategies.yaml  # the four chunking strategies
  scripts/run_experiment.py  # the full run, staged A-E
  tests/           # unit tests (chunkers, metrics, end to end)
  results/         # results.json, summary.csv, RESULTS.md, plots
  docs/            # PRD.md, TRD.md, EXECUTION.md
```

Raw SQuAD data lives in `.data/` (git-ignored). `results/results.json`
is the source of truth; `summary.csv` and `RESULTS.md` are derived from it.

## How to run

```bash
cd ~/workspace/rag-chunking-eval
mkdir -p ~/pip-tmp
TMPDIR=~/pip-tmp .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
TMPDIR=~/pip-tmp .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q            # tests first, must be green
.venv/bin/python scripts/run_experiment.py      # full run, stages A-E
```

Stages are resumable: `--stages B` runs only stage B, and each stage
rewrites only its own outputs. The full run needs network once (SQuAD
download, model weights); after that it runs offline.

## Test status

48 tests, all green (`pytest tests/ -q`).
