# TRD: RAG Chunking Evaluation Harness

## 1. Design goal

Isolate the effect of chunking strategy on two things: how much gold evidence
gets retrieved (recall) and how much of the generated answer the evidence
actually supports (faithfulness). Everything else is frozen. The experiment is
a controlled comparison, not a leaderboard.

## 2. Pipeline

For each strategy, the pipeline runs these steps in order:

1. Chunk the corpus documents with the strategy under test.
2. Embed every chunk with the fixed embedder and store the vectors in a flat
   index (brute-force cosine search; corpora are small, no ANN library
   needed).
3. For each question, embed the question, retrieve the top k=5 chunks.
4. Build a prompt from the retrieved chunks plus the question, and generate
   an answer with the fixed generator.
5. Score recall@k against the gold annotations (Section 5).
6. Score faithfulness of the answer with the NLI judge (Section 6).

Steps 1 through 4 are the RAG system. Steps 5 and 6 are the measurement.

## 3. Controlled variables

These are identical across all four strategies. If any of them changes, the
run is invalid and must be redone.

- Embedder: sentence-transformers `all-MiniLM-L6-v2`, run locally on CPU.
  A small, fast embedding model that turns text into vectors.
- Generator: `google/flan-t5-base`, run locally on CPU. A modest
  sequence-to-sequence model that writes answers from the retrieved context.
  Input is capped at 450 tokens, cut with the T5 tokenizer; only the context
  side is cut (prompt framing and question are always kept), and the
  truncation is logged per strategy.
- k = 5. Every question retrieves exactly five chunks.
- NLI judge: `cross-encoder/nli-deberta-v3-small`. A cross-encoder reads the
  premise and hypothesis together and outputs probabilities for entailment,
  neutral, and contradiction.
- Prompt template: identical wording for every strategy.

## 4. Strategies under test

Only the chunking changes. Definitions:

- **fixed-500/0**: split text into 500-character windows, no overlap.
- **fixed-500/100**: 500-character windows with 100 characters of overlap
  between consecutive windows. This is an ablation: it differs from fixed-500/0
  only in overlap, so any difference in score is attributable to overlap
  alone.
- **semantic**: split text into sentences, embed adjacent sentences, and cut
  the chunk where cosine similarity between neighboring sentences falls below
  0.5. Chunks follow topic boundaries instead of character counts.
- **hierarchical**: store child chunks of about 250 characters. Retrieval
  scores the children. The top-k children are then expanded to their parent
  contexts of about 1000 characters, and the parents are what the generator
  sees. This separates retrieval granularity (small, precise) from generation
  context (large, rich).

All four are configured in `configs/strategies.yaml`. Adding a strategy means
adding one entry there plus one chunker implementation; nothing else changes.

## 5. Corpora

### 5.1 Synthetic corpus (stage one)

20 short documents written for this experiment. 40 questions. Key facts are
planted by hand, so every question has a known answer. The harness is sanity
checked here: if recall on hand-planted facts is bad, the bug is in the
harness, not the chunker. We also state our predicted strategy ordering
before running (Section 8). The synthetic corpus is a toy. It proves the
harness works. It proves nothing about real documents.

### 5.2 SQuAD 2.0 slice (stage two)

SQuAD 2.0 is a public reading-comprehension dataset: Wikipedia passages,
questions, and answer spans. We take 60 answerable questions from its
development set. Each question's context passage becomes a document. Gold
evidence is derived, not hand-labeled: for each question we take the answer
span's character offset (`answer_start`), find which sentence contains it, and
mark that sentence as gold. SQuAD is extractive QA (answers are copied from
the passage), not real open-domain RAG where answers are synthesized. Say so
when reporting.

### 5.3 Gold standard design

Annotations live at the sentence level, not the chunk level. A sentence id
looks like `doc42:s3`: document 42, sentence 3. Sentence ids are strategy
independent, which is the whole point. Chunk ids depend on the strategy, so
gold labels can never be chunk ids.

After chunking, each chunker records, for every chunk, the sentence ids it
contains, computed by character-offset overlap between the chunk and the
sentence spans. Recall@k for a question is then the fraction of that
question's gold sentence ids that are covered by the top-k retrieved chunks.
A sentence counts as covered only if it lies fully inside at least one
retrieved chunk. Partial overlap does not count.

For hierarchical, recall is measured on the expanded parent context actually
fed to the generator, because that is what the strategy delivers. Measuring
it on the unexpanded children would punish the strategy for a design choice
that is part of the strategy.

## 6. Metrics

### 6.1 Recall@k

Defined in Section 5.3. Reported as the mean over questions, plus the
distribution (min, quartiles, max). A single mean hides questions where
retrieval failed completely; the distribution does not.

### 6.2 Faithfulness

For each question with a non-empty generated answer:

1. Split the answer into sentences. Each sentence is one claim.
2. Form the premise by concatenating the top-k retrieved chunks in rank
   order.
3. For each claim sentence, run the NLI judge with premise and hypothesis.
   A claim is supported when entailment probability is at least 0.5 and
   contradiction probability is below 0.5.
4. The question's score is the fraction of its claims that are supported.

Faithfulness is the mean of question scores over questions with non-empty
answers. Report n (the number of questions scored). Questions with empty
answers are counted separately and excluded, not scored as zero. Scoring an
empty answer as zero would conflate "the model said nothing" with "the model
hallucinated," which are different failures.

## 7. Model and environment choices

- The embedder and generator are small and local so the full run finishes on
  a CPU-only machine with no API keys and no cost.
- flan-t5-base is small. Its answers are weaker than a large model would
  produce. Faithfulness numbers reflect what a small generator does with the
  retrieved context, not a ceiling for RAG in general.
- k=5 is arbitrary. It is a common default, not a tuned value. Do not claim
  the results generalize to other k.
- Sentence splitting is regex-based and approximate. Sentence boundary errors
  propagate into gold ids and claim splitting. This is a known noise source.
- The NLI judge is a proxy, not ground truth. It is a cross-encoder trained
  on MNLI-style data. It can be fooled by lexical overlap: a claim that
  reuses the context's words can score as entailed without being logically
  supported, and a correctly paraphrased claim can score as neutral. Treat
  faithfulness as a directional signal, not a verdict.

## 8. Predicted ordering (falsifiable)

Stated before the synthetic run, applied to both recall and faithfulness:

hierarchical >= semantic > fixed-500/100 > fixed-500/0

Reasoning: hierarchical gets the best of both sizes, semantic respects topic
boundaries, overlap helps fixed windows a little, and no-overlap windows
split facts at arbitrary points. Report the actual numbers either way. If the
ordering breaks, the report says so and we investigate, not rationalize.

## 9. Threats to validity

- **Corpus size.** 60 SQuAD questions is a slice, not a benchmark. Confidence
  intervals will be wide. Report them.
- **Extractive bias.** SQuAD answers are substrings of the passage. This
  favors strategies that retrieve exact sentences and may understate the
  value of broader context. Real RAG questions are often not extractive.
- **Gold derivation noise.** The sentence containing the answer span is a
  heuristic. If the true evidence spans two sentences, we under-label.
- **Judge bias.** Section 7 covers the NLI judge's failure modes. A second
  judge (or a human spot check on a sample) would strengthen the claim; it is
  out of scope for this build and listed as future work.
- **Single k, single embedder, single generator.** Any of these could
  interact with chunking strategy. We measure one point in that space and say
  so.
- **Truncation.** The 450-token context cap means hierarchical's
  larger parents are more likely to be truncated. Truncation events are
  logged per strategy so this effect is visible, not hidden.

## 10. Repo layout

```
rag-chunking-eval/
  src/rag_chunking_eval/
    chunking/      # one module per strategy; each fills sentence_ids per chunk
    corpus/        # synthetic builder, SQuAD loader, gold derivation
    evaluation/    # retrieval, generation, recall, faithfulness, reporting
    text.py        # shared text utils (sentence splitting, offsets)
  tests/           # unit tests for chunkers, gold mapping, metrics
  configs/
    strategies.yaml  # the four strategies and their parameters
  results/
    synthetic_real_embedder.json  # stage A: recall summary on the toy corpus
    synthetic_check.md            # stage A: predicted vs actual ordering
    squad_slice.json              # stage B: the 60 SQuAD questions + doc count
    retrieval.json                # stage B: 240 records (60 q x 4 strategies)
    answers.json                  # stage C: 240 records with generated answers
    results.json                  # stage D: final records + "run" metadata block
    summary.csv                   # stage E: per-strategy means
    RESULTS.md                    # stage E: generated report (prediction, actuals, limits)
    recall.png                    # stage E: bar chart, flat in results/
    faithfulness.png              # stage E: bar chart, flat in results/
    RUNINFO.md                    # pip freeze captured at run time

`results.json` is the source of truth. `summary.csv`, `RESULTS.md`, and the
two PNGs are derived from it and must be regenerable from it plus
`configs/strategies.yaml` alone. There is no `results/plots/` subdirectory:
the charts are written flat into `results/`.
  docs/            # PRD.md, TRD.md, EXECUTION.md (this file set)
  README.md        # what the repo is and how to run it
```

`results.json` is the source of truth. `summary.csv` and `RESULTS.md` are
derived from it and must be regenerable from it alone.

## 11. Reproducibility contract

- Fixed random seeds wherever randomness exists.
- `configs/strategies.yaml` plus the corpus version fully determine a run.
- Model downloads happen once and are cached locally; the run itself needs
  no network.
- RESULTS.md records the date, the machine description (CPU-only), the model
  versions, and the exact command used.

## 12. Exact environment

The run box is fixed. If the box changes, say so in RESULTS.md.

- Python 3.12.3, virtualenv at `.venv` in the repo root.
- 2 CPUs, about 7 GB RAM, CPU-only. No GPU.
- torch must come from the CPU index:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
  A PyPI install pulls a CUDA wheel (gigabytes of dead weight). The known
  good state is torch 2.14.0+cpu.
- `TMPDIR=~/pip-tmp` is mandatory on every pip call. `/tmp` is a 512 MB
  tmpfs; pip fails without the override. `~/pip-tmp` must exist first.
- Dependency versions are captured at run time with
  `.venv/bin/pip freeze > results/RUNINFO.md`. This is part of the setup
  procedure in EXECUTION.md step 0, not an afterthought.
- The run script strips bracketed IPv6 literals from no_proxy/NO_PROXY at
  startup. httpx 0.28 (used by huggingface-hub) breaks on values like
  `[::1]`; the script handles it, so do not "fix" your environment to match.

## 13. Pre-flight checklist

Every item must be green before any long run starts. No launch on red.

- (a) Environment matches Section 12. Check: `.venv/bin/python --version`
  prints 3.12.3, `.venv/bin/python -c "import torch; print(torch.__version__)"`
  ends in `+cpu`, and `torch.cuda.is_available()` is False.
- (b) Credentials valid for the push step: `gh_publish.py check` prints OK
  login. The run itself needs no credentials; the repo push afterwards does.
- (c) Data reachable: the SQuAD URL
  `https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json`
  answers, or `.data/dev-v2.0.json` already exists from a previous download.
  Same for the HuggingFace Hub (needed once per model for the first weight
  download).
- (d) Small-scale dry run passes end to end: `pytest` green (48 tests,
  including the synthetic end-to-end test with stub embeddings) plus
  `--stages A` completes and writes `results/synthetic_check.md`.

## 14. Failure modes and fixes

| Symptom | Fix |
|---|---|
| HuggingFace Hub unreachable (model download hangs or errors) | Verify network, retry. If it persists, report a blocker; do not fake the weights. |
| pip fails with `/tmp` full | Re-run with `TMPDIR=~/pip-tmp`; create the dir first. |
| OOM on the 7 GB box | Load one model at a time; the stages already `del` each model and call `gc.collect()` between stages. Keep it that way. |
| A CUDA torch wheel got installed by accident | Uninstall and reinstall from the CPU index in Section 12. Check with `torch.cuda.is_available()`. |
| SQuAD download fails | Retry; check the URL in Section 13(c). The `.data/` cache makes this a one-time risk. |
| flan-t5 generation is slow on CPU | Expected. 240 generations, sequential, on 2 cores. Plan wall time around it; do not parallelize naively. |
| Wrong NLI label for entailment/contradiction | Read `model.config.id2label` at runtime; never hardcode indices. The run script already does this. |
| Empty generated answers | They are counted (`n_empty_answers`) and excluded from the faithfulness mean, not scored as zero. Do not change this rule to make numbers look better. |
| Non-reproducible sampling | Generation is greedy (`do_sample=False`); synthetic corpus uses seed=7; the SQuAD slice is built in deterministic order. |

## 15. Verification criteria (gates)

Each phase has a gate. A phase is done only when its gate passes.

- Unit tests: `pytest` green, 48 tests (bar: 43+; the count may grow, never shrink silently).
- Stage A: `results/synthetic_real_embedder.json` and
  `results/synthetic_check.md` exist; the check file records the predicted
  ordering and the actual synthetic ordering. A four-way tie is a valid
  outcome, not a failure.
- Stage B: `results/retrieval.json` holds 240 records (60 questions x 4
  strategies); every record has exactly 5 `retrieved_ids`.
- Stage C: `results/answers.json` holds 240 records; context truncations are
  counted per strategy.
- Stage D: `results/results.json` written; empty answers have
  `faithfulness: null` and are excluded from means.
- Analysis: `scripts/analyze_results.py` exits 0. (Known open bug, Sept 28:
  it expects a bare records list while stage D writes a `{"run","records"}`
  envelope; see EXECUTION.md step 5 for the workaround until fixed.)
- Plots: `results/recall.png` and `results/faithfulness.png` exist at the
  top level of `results/`, are non-empty PNGs, and regenerate from
  `summary.csv`. (Known open bug, Sept 28: `make_plots.py` expects different
  CSV column names than stage E writes; see EXECUTION.md step 6.)
- Report: `RESULTS.md` states n=60, the prediction, both actual orderings,
  the per-strategy numbers with `n_empty_answers`, and the limits list.

## 16. Rollback plan

- Every phase ends in an atomic git commit. Commit the code change and its
  test result, not a pile of unrelated work.
- Rollback means returning to the last green commit: `git revert` (or
  `git checkout <commit> -- .` for files) and re-running the gates in
  Section 15.
- `results/` is fully regenerable from `results.json` plus
  `configs/strategies.yaml` (`--stages E` rebuilds the summary, report, and
  both PNGs). Regenerating `results.json` itself requires re-running stages
  B-D; there is no shortcut for that, so guard the green `results.json` like
  it is expensive, because it is.
