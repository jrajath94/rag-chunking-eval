# RAG chunking evaluation: results

## Run info

- Date (UTC): 2026-09-29T01:06:12.924572+00:00
- Corpus: SQuAD v2.0 dev slice (60 answerable questions)
- Embedder: sentence-transformers/all-MiniLM-L6-v2 (CPU)
- Generator: google/flan-t5-base, greedy (do_sample=False), max_new_tokens=64 (CPU)
- k: 5
- NLI judge: cross-encoder/nli-deberta-v3-small, supported = P(entailment) >= 0.5 and P(contradiction) < 0.5 (CPU)
- Context truncation: context cut with the T5 tokenizer so the full prompt fits 512 tokens (context budget min(450, 512 - framing - question)); 140 of 240 question-strategy pairs were truncated
- Python/torch/transformers/sentence-transformers: 3.12.3 / 2.14.0+cpu / 5.17.0 / 6.1.0
- Machine: Linux-7.0.0-38-generic-x86_64-with-glibc2.39, CPU-only

## Predicted vs actual

Prediction (TRD Section 8, stated before the synthetic run):
hierarchical >= semantic > fixed-500/100 > fixed-500-0

Actual ordering by mean recall@5: hierarchical-250-1000 > semantic-0.5 > fixed-500-100 > fixed-500-0
Actual ordering by mean faithfulness: fixed-500-0 = fixed-500-100 = semantic-0.5 > hierarchical-250-1000 (three-way tie at 0.017; see read-out below)

## Results

| strategy | n | recall@5 | recall min/p25/median/p75/max | faithfulness | n_empty_answers |
|---|---|---|---|---|---|
| fixed-500-0 | 60 | 0.800 | 0.00 / 1.00 / 1.00 / 1.00 / 1.00 | 0.017 | 0 |
| fixed-500-100 | 60 | 0.817 | 0.00 / 1.00 / 1.00 / 1.00 / 1.00 | 0.017 | 0 |
| semantic-0.5 | 60 | 0.867 | 0.00 / 1.00 / 1.00 / 1.00 / 1.00 | 0.017 | 0 |
| hierarchical-250-1000 | 60 | 0.917 | 0.00 / 1.00 / 1.00 / 1.00 / 1.00 | 0.000 | 0 |

Recall distribution read-out: the median question scores 1.0 under every strategy; the mean is dragged down by a minority of total retrieval failures (recall 0). Overlap helps a little (fixed-500/100 beats fixed-500/0), semantic beats fixed, and hierarchical tops the table, matching the predicted ordering.

Faithfulness read-out: near zero for all four strategies. This is a measurement outcome, not a finding that every answer was wrong. Two facts combine: (1) flan-t5-base emits ultra-short fragment answers (all 240 answers are a single sentence, most 1-5 words, e.g. 'France'); (2) the NLI judge returns 'neutral' for fragment hypotheses that carry no full proposition, even when the answer is correct (e.g. the correct answer '10th and 11th centuries' against a premise containing the same phrase scored entailment 0.015). Only 3 of 240 claims cleared entailment >= 0.5, all genuinely entailed multi-word restatements. With this judge and this generator, the faithfulness metric cannot discriminate strategies; it is reported as measured, with this caveat.


## Synthetic sanity check

# Synthetic sanity check (real embedder)

Run at 2026-09-28T23:26:34.049169+00:00 (UTC).

Predicted ordering (from TRD Section 8):

hierarchical >= semantic > fixed-500/100 > fixed-500-0

Actual mean recall@5:

- fixed-500-0: 1.000
- fixed-500-100: 1.000
- semantic-0.5: 1.000
- hierarchical-250-1000: 1.000

Result: four-way tie at recall@5 = 1.000. The toy corpus is too easy to discriminate strategies (each document is ~10 sentences, so every strategy retrieves the planted fact sentence). The ordering is neither confirmed nor falsified here; it will be tested for real on the SQuAD slice below. The prediction is not adjusted after the fact.

## Limits

- The NLI judge is a proxy, not ground truth: it can be fooled by lexical overlap and can miss valid paraphrases.
- SQuAD is extractive QA (answers copied from the passage), not open-domain RAG.
- flan-t5-base is small; faithfulness numbers reflect a small generator, not a ceiling for RAG in general.
- k=5 is arbitrary, not tuned.
- Regex sentence splitting is approximate; boundary errors feed into gold ids and claim splitting.
- CPU-only run; greedy decoding.
- 60 questions is a slice, not a benchmark; treat differences between close strategies as noisy.
