# PRD: RAG Chunking Evaluation Harness

## Problem

Retrieval-augmented generation (RAG) works by splitting source documents into
pieces (chunks), retrieving the pieces most similar to a user question, and
handing them to a language model that writes the answer. The splitting step,
called chunking, is the least studied part of the pipeline. Practitioners pick
a chunk size and overlap by feel. There is no small, reproducible experiment
that shows how that choice changes retrieval quality and answer quality. This
project builds that experiment.

## What we are building

A harness that takes a set of documents and questions, runs the same RAG
pipeline four times with only the chunking strategy changed, and reports two
numbers per strategy: recall@k and faithfulness.

Recall@k is the share of a question's gold evidence that appears in the top k
retrieved chunks. Faithfulness is the share of sentences in the generated
answer that are logically supported by the retrieved context, as judged by a
natural language inference (NLI) model. NLI means the judge reads a piece of
context (the premise) and a claim sentence (the hypothesis) and decides
whether the premise supports the claim, contradicts it, or neither.

Everything except the chunker is held constant: the same embedder, the same
generator, the same k=5, the same judge. The four strategies under test are
fixed-size 500 characters with no overlap, fixed-size 500 characters with
100-character overlap, semantic splitting on sentence similarity, and
hierarchical retrieval with parent expansion. Each is described fully in the
TRD.

## Users

The reader is a working engineer who builds or evaluates RAG systems and
wants an honest, reproducible way to compare chunking choices. A secondary
user is the resume audience: the single claim this repo will carry is "built a
RAG chunking evaluation harness; measured recall and faithfulness across
chunking strategies on SQuAD 2.0." That claim must stay true no matter how
the numbers turn out.

## Method in brief

We follow a Russian-doll order: small inside, real outside. First a synthetic
toy corpus (20 short documents, 40 questions, facts planted by hand, gold
answers known in advance). The harness must behave sanely here, and we state a
predicted strategy ordering before running. Then a real public slice: 60
answerable questions from the SQuAD 2.0 development set. Only after both
stages pass do we run the full experiment and publish results.

## Success criteria

1. The pipeline runs end to end on a CPU-only machine with no paid APIs.
2. Results are reproducible: same corpus, same config, same numbers.
3. Every result ships with its config (strategies.yaml), raw output
   (results.json), and a written summary (RESULTS.md).
4. The stated prediction is either confirmed or reported as falsified. A
   falsified prediction still counts as success; the point is the measurement,
   not the outcome.
5. All known limits are written down (see TRD). Nothing is oversold.

## Non-goals

- We are not finding the globally best chunker. We compare four specific
  strategies on two corpora.
- We are not tuning the embedder, the generator, or k. They are fixed control
  variables.
- We are not building a production RAG service. There is no API, no UI, no
  latency measurement.
- We are not using human judges. Faithfulness is measured by an NLI model,
  which is a proxy. Its limits are documented in the TRD.
