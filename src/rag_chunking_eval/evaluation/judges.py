"""NLI judges for the faithfulness metric.

split_claims turns a generated answer into atomic claims (one per
sentence). Judges score each claim against the retrieved context:
True = supported (entailed), False = not supported.
"""

from __future__ import annotations

from rag_chunking_eval.text import split_sentences


def split_claims(answer: str) -> list[str]:
    """Split a generated answer into one claim per sentence."""
    return [text for _, _, text in split_sentences(answer)]


class StubNLIFaithfulnessJudge:
    """Deterministic judge for tests: looks each claim up in a fixed dict.

    Raises KeyError on an unknown claim so tests fail loudly instead of
    silently judging something unexpected.
    """

    def __init__(self, claim_to_supported: dict[str, bool]):
        self.claim_to_supported = dict(claim_to_supported)

    def score(self, premise: str, claims: list[str]) -> list[bool]:
        return [self.claim_to_supported[claim] for claim in claims]


class HFNLIJudge:
    """Real judge: cross-encoder NLI over (premise, claim) pairs.

    Uses cross-encoder/nli-deberta-v3-small by default. Imports are lazy so
    unit tests never need transformers or torch installed. Only used in the
    real experiment run, never in tests.

    A claim counts as supported when entailment probability >= 0.5 and
    contradiction probability < 0.5. The premise is truncated to ~400 tokens
    with the model's own tokenizer so long contexts still fit.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.tokenizer = self.model.tokenizer
        labels = [self.model.model.config.id2label[i] for i in range(3)]
        self.entailment_idx = labels.index("entailment")
        self.contradiction_idx = labels.index("contradiction")

    def _truncate(self, text: str, max_tokens: int = 400) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        return self.tokenizer.decode(ids)

    def score(self, premise: str, claims: list[str]) -> list[bool]:
        import numpy as np

        premise = self._truncate(premise)
        logits = np.asarray(self.model.predict([(premise, c) for c in claims]))
        # Softmax over the three NLI labels.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        entail = probs[:, self.entailment_idx]
        contra = probs[:, self.contradiction_idx]
        return [bool(e >= 0.5 and c < 0.5) for e, c in zip(entail, contra)]
