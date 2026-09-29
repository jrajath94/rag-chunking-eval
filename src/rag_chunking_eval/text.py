"""Sentence splitting with character offsets.

Regex-based and dependency-free, so unit tests never need a download.
Splits on sentence-ending punctuation followed by whitespace and a capital
letter, digit, or quote. Good enough for the synthetic corpus and SQuAD
contexts; documented as an approximation in the TRD.
"""

import re

_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split text into sentences.

    Returns a list of (char_start, char_end, sentence_text). Offsets are
    tight: they cover the stripped sentence, not surrounding whitespace.
    Deterministic. Empty or whitespace-only text returns [].
    """
    spans: list[tuple[int, int, str]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        piece = text[start : m.start()]
        stripped = piece.strip()
        if stripped:
            s = start + piece.index(stripped)
            spans.append((s, s + len(stripped), stripped))
        start = m.end()
    tail = text[start:]
    stripped = tail.strip()
    if stripped:
        s = start + tail.index(stripped)
        spans.append((s, s + len(stripped), stripped))
    return spans
