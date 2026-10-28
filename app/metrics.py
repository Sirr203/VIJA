"""Evaluate WER (Word Error Rate) and CER (Character Error Rate)."""
import re
import unicodedata
from typing import Callable, Sequence


def _edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def normalize_chars(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w]|_", "", text)
    return list(text)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_chars(reference), normalize_chars(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def metric_for(language: str) -> tuple[str, Callable[[str, str], float]]:
    """Returns ('CER', cer) for Japanese, ('WER', wer) for other languages."""
    if language.strip().lower() in ("japanese", "ja"):
        return "CER", cer
    return "WER", wer
