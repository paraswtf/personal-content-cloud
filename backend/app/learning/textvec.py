"""TF-IDF vectors for the text learners (document types, tags): the same tokenisation as the
document parsers (Devanagari-aware), unigrams and bigrams, add-one smoothed idf, cosine over
L2-normalised sparse vectors capped to their strongest terms."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

CAP = 400
MAX_TOKENS = 20000


def tokens(text: str | None) -> list[str]:
    """Lowercased word tokens plus adjacent bigrams ('a b'), at most MAX_TOKENS words."""
    from app.documents.fields import normalize_text, word_tokens

    words = [w.lower() for w in word_tokens(normalize_text(text or ""))][:MAX_TOKENS]
    out = list(words)
    out.extend(f"{a} {b}" for a, b in zip(words, words[1:]))
    return out


def build_idf(docs: Iterable[Iterable[str]]) -> tuple[dict[str, float], int]:
    """(idf per term, number of documents) with add-one smoothing: log((N + 1) / (df + 1)) + 1."""
    df: Counter[str] = Counter()
    n = 0
    for doc in docs:
        n += 1
        df.update(set(doc))
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    return idf, n


def vector(toks: Iterable[str], idf: dict[str, float], n_docs: int, cap: int = CAP) -> dict[str, float]:
    """L2-normalised tf-idf of the tokens (unseen terms weigh 1.0). Keeps the `cap` strongest terms."""
    tf = Counter(toks)
    if not tf:
        return {}
    # An unseen term says nothing about any exemplar: it keeps a neutral weight (1.0) so a few new
    # words do not drown the terms the exemplars share.
    weights = {t: (1.0 + math.log(c)) * idf.get(t, 1.0) for t, c in tf.items()}
    if len(weights) > cap:
        keep = sorted(weights.items(), key=lambda kv: -kv[1])[:cap]
        weights = dict(keep)
    norm = math.sqrt(sum(w * w for w in weights.values()))
    if norm <= 0:
        return {}
    return {t: w / norm for t, w in weights.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return float(sum(w * b[t] for t, w in a.items() if t in b))
