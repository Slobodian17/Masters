"""Text normalisation for queries and table labels.

Shared by the structural index and the scope gate. Retrieval accuracy depends
on this tokeniser, so changes to a profile's suffix or stopword lists require
re-running `python -m eval.eval_retrieval` for that language.

The lists themselves live in `core.lang`; only the mechanism is here.
"""
import re
from functools import lru_cache

from core import lang

_WORD_RE = re.compile(r"[0-9]+|[^\W\d_]+", re.UNICODE)


@lru_cache(maxsize=None)
def _rules(code: str):
    p = lang.get(code)
    return tuple(p.SUFFIXES), p.STEM_MIN_KEEP, bool(p.STEM_ITERATE), p.STOPWORDS, p.MIN_TOKEN


def _strip_once(word: str, suffixes: tuple, min_keep: int) -> str:
    for suf in suffixes:
        if len(word) - len(suf) >= min_keep and word.endswith(suf):
            return word[: -len(suf)]
    return word


@lru_cache(maxsize=100_000)
def stem(word: str, code: str = None) -> str:
    suffixes, min_keep, iterate, _, _ = _rules(code or lang.active_lang())
    if not suffixes:
        return word
    out = _strip_once(word, suffixes, min_keep)
    if not iterate:
        return out
    # English strips to a fixed point so that "proceeds" and "proceed" meet at
    # the same stem. Ukrainian stays on one pass to keep its frozen baseline
    # reproducible; see the note in its profile.
    while True:
        nxt = _strip_once(out, suffixes, min_keep)
        if nxt == out:
            return out
        out = nxt


def tokenize(text: str, code: str = None) -> set:
    """Text -> set of normalised tokens.

    Digits are kept verbatim: thresholds such as '150000' come straight from
    the query, and years and quarters are handled by `temporal_index` but do no
    harm here.
    """
    code = code or lang.active_lang()
    _, _, _, stopwords, min_token = _rules(code)
    out = set()
    for w in _WORD_RE.findall(str(text).lower()):
        if w in stopwords:
            continue
        if w.isdigit():
            # The minimum length applies to words, not to digits. It used to
            # apply to both, which silently erased the only thing telling
            # "tier 1 capital" from "tier 2 capital" -- measured on FinQA,
            # 4.6% of tables hold distinct rows that then tokenise alike.
            out.add(w)
            continue
        if len(w) < min_token:
            continue
        out.add(stem(w, code))
    return out


def reset_cache():
    """Drop cached rules and stems; needed when a test switches language."""
    _rules.cache_clear()
    stem.cache_clear()
