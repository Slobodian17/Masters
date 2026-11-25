"""Schema-aware scope gate, applied before query classification.

Decides whether the document contains the data a query asks about. This is a
property of the document, not of the query's wording, so it is kept separate
from the 8-class query type classifier, which never sees the schema.

Reuses the fuzzy-matching approach of the L0 repair in `exceptions_taxonomy`:
if no content term of the query resembles any row label or header, there is
nothing to answer from.

The gate is deliberately conservative — it rejects only when *no* content term
matches. A false rejection costs more than a miss: a query that slips through
still reaches the verification cascade and can abstain there, while a rejected
one gets no chance at all.
"""
import difflib

from core import lang
from core.exceptions_taxonomy import _row_label_candidates
from core.text_utils import tokenize

IN_SCOPE = "IN_SCOPE"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

# The similarity cutoff and the operation vocabulary are per-language:
# Ukrainian needs a looser cutoff because inflected forms of the same term
# differ by several characters, English does not.


def _schema_tokens(tables: dict, extra_text: str = None) -> set:
    """Every token the document actually contains: row labels and headers.

    `extra_text` carries a table caption when one is supplied. It names the
    subject the table itself often leaves out -- rows of dates under a caption
    that says what the dates measure -- so without it the gate refuses
    questions the table can answer.
    """
    tokens = set()
    if extra_text:
        tokens |= tokenize(extra_text)
    for name, df in tables.items():
        tokens |= tokenize(name)
        for col in df.columns:
            tokens |= tokenize(col)
    for label in _row_label_candidates(tables):
        tokens |= tokenize(label)
    return tokens


def _is_operation_word(token: str) -> bool:
    words = lang.get().OPERATION_WORDS
    return any(token.startswith(w) or w.startswith(token) for w in words)


def _content_tokens(query: str) -> set:
    """Query tokens naming the subject rather than the operation or period."""
    out = set()
    for t in tokenize(query):
        if t.isdigit() or _is_operation_word(t):
            continue
        out.add(t)
    return out


def check(query: str, tables: dict, extra_text: str = None) -> dict:
    """Return {"status", "coverage", "matched", "unmatched", "content_tokens"}."""
    content = _content_tokens(query)
    schema = _schema_tokens(tables, extra_text)

    if not content:
        # Nothing subject-specific to judge, e.g. "what is the maximum overall".
        return {"status": IN_SCOPE, "coverage": 1.0, "matched": [],
                "unmatched": [], "content_tokens": []}

    schema_list = sorted(schema)
    matched, unmatched = [], []
    for t in sorted(content):
        hit = difflib.get_close_matches(t, schema_list, n=1,
                                        cutoff=lang.get().TOKEN_SIMILARITY)
        (matched if hit else unmatched).append(t)

    coverage = len(matched) / len(content)
    status = IN_SCOPE if matched else OUT_OF_SCOPE
    return {"status": status, "coverage": round(coverage, 3), "matched": matched,
            "unmatched": unmatched, "content_tokens": sorted(content)}


def is_in_scope(query: str, tables: dict, extra_text: str = None) -> bool:
    return check(query, tables, extra_text)["status"] == IN_SCOPE
