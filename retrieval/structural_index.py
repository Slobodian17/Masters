"""Retrieval index 1/3: lexical match against a table's structural signature.

Tables in a financial document differ mainly by their row labels, and queries
name those labels almost verbatim, which makes lexical overlap the strongest
and cheapest signal. Semantics are left to the vector index.

Scores are normalised by query length rather than table size, otherwise the
largest table in a document would win by accident.
"""
from core.text_utils import tokenize


def table_signature(df) -> dict:
    """Row-label tokens and header tokens, kept separate.

    A row-label hit is far more informative than a header hit, since headers
    are nearly identical across tables in a report.
    """
    label_tokens = set()
    if df.shape[1]:
        for val in df.iloc[:, 0].tolist():
            label_tokens |= tokenize(val)

    header_tokens = set()
    for col in df.columns:
        header_tokens |= tokenize(col)

    return {"labels": label_tokens, "headers": header_tokens}


def score(query: str, df) -> float:
    """Share of query tokens covered by the table, in [0, 1]."""
    q = tokenize(query)
    if not q:
        return 0.0
    sig = table_signature(df)
    hit_labels = len(q & sig["labels"])
    hit_headers = len(q & (sig["headers"] - sig["labels"]))
    # a row label counts three times as much as a header
    return min(1.0, (3 * hit_labels + hit_headers) / (3 * len(q)))


def rank(query: str, tables: dict) -> list:
    """[(name, score), ...] descending, with a stable order on ties."""
    scored = [(name, score(query, df)) for name, df in tables.items()]
    return sorted(scored, key=lambda kv: (-kv[1], kv[0]))
