"""Retrieval index 2/3: temporal coverage.

Financial tables are indexed by period and queries almost always name one, so
a table that does not cover the requested period is unlikely to be the right
one. Kept separate from the lexical index because it acts as a filter: missing
coverage is strong negative evidence, and it is the only signal that separates
tables with identical structure but different years.

A query naming no period scores neutrally for every table rather than zero,
since zero would falsely claim no table fits.
"""
import re

NEUTRAL = 0.5

_QUARTER_RE = re.compile(r"\bq\s*([1-4])[\s_\-/]*((?:19|20)\d{2})\b", re.IGNORECASE)
_QUARTER_ALT_RE = re.compile(r"\b((?:19|20)\d{2})[\s_\-/]*q\s*([1-4])\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def parse_periods(text: str) -> dict:
    """Extract periods mentioned in the text.

    A year appearing only inside a quarter is not added separately, otherwise
    "Q3 2024" would match any table containing any quarter of 2024.
    """
    s = str(text)
    quarters = set()
    consumed = []

    for m in _QUARTER_RE.finditer(s):
        quarters.add((int(m.group(2)), int(m.group(1))))
        consumed.append(m.span())
    for m in _QUARTER_ALT_RE.finditer(s):
        quarters.add((int(m.group(1)), int(m.group(2))))
        consumed.append(m.span())

    years = set()
    for m in _YEAR_RE.finditer(s):
        if any(a <= m.start() and m.end() <= b for a, b in consumed):
            continue
        years.add(int(m.group(1)))

    return {"quarters": quarters, "years": years}


def table_periods(df) -> dict:
    """Periods covered by the table, read from its column headers."""
    quarters, years = set(), set()
    for col in df.columns:
        p = parse_periods(col)
        quarters |= p["quarters"]
        years |= p["years"]
    years |= {y for y, _ in quarters}  # a table with Q3_2024 also covers 2024
    return {"quarters": quarters, "years": years}


def score(query: str, df) -> float:
    """Share of the query's periods covered by the table, in [0, 1]."""
    q = parse_periods(query)
    if not q["quarters"] and not q["years"]:
        return NEUTRAL

    t = table_periods(df)
    wanted, covered = 0, 0

    for qq in q["quarters"]:
        wanted += 1
        if qq in t["quarters"]:
            covered += 1
    for y in q["years"]:
        wanted += 1
        if y in t["years"]:
            covered += 1

    return covered / wanted if wanted else NEUTRAL


def rank(query: str, tables: dict) -> list:
    scored = [(name, score(query, df)) for name, df in tables.items()]
    return sorted(scored, key=lambda kv: (-kv[1], kv[0]))
