"""Detects whether a query asks for a relative value (percent, share, margin).

Used by the range signal in `verifier` and by the mock semantic check: a margin
of 608% is clearly wrong, while a sum of 608,000 is ordinary.

The two pattern families come from the active language profile. Their sizes
differ for a reason worth recording: Ukrainian uses one word, "відсотки", for
both a rate and interest paid on a loan, so the interest family exists purely
to pull those queries back out of the relative branch. English has separate
words and that family is empty.
"""
from core import lang


def _relative_re():
    return lang.compile_any(tuple(lang.get().RELATIVE_PATTERNS))


def _interest_re():
    return lang.compile_any(tuple(lang.get().INTEREST_PATTERNS))


def is_interest_query(text: str) -> bool:
    """True when a percent-word in this query means money, not a rate."""
    return bool(_interest_re().search(str(text)))


def is_relative_query(text: str) -> bool:
    """True when the answer is expected as a percentage or share."""
    s = str(text)
    if is_interest_query(s):
        # An explicit "%" still signals the unit even in a loan-interest query.
        return "%" in s
    return bool(_relative_re().search(s))
