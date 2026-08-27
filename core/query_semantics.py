"""Detects whether a query asks for a relative value (percent, share, margin).

Used by the range signal in `verifier` and by the mock semantic check: a margin
of 608% is clearly wrong, while a sum of 608,000 is ordinary.

Naive prefix matching on "відсот" is not enough. Ukrainian uses the same word
for "percent" and for "interest on a loan", so "сплачені відсотки за
кредитами" is an absolute amount, not a rate. The explicit patterns below keep
the two apart.
"""
import re

# Patterns that mark a query as asking for a relative value.
_RELATIVE_PATTERNS = (
    r"%",
    r"\bу\s+відсотк",
    r"\bв\s+відсотк",
    r"на\s+скільки\s+відсотк",
    r"відсотк\w*\s+(?:зрост|приріст|зменш|падінн)",
    r"маржа",
    r"маржинальн",
    r"рентабельн",
    r"частк",
    r"приріст",
    r"зростанн",
    r"темп\w*\s+зрост",
)

_RELATIVE_RE = re.compile("|".join(_RELATIVE_PATTERNS), re.IGNORECASE | re.UNICODE)

# Contexts where "відсотки" means loan interest. Checked first, takes priority.
_INTEREST_PATTERNS = (
    r"відсотк\w*\s+за\s+(?:кредит|позик|депозит|облігац)",
    r"(?:сплачен|нарахован|отриман)\w*\s+відсотк",
    r"відсотков\w*\s+(?:витрат|дохід|ставк)",
)

_INTEREST_RE = re.compile("|".join(_INTEREST_PATTERNS), re.IGNORECASE | re.UNICODE)


def is_interest_query(text: str) -> bool:
    """True when "відсотки" means loan interest, i.e. a monetary amount."""
    return bool(_INTEREST_RE.search(str(text)))


def is_relative_query(text: str) -> bool:
    """True when the answer is expected as a percentage or share."""
    s = str(text)
    if is_interest_query(s):
        # An explicit "%" still signals the unit even in a loan-interest query.
        return "%" in s
    return bool(_RELATIVE_RE.search(s))
