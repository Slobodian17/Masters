"""Verification cascade: four surrogate signals forming the vector v.

v_schema, v_range, v_cross_table and v_semantic replace the test oracle that
does not exist for open numeric queries. The vector feeds `calibration`.
"""
import os
import math
import re

import numpy as np

from core import lang
from core.query_semantics import is_relative_query
from generation.llm_client import call_llm

TOL = 0.005  # rounding tolerance for cross-table matching


def _v_schema(code: str, tables: dict) -> bool:
    """True if the code actually reads a table instead of hardcoding a value.

    Cheap but effective against hallucinated constants.
    """
    names = list(tables.keys()) + (["df"] if tables else [])
    return any(re.search(rf"\b{re.escape(n)}\b", code) for n in names)


def _all_numeric_values(tables: dict) -> np.ndarray:
    vals = []
    for df in tables.values():
        num = df.select_dtypes(include=[np.number]).to_numpy(dtype=float).flatten()
        vals.append(num[~np.isnan(num)])
    return np.concatenate(vals) if vals else np.array([])


def _v_range(v, tables: dict, query: str = "") -> bool:
    if v is None:
        return True
    # Relative values (margin, growth) have a different plausible range than
    # absolute amounts. Without this split a wrong 608% margin passes simply
    # because 608 is below the millions in the table.
    if is_relative_query(query):
        return -100.0 <= v <= 500.0
    all_numeric = _all_numeric_values(tables)
    if not len(all_numeric):
        return True  # nothing to compare against
    lo, hi = float(all_numeric.min()), float(all_numeric.max())
    margin = 0.1 * (abs(hi) + 1)
    return (lo - margin) <= v <= (hi * 5 + margin)  # loose bound for sums


def _v_cross_table(v, tables: dict):
    """True when the same value appears in another table, None otherwise.

    None is neutral rather than negative: not every value has to reappear.
    """
    if v is None:
        return None
    all_numeric = _all_numeric_values(tables)
    if not len(all_numeric):
        return None
    close = np.abs(all_numeric - v) <= TOL * max(abs(v), 1)
    return True if close.any() else None


# Off by default: the confirmed run (5.26) included the signal and must stay
# reproducible as measured.
SKIP_SEMANTIC_ENV = "VKSK_NO_SEMANTIC"


def skip_semantic() -> bool:
    return os.environ.get(SKIP_SEMANTIC_ENV, "0").strip() not in ("", "0", "false", "no")


def _v_semantic(query: str, v, tables: dict):
    """Single yes/no plausibility question to the LLM, not a full check.

    The reply is read with whole-word matching. Substring matching looks
    equivalent and is not: in English "no" hides inside "not", "none",
    "nothing", "cannot" and "know", and in Ukrainian the same holds for
    "ni" inside longer words, so a hedged non-answer would be recorded as a
    confident refutation and the signal would fail silently.
    """
    if skip_semantic():
        # Returning None is not a shortcut: _heuristic_confidence already
        # renormalises over the applicable signals, so an absent signal is a
        # defined state rather than a missing one. Replayed on the 300
        # confirmatory questions (eval.finqa_semantic_ablation), omitting it
        # flipped the accept/reject decision at 0 of 391 steps -- maximum
        # confidence drift 0.111, never across the threshold. So the repair
        # trajectory, and therefore the candidate value, is unchanged, and the
        # call it costs is 25% of a run's token budget.
        return None
    prof = lang.get()
    prompt = prof.SEMANTIC_PROMPT.format(query=query, value=v)
    try:
        resp = call_llm(prompt, tag="semantic")
    except Exception:
        return None
    return lang.parse_yes_no(resp, prof)


def as_scalar(result):
    """-> float, or None when the result is not a single number.

    A one-element container counts: `.values[0]` and `.sum()` legitimately
    return numpy scalars and one-element arrays. A container holding anything
    else does not, and this is the distinction the old check missed --
    `float(df)` raises, the fallback asked only whether `str(df)` was
    non-empty, and an entire DataFrame was recorded as a well-typed answer.
    Measured on FinQA, that let a two-cell frame through with full confidence.
    """
    if result is None or isinstance(result, bool):
        return None
    size = getattr(result, "size", None)
    if size is not None and size != 1:
        return None
    if size is None and hasattr(result, "__len__") and not isinstance(result, str):
        if len(result) != 1:
            return None
        result = list(result)[0]
    try:
        return float(np.asarray(result).reshape(-1)[0]) if size == 1 else float(result)
    except (TypeError, ValueError, IndexError):
        return None


def verify(code: str, result, query: str, tables: dict) -> dict:
    signals = {}

    v = as_scalar(result)
    if v is not None:
        signals["v_type_ok"] = math.isfinite(v)
    else:
        # Not a number. For numeric table QA that is a failure, not an
        # unknown: an answer that is not a scalar cannot be the answer.
        signals["v_type_ok"] = False

    signals["v_schema_ok"] = _v_schema(code, tables)
    signals["v_range_ok"] = _v_range(v, tables, query)
    signals["v_cross_table"] = _v_cross_table(v, tables)
    signals["v_semantic_ok"] = _v_semantic(query, result, tables)

    return signals
