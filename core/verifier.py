"""Verification cascade: four surrogate signals forming the vector v.

v_schema, v_range, v_cross_table and v_semantic replace the test oracle that
does not exist for open numeric queries. The vector feeds `calibration`.
"""
import math
import re

import numpy as np

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


def _v_semantic(query: str, v, tables: dict):
    """Single yes/no plausibility question to the LLM, not a full check."""
    prompt = (
        f"Запит користувача: {query}\n"
        f"Отримана числова відповідь: {v}\n"
        "Чи ця відповідь виглядає правдоподібно як результат саме такого запиту "
        "(без урахування точності — лише порядок величини й здоровий глузд)? "
        "Відповідай рівно одним словом: так або ні."
    )
    try:
        resp = call_llm(prompt).strip().lower()
    except Exception:
        return None
    if "так" in resp:
        return True
    if "ні" in resp:
        return False
    return None


def verify(code: str, result, query: str, tables: dict) -> dict:
    signals = {}

    try:
        v = float(result)
        signals["v_type_ok"] = math.isfinite(v)
    except (TypeError, ValueError):
        v = None
        signals["v_type_ok"] = result is not None and str(result).strip() != ""

    signals["v_schema_ok"] = _v_schema(code, tables)
    signals["v_range_ok"] = _v_range(v, tables, query)
    signals["v_cross_table"] = _v_cross_table(v, tables)
    signals["v_semantic_ok"] = _v_semantic(query, result, tables)

    return signals
