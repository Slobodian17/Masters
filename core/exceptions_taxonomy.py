"""Exception taxonomy and the deterministic L0 repair, which uses no LLM."""
import difflib
import re

DETERMINISTIC_FIXABLE = {"KeyError", "IndexError"}
LLM_REQUIRED = {"TypeError", "ValueError", "SyntaxError", "AttributeError"}

# Not an exception the code raised: the reply was cut at the token limit and
# what ran was half a program. It is separated because the repair strategy
# differs in kind -- regenerating the same prompt under the same limit cannot
# succeed, however many attempts it is given. Measured on FinQA: 112 of 116
# failures were this, each burning three repair iterations to no effect.
#
# This is the clearest case for typed escalation (H4): one strategy for every
# error class is not merely suboptimal here, it is structurally incapable.
TRUNCATION = "TRUNCATION"
BUDGET_FIXABLE = {TRUNCATION}


def classify_exception(e: Exception) -> str:
    return type(e).__name__


def _row_label_candidates(tables: dict) -> list[str]:
    """Text values from non-numeric columns, used as fuzzy-match candidates."""
    candidates = []
    for df in tables.values():
        for col in df.select_dtypes(include="object").columns:
            candidates.extend(df[col].astype(str).unique().tolist())
    return list(dict.fromkeys(candidates))  # unique, order preserved


def deterministic_repair(code: str, e: Exception, tables: dict):
    """Attempt a repair without calling the LLM.

    Returns patched code, or None when no deterministic fix applies — the
    escalation policy then raises the level to L1/L2.
    """
    err_type = classify_exception(e)

    if err_type == "KeyError":
        bad_key = e.args[0] if getattr(e, "args", None) else None
        if not isinstance(bad_key, str):
            return None
        all_columns = [c for df in tables.values() for c in df.columns]
        match = difflib.get_close_matches(bad_key, all_columns, n=1, cutoff=0.4)
        if not match:
            return None
        return code.replace(f"'{bad_key}'", f"'{match[0]}'").replace(f'"{bad_key}"', f'"{match[0]}"')

    if err_type == "IndexError":
        # Typical case: `.values[0]` on an empty filter result, because a
        # string literal in the code matches no real row label.
        candidates = _row_label_candidates(tables)
        literals = [a or b for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"", code)]
        for lit in literals:
            if lit in candidates:
                continue  # already valid
            match = difflib.get_close_matches(lit, candidates, n=1, cutoff=0.5)
            if match:
                return code.replace(f"'{lit}'", f"'{match[0]}'").replace(f'"{lit}"', f'"{match[0]}"')
        return None

    return None
