"""Inductive detection of arithmetic invariants inside a table.

A financial statement is not a bag of independent numbers: reporting structure
imposes hard arithmetic ties between them — component rows sum to a subtotal,
quarters sum to a year, liabilities plus equity equal assets.

Those ties are recoverable from the numbers alone, with no accounting
knowledge and no annotation. Once recovered they act as a partial oracle: an
answer that contradicts one is not merely implausible, it is arithmetically
impossible.

This differs in kind from the probabilistic signals of the verification
cascade. Those lower a confidence score; an invariant violation refutes.

WHAT IT DOES NOT DO. An invariant refutes but never confirms. Satisfying one
is no proof of correctness — a value can be consistent with a sum and still
answer the wrong question. The result is a one-sided filter: "definitely no"
or "cannot tell".

FALSE INVARIANTS. On a small table coincidental matches are unavoidable, so a
candidate is accepted only when it holds for EVERY period column at once.
Holding for three columns out of eight is noise.

SEARCH SPACE. Enumerating every subset is 2**R and hopeless for a 28-row
statement. Two families cover the structures that actually occur in reports:

  * contiguous blocks — a subtotal under a group of components;
  * small combinations (k=2,3) — non-adjacent ties such as
    liabilities + equity = assets.
"""
import itertools

import numpy as np
import pandas as pd

# Tolerance for a match.
#
# MEASURED, not chosen: at a 0.5% relative tolerance the all-columns filter
# admitted 30 invariants of which only 12 were genuine. The reason is that
# rows in a statement are largely proportional to one another, so an arbitrary
# combination whose coefficients happen to add up stays within a loose band in
# EVERY column at once. Holding across all columns is therefore necessary but
# not sufficient; tolerance is the second, decisive guard.
#
# At 1e-4 the coincidences vanish and only genuine ties remain.
ABS_TOL = 0.5
REL_TOL = 1e-4

# Statements are printed rounded, so a tie between independently rounded
# values carries an error of up to half a rounding step per member. When
# `granularity` is known, tolerance follows from it instead of a fixed
# percentage: k members rounded to step g admit at most k * g / 2 of drift.
#
# Subtotals in most published statements are sums of the PRINTED values, so
# they match exactly and the tight setting applies. Independently rounded
# totals need the granularity rule. Which regime real documents follow is a
# question for the next step, not an assumption to bake in here.
USE_GRANULARITY_TOLERANCE = False

# Upper bound on non-contiguous combination size. Beyond three the search
# grows fast and the extra hits are mostly derived from smaller ones.
MAX_COMBINATION = 3

# A block must have at least two members: a one-row "sum" is a duplicate row,
# not a structural tie.
MIN_MEMBERS = 2


def _numeric_matrix(df: pd.DataFrame):
    """-> (matrix R x C, column names, row labels).

    The first column holds labels; every other column is coerced to float and
    dropped if it holds no numbers at all.
    """
    if df.shape[1] < 2:
        return np.zeros((0, 0)), [], []

    labels = [str(v) for v in df.iloc[:, 0].tolist()]
    cols, series = [], []
    for c in df.columns[1:]:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            cols.append(str(c))
            series.append(s.to_numpy(dtype=float))

    if not series:
        return np.zeros((0, 0)), [], labels
    return np.vstack(series).T, cols, labels


def detect_granularity(matrix: np.ndarray) -> float:
    """Smallest rounding step every value in the table is a multiple of.

    Returns 1.0 when values are not round numbers.
    """
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 1.0
    for step in (1000.0, 100.0, 10.0):
        residue = np.mod(finite, step)
        if np.allclose(residue, 0.0) or np.allclose(residue, step):
            return step
    return 1.0


def _tolerance(target_value: float, n_members: int = 1, granularity: float = 1.0) -> float:
    if USE_GRANULARITY_TOLERANCE:
        return max(ABS_TOL, n_members * granularity / 2.0)
    return max(ABS_TOL, REL_TOL * max(abs(target_value), 1.0))


def _holds_everywhere(member_sums: np.ndarray, target_row: np.ndarray,
                      n_members: int = 1, granularity: float = 1.0) -> tuple:
    """True when the sum matches the target in EVERY column.

    Returns (holds, worst_relative_error). NaN on either side fails the check:
    a missing value is not evidence of a tie.
    """
    if np.isnan(member_sums).any() or np.isnan(target_row).any():
        return False, float("inf")

    worst = 0.0
    for s, t in zip(member_sums, target_row):
        if abs(s - t) > _tolerance(t, n_members, granularity):
            return False, float("inf")
        worst = max(worst, abs(s - t) / max(abs(t), 1.0))
    return True, worst


def _is_degenerate(matrix: np.ndarray, members: tuple, target: int) -> bool:
    """Reject ties that hold only because the numbers involved are zero."""
    if np.allclose(matrix[target], 0.0):
        return True
    if np.allclose(matrix[list(members)], 0.0):
        return True
    return False


def find_row_sums(df: pd.DataFrame, table_name: str = "") -> dict:
    """Rows whose sum equals another row, in every column.

    Returns {"invariants": [...], "candidates_tested": n, "rejected": n}.
    """
    matrix, cols, labels = _numeric_matrix(df)
    if matrix.size == 0:
        return {"invariants": [], "candidates_tested": 0, "rejected": 0}

    granularity = detect_granularity(matrix)
    n_rows = matrix.shape[0]
    found, seen = [], set()
    tested = 0

    def _try(members: tuple, target: int, form: str):
        nonlocal tested
        key = (frozenset(members), target)
        if key in seen:
            return
        seen.add(key)
        tested += 1

        if _is_degenerate(matrix, members, target):
            return
        holds, err = _holds_everywhere(matrix[list(members)].sum(axis=0), matrix[target],
                                       len(members), granularity)
        if not holds:
            return
        found.append({
            "kind": "row_sum",
            "table": table_name,
            "form": form,
            "target_index": target,
            "target_label": labels[target],
            "member_indices": list(members),
            "member_labels": [labels[i] for i in members],
            "n_members": len(members),
            "n_columns": len(cols),
            "max_rel_error": err,
        })

    # contiguous blocks
    for start in range(n_rows):
        for end in range(start + MIN_MEMBERS, n_rows + 1):
            members = tuple(range(start, end))
            for target in range(n_rows):
                if target in members:
                    continue
                _try(members, target, "contiguous")

    # small non-contiguous combinations
    for k in range(MIN_MEMBERS, MAX_COMBINATION + 1):
        for members in itertools.combinations(range(n_rows), k):
            for target in range(n_rows):
                if target in members:
                    continue
                _try(members, target, "combination")

    return {"invariants": found, "candidates_tested": tested,
            "rejected": tested - len(found)}


def find_column_sums(df: pd.DataFrame, table_name: str = "") -> dict:
    """Columns whose sum equals another column, in every row.

    The quarters-to-year tie lives here. A table holding only quarters, with
    no aggregate column, yields nothing — which is a property of the document,
    not a failure of the search.
    """
    matrix, cols, labels = _numeric_matrix(df)
    if matrix.size == 0:
        return {"invariants": [], "candidates_tested": 0, "rejected": 0}

    granularity = detect_granularity(matrix)
    n_cols = matrix.shape[1]
    found, seen = [], set()
    tested = 0

    for k in range(MIN_MEMBERS, n_cols):
        for members in itertools.combinations(range(n_cols), k):
            for target in range(n_cols):
                if target in members:
                    continue
                key = (frozenset(members), target)
                if key in seen:
                    continue
                seen.add(key)
                tested += 1

                tcol = matrix[:, target]
                mcols = matrix[:, list(members)].sum(axis=1)
                if np.allclose(tcol, 0.0) or np.allclose(matrix[:, list(members)], 0.0):
                    continue
                holds, err = _holds_everywhere(mcols, tcol, len(members), granularity)
                if not holds:
                    continue
                found.append({
                    "kind": "col_sum",
                    "table": table_name,
                    "form": "combination",
                    "target_index": target,
                    "target_label": cols[target],
                    "member_indices": list(members),
                    "member_labels": [cols[i] for i in members],
                    "n_members": len(members),
                    "n_columns": matrix.shape[0],
                    "max_rel_error": err,
                })

    return {"invariants": found, "candidates_tested": tested,
            "rejected": tested - len(found)}


def detect(tables: dict) -> dict:
    """Run both families over every table.

    -> {"invariants": [...], "candidates_tested": n, "rejected": n,
        "per_table": {name: {...}}}
    """
    all_inv, per_table = [], {}
    total_tested = total_rejected = 0

    for name, df in tables.items():
        rows = find_row_sums(df, name)
        colsums = find_column_sums(df, name)

        inv = rows["invariants"] + colsums["invariants"]
        tested = rows["candidates_tested"] + colsums["candidates_tested"]
        rejected = rows["rejected"] + colsums["rejected"]

        per_table[name] = {
            "shape": tuple(df.shape),
            "n_invariants": len(inv),
            "n_row_sums": len(rows["invariants"]),
            "n_col_sums": len(colsums["invariants"]),
            "candidates_tested": tested,
            "rejected": rejected,
        }
        all_inv.extend(inv)
        total_tested += tested
        total_rejected += rejected

    return {"invariants": all_inv, "candidates_tested": total_tested,
            "rejected": total_rejected, "per_table": per_table}


def rows_under_constraint(invariants: list) -> dict:
    """-> {table: set of row labels participating in at least one invariant}."""
    out = {}
    for inv in invariants:
        if inv["kind"] != "row_sum":
            continue
        s = out.setdefault(inv["table"], set())
        s.add(inv["target_label"])
        s.update(inv["member_labels"])
    return out


def check_value(value: float, invariant: dict, df: pd.DataFrame, column: str) -> dict:
    """Test one answer against one invariant, for a given period column.

    Only meaningful when `value` is claimed to be the invariant's target cell.
    -> {"verdict": "REFUTED" | "CONSISTENT" | "UNDECIDABLE", ...}
    """
    matrix, cols, labels = _numeric_matrix(df)
    if matrix.size == 0 or column not in cols:
        return {"verdict": "UNDECIDABLE", "reason": "column not numeric"}

    c = cols.index(column)
    expected = float(matrix[invariant["member_indices"], c].sum())
    if np.isnan(expected):
        return {"verdict": "UNDECIDABLE", "reason": "missing member values"}

    if abs(value - expected) <= _tolerance(expected):
        return {"verdict": "CONSISTENT", "expected": expected, "given": float(value)}
    return {"verdict": "REFUTED", "expected": expected, "given": float(value),
            "delta": float(value) - expected}


def independent_count(invariants: list) -> dict:
    """How many of the found ties carry independent information.

    Each invariant is the linear equation sum(members) - target = 0, so the
    number of independent ones is the rank of the system. This matters because
    substituting one relation into another yields a third that is valid but
    adds nothing: revenue = cogs + gross and gross = opex + operating together
    imply revenue = cogs + opex + operating.

    -> {table: {"total": n, "independent": r, "derived": n - r}}
    """
    by_table = {}
    for inv in invariants:
        if inv["kind"] != "row_sum":
            continue
        by_table.setdefault(inv["table"], []).append(inv)

    out = {}
    for table, items in by_table.items():
        width = max(max(i["member_indices"] + [i["target_index"]]) for i in items) + 1
        rows = []
        for i in items:
            v = np.zeros(width)
            for m in i["member_indices"]:
                v[m] += 1.0
            v[i["target_index"]] -= 1.0
            rows.append(v)
        rank = int(np.linalg.matrix_rank(np.vstack(rows))) if rows else 0
        out[table] = {"total": len(items), "independent": rank,
                      "derived": len(items) - rank}
    return out


def describe(inv: dict) -> str:
    """One-line human-readable form of an invariant."""
    members = " + ".join(inv["member_labels"][:4])
    if inv["n_members"] > 4:
        members += f" + ... ({inv['n_members']} members)"
    return f"[{inv['table']}] {members} = {inv['target_label']}"
