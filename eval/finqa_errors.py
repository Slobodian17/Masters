"""Where the accepted wrong answers come from, and what B2 could catch.

Reads cached runs only; no LLM call, no budget.

THE READ SET IS INFERRED, NOT TRACED. The sandbox does not yet record which
cells a program touched, so the set is reconstructed from the code's string
literals and column references. That inference is checked rather than assumed:
for every case the reported value is tested against simple functions of the
inferred cells, and the share where it reconciles is reported. A low
reconciliation rate would mean the inference is wrong and the classification
below it cannot be trusted.

B2 here is a SIMULATION of the invariant check on inputs, under this
definition:

    REFUTED when a value is reported and no cell was read      (silent zero)
    REFUTED when the read set holds both an invariant's target
            and its members                                    (double count)
    REFUTED when the value is not any simple function of the
            cells that were read                               (unsupported)
    UNDECIDABLE otherwise

The double-count clause is the one that is not obvious. In a single-table
gold setting every honest computation over real cells is automatically
consistent with invariants derived from those same cells, so an invariant
cannot refute it -- unless the program summed a subtotal together with the
components it already contains, which double-counts while touching nothing
but genuine values.

Run:  python -m eval.finqa_errors --tags ab_no_v2,ab_yes_v2
"""
import argparse
import ast
import collections
import itertools
import os
import re

import numpy as np
import pandas as pd

from core import invariants
from eval.finqa_replay import load_cache
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))

# Factors a program may legitimately introduce without reading them from the
# table: the percent convention and the "in thousands/millions" caption.
UNIT_CONSTANTS = (1.0, 2.0, 100.0, 1000.0, 1e6, 1e9)

TOL = 0.01

SCALE = "масштаб/одиниця"
SILENT_ZERO = "тихий нуль"
WRONG_ROW = "не той рядок"
WRONG_COL = "не той стовпець"
WRONG_OP = "не та операція"
PARTIAL = "часткова агрегація"
DOUBLE_COUNT = "подвійний рахунок"
OTHER = "інше"


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def final_code(rec: dict):
    for s in reversed(rec.get("steps") or []):
        if s.get("signals") is not None and s.get("code"):
            return s["code"]
    for s in reversed(rec.get("steps") or []):
        if s.get("code"):
            return s["code"]
    return None


def code_literals(code: str) -> list:
    """String literals in the program, in source order."""
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return re.findall(r"'([^']*)'|\"([^\"]*)\"", code or "")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return out


def code_numbers(code: str) -> list:
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            out.append(float(node.value))
    return out


_OP_NODES = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply",
             ast.Div: "divide"}
_OP_CALLS = {"sum": "add", "mean": "table_average", "max": "table_max",
             "min": "table_min"}


def code_ops(code: str) -> set:
    ops = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ops
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            name = _OP_NODES.get(type(node.op))
            if name:
                ops.add(name)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = _OP_CALLS.get(node.func.attr)
            if name:
                ops.add(name)
    return ops


def infer_read_set(code: str, df: pd.DataFrame) -> dict:
    """-> {"rows": [i], "cols": [name], "cells": [(i, col)], "matched_literals": n}

    A literal that equals a row label selects that row; a literal that equals a
    column name selects that column. When no column is named, every numeric
    column counts as read, because the program addressed the row as a whole.
    """
    if df is None or df.empty:
        return {"rows": [], "cols": [], "cells": [], "matched": 0, "literals": 0}

    labels = [_norm(v) for v in df.iloc[:, 0].tolist()]
    cols = [str(c) for c in df.columns[1:]]
    col_norm = {_norm(c): c for c in cols}

    lits = code_literals(code or "")
    rows, hit_cols, matched = [], [], 0
    for lit in lits:
        n = _norm(lit)
        if n in labels:
            rows.append(labels.index(n))
            matched += 1
        elif n in col_norm:
            hit_cols.append(col_norm[n])
            matched += 1

    rows = sorted(set(rows))
    hit_cols = list(dict.fromkeys(hit_cols)) or cols
    cells = [(r, c) for r in rows for c in hit_cols]
    return {"rows": rows, "cols": hit_cols, "cells": cells,
            "matched": matched, "literals": len(lits)}


def cell_values(df: pd.DataFrame, cells: list) -> list:
    out = []
    for r, c in cells:
        try:
            v = float(pd.to_numeric(df.iloc[r][c], errors="coerce"))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if np.isfinite(v):
            out.append(v)
    return out


def derivable(value, vals: list) -> bool:
    """True when `value` is a simple function of the read cells.

    Deliberately generous: the question is whether the value COULD have come
    from these cells, so a false 'yes' is safer than refuting honest work.
    """
    if value is None or not vals:
        return False
    v = float(value)

    def close(a):
        return abs(v - a) <= TOL * max(abs(a), 1.0)

    cands = list(vals) + [sum(vals), float(np.mean(vals))]
    if len(vals) <= 8:
        for k in range(2, min(len(vals), 4) + 1):
            for combo in itertools.combinations(vals, k):
                cands.append(sum(combo))
    for a, b in itertools.permutations(vals, 2) if len(vals) <= 12 else []:
        cands.append(a - b)
        if b:
            cands.append(a / b)
            cands.append((a - b) / b)
    for base in list(cands):
        for k in UNIT_CONSTANTS:
            cands.append(base * k)
            cands.append(base / k)
    return any(close(c) for c in cands if np.isfinite(c))


SUMMING_OPS = {"add", "table_average", "table_max", "table_min"}


def double_counted(read_rows: list, inv_list: list, ops: set) -> dict:
    """The program SUMMED an invariant's target together with its members.

    The summing requirement is not a detail. "What share of total assets is
    cash" legitimately reads both the component and the total and divides
    them; flagging that as double counting would refute correct work. Only an
    addition over both is an error.
    """
    if not (ops & SUMMING_OPS):
        return {}
    rows = set(read_rows)
    for inv in inv_list:
        if inv["kind"] != "row_sum":
            continue
        members = set(inv["member_indices"])
        if inv["target_index"] in rows and members & rows:
            return {"invariant": invariants.describe(inv),
                    "target": inv["target_index"],
                    "members_also_read": sorted(members & rows)}
    return {}


def classify(rec: dict, df: pd.DataFrame, inv_list: list) -> dict:
    code = final_code(rec) or ""
    read = infer_read_set(code, df)
    vals = cell_values(df, read["cells"])
    value = rec.get("candidate")
    gold_rows = set(rec.get("gold_rows") or [])

    ops = code_ops(code)
    ok_derivable = derivable(value, vals)
    dc = double_counted(read["rows"], inv_list, ops)

    # The inference failed when the program clearly selected rows -- it has
    # string literals -- but none matched a label. Selection by regex or by
    # positional index does that. Such a case says nothing about B2 and must
    # not be counted as a catch.
    inference_failed = bool(read["literals"]) and read["matched"] == 0

    if rec.get("candidate_scale_error"):
        kind = SCALE
    elif value is not None and abs(float(value)) < 1e-12 and not read["rows"]:
        kind = SILENT_ZERO
    elif dc:
        kind = DOUBLE_COUNT
    elif gold_rows and read["rows"] and not (set(read["rows"]) & gold_rows):
        kind = WRONG_ROW
    elif ops and rec.get("gold_ops") and not (ops & set(rec["gold_ops"])):
        kind = WRONG_OP
    elif gold_rows and set(read["rows"]) and set(read["rows"]) < gold_rows:
        kind = PARTIAL
    elif gold_rows and read["rows"] and (set(read["rows"]) & gold_rows):
        kind = WRONG_COL
    else:
        kind = OTHER

    # Simulated B2, deliberately conservative: whatever the inference cannot
    # vouch for counts as UNDECIDABLE, never as a catch.
    b2_reasons = []
    if not inference_failed:
        if (value is not None and abs(float(value)) > 1e-12
                and not read["rows"] and not read["literals"]):
            b2_reasons.append("значення повідомлено, клітинок не прочитано")
        if dc:
            b2_reasons.append("підсумок додано разом із компонентами")
        if vals and read["rows"] and not ok_derivable:
            b2_reasons.append("значення не виводиться з прочитаного")

    return {
        "id": rec["id"],
        "kind": kind,
        "candidate": value,
        "gold": rec.get("gold"),
        "n_read_rows": len(read["rows"]),
        "n_read_cells": len(read["cells"]),
        "literals_matched": f"{read['matched']}/{read['literals']}",
        "derivable": ok_derivable,
        "inference_failed": inference_failed,
        "b2_would_catch": bool(b2_reasons),
        "b2_reason": "; ".join(b2_reasons),
        "n_invariants": len(inv_list),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="ab_no_v2,ab_yes_v2")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    records = {r["id"]: r for r in fq.load_split(args.split)}

    print("=" * 78)
    print("ЗВІДКИ БЕРУТЬСЯ ПРИЙНЯТІ ХИБНІ ВІДПОВІДІ, І ЩО З НИХ ЗЛОВИВ БИ B2")
    print("=" * 78)
    print("Нуль звернень до LLM. Множина прочитаних клітинок ВИВЕДЕНА з коду,")
    print("а не трасована — дивись рівень звірки нижче.")
    print()

    per_tag = {}
    for tag in tags:
        rows = load_cache(tag)
        wrong = [r for r in rows
                 if r["status"] == "ACCEPT" and r["answer_correct"] is False]
        out = []
        for r in wrong:
            rec = records.get(r["id"])
            df = fq.to_tables(rec).get("table") if rec else None
            inv = invariants.detect({"t": df})["invariants"] if df is not None and not df.empty else []
            r = dict(r)
            gold_rows = set(rec["qa"].get("ann_table_rows") or []) if rec else set()
            if rec and fq.header_is_data(rec.get("table") or []):
                gold_rows = {g + 1 for g in gold_rows}
            r["gold_rows"] = sorted(gold_rows)
            out.append(classify(r, df, inv))
        per_tag[tag] = pd.DataFrame(out)
        print(f"  {tag}: прийнятих хибних {len(out)} з {len(rows)}")

    allrows = pd.concat(per_tag.values(), ignore_index=True)
    n = len(allrows)

    print()
    print("--- Рівень звірки виведеної множини читань ---")
    print(f"  значення виводиться з прочитаних клітинок: "
          f"{int(allrows.derivable.sum())}/{n} = {allrows.derivable.mean():.0%}")
    bad = int(allrows.inference_failed.sum())
    print(f"  виведення НЕ вдалось (вибірка регуляркою чи за індексом): "
          f"{bad}/{n} = {bad/n:.0%}")
    print("  Ці випадки виключені з симуляції B2: вони нічого про неї не кажуть.")

    print()
    print("--- Типи помилок ---")
    for tag, df in per_tag.items():
        c = collections.Counter(df.kind)
        total = len(df)
        print(f"\n  {tag} (n={total}):")
        for k, v in c.most_common():
            print(f"    {k:22s} {v:2d}  ({v/total:.0%})")

    print()
    print("--- ЧИ ЗЛОВИВ БИ B2 (симуляція, gold-режим) ---")
    for tag, df in per_tag.items():
        k = int(df.b2_would_catch.sum())
        print(f"  {tag}: {k}/{len(df)} = {k/len(df):.0%}")
    k = int(allrows.b2_would_catch.sum())
    print(f"  РАЗОМ: {k}/{n} = {k/n:.0%}")
    if k:
        print("\n  спіймані випадки:")
        for _, r in allrows[allrows.b2_would_catch].iterrows():
            print(f"    [{r['kind']}] {r['b2_reason']}  (ans={r['candidate']}, gold={r['gold']})")

    print()
    print("--- Інваріанти в таблицях цих питань ---")
    print(f"  середньо інваріантів: {allrows.n_invariants.mean():.2f}")
    print(f"  питань, де інваріантів нема взагалі: "
          f"{int((allrows.n_invariants == 0).sum())}/{n}")

    out = os.path.join(HERE, "finqa_error_analysis.csv")
    allrows.to_csv(out, index=False)
    print()
    print("Збережено:", out)
    return allrows


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
