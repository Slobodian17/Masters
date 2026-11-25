"""Verification by EXECUTION, not by judgement. Zero API calls.

The agreement check of 5.26 asks a second model what it thinks. That check
inherits the generator's misreading: on 53 of 300 questions both routes agreed
on the same wrong number. This module asks a different kind of question, one
no model can be wrong about: *which cells does the answer actually depend on?*

The answer is measured, not reported. Take the cached program, perturb one
cell of the table, run it again, and see whether the answer moves. Do that for
every cell and the dependency set falls out of the executions. Nothing is
asked of an LLM, so nothing can be inherited from it.

Six signals, all oracle-free -- none of them looks at the gold answer:

  DEP_NONE     no cell changes the answer: the number was not computed
  DEP_SPREAD   the answer depends on more cells than the code aggregates
  ROW_PERM     shuffling row order changes the answer: positional indexing
  COL_PERM     shuffling data columns changes the answer: positional indexing
  ROW_CANARY   an added row the question never mentions changes the answer
  COL_CANARY   an added column the question never mentions changes the answer
  ROW_DROP     dropping a row the answer does NOT depend on changes it anyway
  SIGN         negative answer from non-negative cells with no subtraction
  CONST_LEAK   the program used a number that is in no table

The gold program is used ONLY afterwards, to ask whether the measured
dependency set points at the wrong cell. It never feeds a signal.

Run:  python -m eval.finqa_perturb --tag confirm300
"""
import argparse
import ast
import collections
import math
import os
import re

import numpy as np
import pandas as pd

from core import pipeline, sandbox
from eval import cache_io
from eval.finqa_confirm import build
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")

# Relative tolerance for "the answer moved". Loose enough to ignore float
# noise, tight enough that a one-cell contribution to a mean is still seen.
RTOL = 1e-9
MAX_CELLS = 120
CANARY_LABEL = "zzz unrelated placeholder item"

AGG_NAMES = {"sum", "mean", "max", "min", "count", "median", "std", "var",
             "prod", "agg", "aggregate", "nlargest", "nsmallest", "idxmax",
             "idxmin", "cumsum", "any", "all", "nunique", "quantile"}


def as_float(v):
    """Scalar float, or None when the program returned something else."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        if len(v) != 1:
            return None
        v = list(v)[0]
    if isinstance(v, pd.DataFrame):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def moved(a, b):
    """Did the answer change? `b is None` counts as movement (it crashed)."""
    if b is None:
        return True
    return abs(a - b) > RTOL * max(abs(a), 1.0)


def final_code(row):
    for s in reversed(row.get("steps") or []):
        if s.get("code"):
            return s["code"]
    return ""


def code_facts(code):
    """What the program itself says about its shape. AST only, no execution."""
    has_agg = has_sub = False
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return {"has_agg": True, "has_sub": True, "parse_ok": False}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fn in AGG_NAMES:
                has_agg = True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            has_sub = True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            has_sub = True
    return {"has_agg": has_agg, "has_sub": has_sub, "parse_ok": True}


# ---- the perturbations ----

def numeric_cells(df, cap=MAX_CELLS):
    """(row index, column name, value) for every parsed numeric cell."""
    out = []
    for col in df.columns[1:]:
        s = pd.to_numeric(df[col], errors="coerce")
        for i, v in s.items():
            if pd.notna(v):
                out.append((i, col, float(v)))
    return out[:cap]


def bump(df, i, col):
    """One cell moved to a value it could not coincidentally equal."""
    d = df.copy(deep=True)
    v = float(pd.to_numeric(d.at[i, col], errors="coerce"))
    d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)
    d.at[i, col] = v * 1.37 + 1.7
    return d


def with_row(df):
    """A row the question never mentions, valued at each column's median.

    The median keeps max() and min() unchanged, so a program that legitimately
    takes an extreme is not flagged; a program that sums or averages the whole
    column is.
    """
    d = df.copy(deep=True)
    new = {df.columns[0]: CANARY_LABEL}
    for col in df.columns[1:]:
        s = pd.to_numeric(d[col], errors="coerce").dropna()
        new[col] = float(s.median()) if len(s) else 0.0
    return pd.concat([d, pd.DataFrame([new])], ignore_index=True)


def with_col(df):
    d = df.copy(deep=True)
    src = pd.to_numeric(d[df.columns[1]], errors="coerce") if len(df.columns) > 1 else None
    d["zzz unrelated placeholder column"] = (
        src.fillna(0.0).astype(float) if src is not None else 0.0)
    return d


def perm_rows(df, seed=7):
    d = df.copy(deep=True)
    idx = list(d.index)
    np.random.RandomState(seed).shuffle(idx)
    return d.loc[idx].reset_index(drop=True)


def perm_cols(df, seed=7):
    d = df.copy(deep=True)
    data = list(d.columns[1:])
    np.random.RandomState(seed).shuffle(data)
    return d[[d.columns[0]] + data]


def drop_row(df, i):
    return df.drop(index=i).reset_index(drop=True)


# ---- one question ----

def run(code, df, name):
    return as_float(sandbox.execute(code, {name: df})[0])


def examine(code, tables):
    """Measure the dependency set and fire the oracle-free signals."""
    name = next(iter(tables))
    df = tables[name]
    base = run(code, df, name)
    out = {"base": base, "deps": [], "signals": [], "n_exec": 1,
           "dep_rows": 0, "parse_ok": True}
    if base is None:
        out["signals"].append("NO_VALUE")
        return out

    cells = numeric_cells(df)
    for i, col, v in cells:
        out["n_exec"] += 1
        if moved(base, run(code, bump(df, i, col), name)):
            out["deps"].append({"row": i, "col": col, "value": v,
                                "label": str(df.iat[i, 0])})
    dep_rows = sorted({d["row"] for d in out["deps"]})
    out["dep_rows"] = len(dep_rows)

    facts = code_facts(code)
    out["parse_ok"] = facts["parse_ok"]

    if not out["deps"]:
        out["signals"].append("DEP_NONE")
    elif not facts["has_agg"] and len(out["deps"]) > 3:
        out["signals"].append("DEP_SPREAD")

    checks = [("ROW_PERM", perm_rows(df)), ("COL_PERM", perm_cols(df)),
              ("ROW_CANARY", with_row(df)), ("COL_CANARY", with_col(df))]
    for tag, d2 in checks:
        out["n_exec"] += 1
        if moved(base, run(code, d2, name)):
            out["signals"].append(tag)

    # Dropping a row the answer does not depend on must leave it alone.
    for i in list(df.index):
        if i in dep_rows:
            continue
        out["n_exec"] += 1
        if moved(base, run(code, drop_row(df, i), name)):
            out["signals"].append("ROW_DROP")
            break

    if base < 0 and not facts["has_sub"] and all(d["value"] >= 0 for d in out["deps"]):
        out["signals"].append("SIGN")
    if pipeline.foreign_literals(code, tables):
        out["signals"].append("CONST_LEAK")
    return out


# ---- localisation, gold used here and nowhere else ----

_NUM = re.compile(r"-?\d+\.?\d*")
_REF = re.compile(r"#\d+")
_CONST = re.compile(r"const_\w+")


def gold_operands(prog):
    # `#0`, `#1` are references to earlier steps, not operands. Left in, the
    # regex reads them as the numbers 0 and 1 and invents operands the gold
    # program never used (defect 21).
    out = []
    for m in _NUM.findall(_REF.sub("", str(prog or ""))):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def cell_operands(prog):
    """Gold operands that must come from a CELL.

    `const_100`, `const_1000000` are scale and percent factors the program
    supplies itself, not values read from the table. Counting them as operands
    marks a program that located its cells correctly and then multiplied by 100
    as having read the wrong cells (defect 22).
    """
    return sorted(set(gold_operands(_CONST.sub("", str(prog or "")))))


def localise(deps, prog):
    """Does the measured dependency set contain the gold operands?

    HIT_ALL   every gold operand is among the cells the answer depends on
    HIT_SOME  some are, some are not -- the extra cells name the wrong ones
    HIT_NONE  none are: the program read the wrong part of the table entirely
    """
    g = cell_operands(prog)
    if not g:
        return "NO_GOLD_OPS", 0, 0
    dep = [d["value"] for d in deps]

    def present(x):
        return any(abs(x - y) <= 1e-4 * max(abs(x), 1.0) for y in dep)

    hit = sum(1 for x in g if present(x))
    extra = sum(1 for y in dep
                if not any(abs(y - x) <= 1e-4 * max(abs(x), 1.0) for x in g))
    if hit == len(g):
        return "HIT_ALL", hit, extra
    return ("HIT_SOME" if hit else "HIT_NONE"), hit, extra


# ---- report ----

ORDER = ["NO_VALUE", "DEP_NONE", "DEP_SPREAD", "ROW_PERM", "COL_PERM",
         "ROW_CANARY", "COL_CANARY", "ROW_DROP", "SIGN", "CONST_LEAK"]

NAMES = {
    "NO_VALUE": "код не дав числа",
    "DEP_NONE": "відповідь не залежить ні від якої клітинки",
    "DEP_SPREAD": "залежить від >3 клітинок без агрегації",
    "ROW_PERM": "перестановка рядків змінює відповідь",
    "COL_PERM": "перестановка стовпців змінює відповідь",
    "ROW_CANARY": "доданий сторонній рядок змінює відповідь",
    "COL_CANARY": "доданий сторонній стовпець змінює відповідь",
    "ROW_DROP": "видалення незалежного рядка змінює відповідь",
    "SIGN": "мінус із невідʼємних клітинок без відніманння",
    "CONST_LEAK": "у коді число, якого немає в таблиці",
}


def dep_txt(deps):
    return " | ".join(f"{d['label']}/{d['col']}={d['value']:g}"
                      for d in deps[:8]).replace(",", ";").replace('"', "'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    code_rows = cache_io.load_rows(os.path.join(CACHE, f"{args.tag}.jsonl"))[0]
    dir_rows = cache_io.load_rows(
        os.path.join(CACHE, f"baseline_table_{args.tag}.jsonl"))[0]
    rows = build(code_rows, dir_rows)
    by_id = {r["id"]: r for r in code_rows}
    recs = {r["id"]: r for r in fq.load_split("test")}
    if args.limit:
        rows = rows[:args.limit]

    print("=" * 78)
    print(f"ВЕРИФІКАЦІЯ ВИКОНАННЯМ — {args.tag}, нуль викликів API")
    print("=" * 78)

    res, repro_ok, repro_bad, no_code, execs = {}, 0, 0, 0, 0
    for r in rows:
        qid = r["id"]
        x = by_id[qid]
        code = final_code(x)
        tables = fq.to_tables(recs.get(qid, {}))
        if not code or not tables:
            no_code += 1
            continue
        e = examine(code, tables)
        execs += e["n_exec"]
        cached = as_float(r["r2"])
        e["repro"] = (cached is not None and e["base"] is not None
                      and not moved(cached, e["base"]))
        if e["repro"]:
            repro_ok += 1
        elif cached is not None:
            repro_bad += 1
        e["correct"] = r["correct"]
        e["gold"] = r["gold"]
        e["cached"] = cached
        e["loc"] = localise(e["deps"], x.get("gold_program"))
        e["prog"] = x.get("gold_program")
        e["question"] = x.get("question")
        res[qid] = e

    print()
    print("--- ВОРОТА ЧЕСНОСТІ: чи відтворюється кеш перезапуском ---")
    print(f"  питань у кеші              : {len(rows)}")
    print(f"  без коду (гейт/порожньо)   : {no_code}")
    print(f"  перезапуск дав те саме     : {repro_ok}")
    print(f"  перезапуск дав ІНШЕ        : {repro_bad}")
    print(f"  усього виконань програм    : {execs}")
    if repro_ok + repro_bad:
        print(f"  відтворюваність            : {repro_ok / (repro_ok + repro_bad):.1%}")
    if repro_bad > repro_ok * 0.1:
        print("  !! відтворюваність низька — усе нижче недостовірне")

    usable = {k: v for k, v in res.items() if v["repro"]}
    wrong = {k: v for k, v in usable.items() if not v["correct"]}
    right = {k: v for k, v in usable.items() if v["correct"]}
    print()
    print(f"  придатних для аналізу: {len(usable)}  "
          f"(хибних {len(wrong)}, правильних {len(right)})")

    print()
    print("--- ВИЯВЛЕННЯ, кожен сигнал окремо (еталон не використано) ---")
    print(f"  {'сигнал':46s} {'хибних':>7s} {'правил.':>8s} {'precis.':>8s}")
    for tag in ORDER:
        w = sum(1 for v in wrong.values() if tag in v["signals"])
        c = sum(1 for v in right.values() if tag in v["signals"])
        if not (w or c):
            continue
        p = w / (w + c) if (w + c) else float("nan")
        print(f"  {NAMES[tag]:46s} {w:7d} {c:8d} {p:7.1%}")

    def combo(tags, title):
        w = sum(1 for v in wrong.values() if set(tags) & set(v["signals"]))
        c = sum(1 for v in right.values() if set(tags) & set(v["signals"]))
        rec = w / max(len(wrong), 1)
        pre = w / (w + c) if (w + c) else float("nan")
        print(f"  {title:46s} {w:7d} {c:8d} {pre:7.1%}  recall {rec:5.1%}")

    print()
    print("--- ВИЯВЛЕННЯ, набори ---")
    print(f"  {'набір':46s} {'хибних':>7s} {'правил.':>8s} {'precis.':>8s}")
    combo(ORDER, "будь-який сигнал")
    combo([t for t in ORDER if t != "CONST_LEAK"], "без CONST_LEAK")
    combo(["NO_VALUE", "DEP_NONE", "ROW_PERM", "COL_PERM"], "лише структурні")
    combo(["ROW_CANARY", "COL_CANARY", "ROW_DROP"], "лише канарки")

    print()
    print("--- ЧИ ВКАЗУЄ ВИМІРЯНА МНОЖИНА НА НЕ ТУ КЛІТИНКУ (тут еталон) ---")
    print(f"  {'':22s} {'хибні':>7s} {'правильні':>10s}")
    for key in ("HIT_ALL", "HIT_SOME", "HIT_NONE", "NO_GOLD_OPS"):
        w = sum(1 for v in wrong.values() if v["loc"][0] == key)
        c = sum(1 for v in right.values() if v["loc"][0] == key)
        print(f"  {key:22s} {w:7d} {c:10d}")
    act = [v for v in wrong.values()
           if v["loc"][0] in ("HIT_SOME", "HIT_NONE") and v["deps"]]
    print(f"  хибних із придатною для ремонту адресою: {len(act)}"
          f"  (є і зайві клітинки, і відсутні gold-операнди)")
    if wrong:
        md = np.median([len(v["deps"]) for v in wrong.values()]) if wrong else 0
        mr = np.median([len(v["deps"]) for v in right.values()]) if right else 0
        print(f"  медіана |залежних клітинок|: хибні {md:.0f}, правильні {mr:.0f}")

    print()
    print("--- ЩО САМЕ ОТРИМАЄ ОДНА LLM-ПЕРЕВІРКА (приклади з хибних) ---")
    print("    Множина нижче ВИМІРЯНА виконанням, не взята зі слів моделі.")
    shown = 0
    for qid, v in res.items():
        if v.get("correct") or not v["repro"] or shown >= 4 or not v["deps"]:
            continue
        shown += 1
        print()
        print(f"  [{qid}]  {str(v['question'])[:84]}")
        print(f"    відповідь коду: {v['base']:.6g}")
        print("    залежить від клітинок:")
        for d in v["deps"][:6]:
            print(f"      {str(d['label'])[:46]:46s} | {str(d['col'])[:22]:22s}"
                  f" | {d['value']:g}")
        print(f"    (еталон {v['gold']}, gold-операнди "
              f"{sorted(set(gold_operands(v['prog'])))[:6]}, вердикт {v['loc'][0]})")

    out = os.path.join(HERE, f"finqa_perturb_{args.tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,correct,repro,n_deps,dep_rows,signals,loc,gold_hit,"
                "dep_extra,n_exec,gold,ours,gold_program\n")
        for qid, v in res.items():
            gp = str(v.get("prog")).replace(",", ";")
            f.write(f"{qid},{int(bool(v.get('correct')))},{int(v['repro'])},"
                    f"{len(v['deps'])},{v['dep_rows']},"
                    f"\"{'|'.join(v['signals'])}\",{v['loc'][0]},{v['loc'][1]},"
                    f"{v['loc'][2]},{v['n_exec']},{v['gold']},{v['cached']},"
                    f"{gp}\n")
    print()
    print("Порядкові дані:", out)

    print()
    print("--- ЦІНА ОДНОЇ LLM-ПЕРЕВІРКИ НА ПОЗНАЧЕНЕ ПИТАННЯ ---")
    flagged_w = sum(1 for v in wrong.values() if v["signals"])
    flagged_c = sum(1 for v in right.values() if v["signals"])
    flagged = flagged_w + flagged_c
    print(f"  позначено всього           : {flagged} з {len(usable)}"
          f"  ({flagged / max(len(usable), 1):.0%})")
    print(f"  виконань програм витрачено : {execs}  (нуль викликів API)")
    for tok in (900, 1200):
        print(f"  {flagged} перевірок x {tok} ток = {flagged * tok:,} ток"
              f" = {flagged * tok / 200000:.2f} доби Groq"
              f" / {flagged / 500:.2f} доби flash-lite")


if __name__ == "__main__":
    main()
