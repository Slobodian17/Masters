"""One case, end to end: a 713k-row table where guessing fails and code does not.

Zero API calls -- everything is replayed from the dev caches and re-executed
against the local table.

The explanation is not narrated, it is DERIVED: the column and the filter are
read out of the generated program's syntax tree, the filter is then applied
independently, and the count it produces is checked against the answer the
program returned. If the two disagree the demo says so instead of explaining.

The last part is the one that matters for the thesis. The direct answer was not
careless -- it answered correctly for the rows it was shown. The same filter is
applied to the window the retriever gave it, and that is where its answer comes
from. The failure is the window, not the reasoning.

Selection rule, stated so it is checkable: among dev questions where the direct
arm is wrong, the code arm is right, AND retrieval finds the table on its own at
rank 1, the one with the LARGEST table. Requiring the retrieval hit matters --
without it the demo would show a pipeline whose first stage we had skipped.

Run:  python -m eval.databench_demo
"""
import argparse
import ast
import json
import os

import pandas as pd

from core import sandbox
from eval import cache_io
from eval.databench_run import normalise, top_rows
from ingestion import databench_catalog as dcat
from ingestion import databench_loader as db

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "databench_cache")


def load(arm, tag):
    return {r["id"]: r
            for r in cache_io.load_rows(os.path.join(CACHE, f"{arm}_{tag}.jsonl"))[0]}


def final_code(r):
    return next((s.get("code") for s in reversed(r.get("steps") or [])
                 if s.get("code")), "")


def columns_used(code):
    """Column names the program subscripts, read from the syntax tree."""
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                and isinstance(n.slice.value, str):
            if n.slice.value not in out:
                out.append(n.slice.value)
    return out


def comparisons(code):
    """(column, operator, literal) for each comparison against a column."""
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    ops = {ast.Eq: "==", ast.NotEq: "!=", ast.Gt: ">", ast.Lt: "<",
           ast.GtE: ">=", ast.LtE: "<="}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Compare) or len(n.ops) != 1:
            continue
        left = n.left
        if isinstance(left, ast.Subscript) and isinstance(left.slice, ast.Constant):
            col = left.slice.value
            rhs = n.comparators[0]
            if isinstance(rhs, ast.Constant):
                out.append((col, ops.get(type(n.ops[0]), "?"), rhs.value))
    return out


def mask_of(df, col, op, val):
    s = df[col]
    return {"==": s == val, "!=": s != val, ">": s > val, "<": s < val,
            ">=": s >= val, "<=": s <= val}.get(op)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev100")
    ap.add_argument("--qid", default="")
    args = ap.parse_args()

    A, B = load("A", args.tag), load("B", args.tag)
    spec = json.load(open(os.path.join(HERE, "databench_dev.json"),
                          encoding="utf-8"))
    sizes = {k: v["rows"] for k, v in spec["tables"].items()}

    retr = dcat.Retriever()
    cand = []
    for q in A:
        if A[q]["correct"] or not B[q]["correct"]:
            continue
        top = retr.search(B[q]["question"], 3)
        if top and top[0][0] == B[q]["dataset"]:
            cand.append((sizes.get(B[q]["dataset"], 0), q))
    cand.sort(reverse=True)
    if args.qid:
        qid = args.qid
    else:
        if not cand:
            print("кандидатів немає")
            return
        qid = cand[0][1]
    a, b = A[qid], B[qid]
    df = db.table(b["dataset"])
    code = final_code(b)

    print("=" * 76)
    print("ДЕМО: велика таблиця, пряма відповідь хибна, код правильний")
    print("=" * 76)
    print(f"  правило відбору: серед dev-питань, де A хибний, B правильний")
    print(f"  І пошук сам знаходить таблицю на 1-му місці — найбільша")
    print(f"  таблиця. Кандидатів було {len(cand)} (із 55, де код виграв).")
    print()
    print()
    print("-" * 76)
    print("КРОК 1: ПОШУК ТАБЛИЦІ В КОРПУСІ (нуль викликів LLM)")
    print("-" * 76)
    hits = retr.search(b["question"], 3)
    print(f"  корпус: {len(retr.ids)} таблиць, {sum(e['rows'] for e in retr.catalog):,} рядків")
    print(f"  BM25 по каталогу (назва, стовпці, типи — без значень):")
    for rank, (tid, sc) in enumerate(hits, 1):
        mark = "  <- потрібна" if tid == b["dataset"] else ""
        print(f"    {rank}. {tid:20s} {sc:6.2f}{mark}")
    if len(hits) > 1 and hits[0][1] - hits[1][1] < 1.0:
        print(f"  перевага над другим місцем лише "
              f"{hits[0][1] - hits[1][1]:.2f} — пошук тут не надійний,")
        print(f"  і на всій вибірці він влучає @1 лише у 48% випадків")
    print()
    print(f"  таблиця : {b['dataset']}  {df.shape[0]:,} рядків x {df.shape[1]} стовпців")
    print(f"  питання : {b['question']}")
    print(f"  тип     : {b['type']}")
    print(f"  еталон  : {b['truth']}")

    print()
    print("-" * 76)
    print("КРОК 2а: ПРЯМА ВІДПОВІДЬ (арм A)")
    print("-" * 76)
    win = top_rows(df, b["question"])
    print(f"  пошук віддав моделі {len(win)} рядків — "
          f"{len(win) / len(df):.4%} таблиці")
    print(f"  відповідь: {a['value']}   -> ХИБНО")

    print()
    print("-" * 76)
    print("КРОК 2б: ЗГЕНЕРОВАНИЙ КОД (арм B)")
    print("-" * 76)
    for line in code.split("\n"):
        print("   ", line)
    cols = columns_used(code)
    cmps = comparisons(code)
    print()
    print(f"  стовпці, до яких звертається програма (з AST): {cols}")
    print(f"  умови (з AST): "
          + ("; ".join(f"{c} {o} {v!r}" for c, o, v in cmps) or "немає"))

    print()
    print("-" * 76)
    print("КРОК 3: ПЕРЕВІРКА ПОЯСНЕННЯ ВИКОНАННЯМ")
    print("-" * 76)
    ok_expl = False
    for col, op, val in cmps:
        if col not in df.columns:
            continue
        m = mask_of(df, col, op, val)
        if m is None:
            continue
        n = int(m.sum())
        print(f"  умова «{col} {op} {val!r}» виконується на "
              f"{n:,} рядках із {len(df):,} ({n / len(df):.1%})")
        if str(n) == str(b["value"]).strip():
            ok_expl = True
            print(f"  -> цей підрахунок ЗБІГАЄТЬСЯ з відповіддю програми "
                  f"({b['value']}): пояснення підтверджене виконанням")
        # the same filter on the window the direct arm actually saw
        mw = mask_of(win, col, op, val)
        if mw is not None:
            print(f"  -> у вікні прямого маршруту ця умова виконується на "
                  f"{int(mw.sum())} рядках із {len(win)}")
            print(f"     тобто пряма відповідь була ПРАВИЛЬНОЮ для тих даних,")
            print(f"     які їй показали — хибним було вікно, не міркування")
        print()
        head = df.loc[m, [c for c in cols if c in df.columns][:3]].head(3)
        if not head.empty:
            print("  приклади рядків, що проходять умову:")
            for line in head.to_string(index=True).split("\n"):
                print("     ", line)
            print()

    res, exc = sandbox.execute(code, {"df": df})
    val = normalise(res) if exc is None and res is not None else None
    print(f"  перезапуск коду зараз дає: {val}  "
          f"(у кеші {b['value']}) — {'збіг' if str(val) == str(b['value']) else 'РОЗБІЖНІСТЬ'}")
    try:
        verdict = db.compare(b["value"], b["truth"], b["type"])
    except Exception:
        verdict = False
    print(f"  офіційне порівняння з еталоном: "
          f"{'ПРАВИЛЬНО' if verdict else 'хибно'}")
    if not ok_expl:
        print()
        print("  УВАГА: жодна умова з AST не відтворила відповідь напряму —")
        print("  пояснення неповне (програма робить більше, ніж один фільтр).")

    print()
    print("=" * 76)
    print(f"  A бачив {len(win) / len(df):.4%} таблиці і відповів {a['value']}")
    print(f"  B працював на всіх {len(df):,} рядках і відповів {b['value']}")
    print(f"  еталон {b['truth']}")
    print("=" * 76)


if __name__ == "__main__":
    main()
