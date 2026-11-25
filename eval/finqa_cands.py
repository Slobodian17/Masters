"""Candidate construction, tuned with zero API calls.

The judge can only choose cells it is shown. Whether the cells a question needs
are in the candidate list at all is a deterministic comparison against the gold
operands -- no model is involved -- so the retrieval half of the method can be
fixed for free, and must be fixed before any more quota goes into the judge.

The first builder (5.38) seeded the list from the cells the program read, and
on the questions where the program read the WRONG cells the right ones were
present only 42% of the time: neighbours of a wrong cell are wrong too. Seeding
has to come from the question as well as from the program.

Variants are scored on two numbers that pull against each other:
  recall  the gold operands are all present among the candidates
  size    how many candidates the judge then has to choose between

Run:  python -m eval.finqa_cands --tag confirm300
"""
import argparse
import os
import re

import pandas as pd

from eval import cache_io
from eval.finqa_confirm import build
from eval.finqa_judge import covers
from eval.finqa_perturb import cell_operands, examine, final_code
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")

STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
        "is", "was", "were", "what", "how", "much", "many", "that", "this",
        "it", "its", "by", "as", "with", "from", "be", "are", "total", "value",
        "amount", "percent", "percentage", "change", "ratio", "difference"}
YEAR = re.compile(r"\b(19|20)\d{2}\b")


def norm(s):
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower()).split()
            if t not in STOP]


def numeric_cells(df):
    out = []
    for col in df.columns[1:]:
        s = pd.to_numeric(df[col], errors="coerce")
        for i, v in s.items():
            if pd.notna(v):
                out.append({"row": i, "col": str(col), "value": float(v),
                            "label": str(df.iat[i, 0])})
    return out


def overlap(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb)


def build_cands(df, deps, question, cfg):
    """One parameterised builder, so the variants differ only in numbers."""
    cells = numeric_cells(df)
    by_key = {(c["row"], c["col"]): c for c in cells}
    picked, order = {}, []

    def add(c):
        k = (c["row"], c["col"])
        if k not in picked:
            picked[k] = c
            order.append(k)

    qt = norm(question)
    qyears = set(YEAR.findall(str(question)))
    qyears = set(m.group(0) for m in YEAR.finditer(str(question)))

    # 1. what the program read
    seeds = [by_key[(d["row"], str(d["col"]))] for d in deps
             if (d["row"], str(d["col"])) in by_key]
    for c in seeds[:cfg["seed"]]:
        add(c)

    # 2. same row, other periods
    for c in seeds[:cfg["seed_row"]]:
        for o in cells:
            if o["row"] == c["row"] and o["col"] != c["col"]:
                add(o)

    # 3. rows with a similar label
    for c in seeds[:cfg["seed_twin"]]:
        want = norm(c["label"])
        sib = sorted((o for o in cells if o["row"] != c["row"]
                      and o["col"] == c["col"]),
                     key=lambda o: -overlap(want, norm(o["label"])))
        for o in sib[:cfg["twin"]]:
            add(o)

    # 4. rows the QUESTION names -- independent of what the program did
    if cfg["q_rows"]:
        rows_by_score = {}
        for c in cells:
            rows_by_score.setdefault(c["row"], overlap(qt, norm(c["label"])))
        best = sorted((r for r, s in rows_by_score.items() if s > 0),
                      key=lambda r: -rows_by_score[r])[:cfg["q_rows"]]
        for r in best:
            picks = [c for c in cells if c["row"] == r]
            # whole row when the question names a period, else the matched cols
            if cfg["q_row_full"]:
                for o in picks:
                    add(o)
            else:
                for o in picks:
                    if not qyears or any(y in o["col"] for y in qyears):
                        add(o)

    # 5. columns the QUESTION names (years and periods)
    if cfg["q_cols"] and qyears:
        for c in cells:
            if any(y in c["col"] for y in qyears):
                add(c)

    # 6. plain lexical top-k over label plus column
    lex = sorted(cells, key=lambda o: -(overlap(qt, norm(o["label"]))
                                        + overlap(qt, norm(o["col"]))))
    for o in lex[:cfg["lex"]]:
        add(o)

    out = [picked[k] for k in order][:cfg["cap"]]
    return sorted(out, key=lambda c: (c["row"], c["col"]))


V0 = {"name": "V0 як зараз", "seed": 6, "seed_row": 3, "seed_twin": 2,
      "twin": 3, "q_rows": 0, "q_row_full": False, "q_cols": False,
      "lex": 4, "cap": 14}
V1 = {**V0, "name": "V1 більше лексики, cap 20", "lex": 8, "cap": 20}
V2 = {**V1, "name": "V2 + стовпці з питання", "q_cols": True}
V3 = {**V2, "name": "V3 + рядки з питання", "q_rows": 2}
V4 = {**V3, "name": "V4 + рядки з питання ПОВНІСТЮ", "q_row_full": True}
V5 = {**V4, "name": "V5 те саме, cap 26", "cap": 26, "q_rows": 3}
V6 = {**V5, "name": "V6 без засіву від коду", "seed": 0, "seed_row": 0,
      "seed_twin": 0}
# The judge repeated the program's own cell choice in 81% of its failures
# (5.39), although it is never told which cells the program used. The list is
# the channel: seeds plus their row and twin neighbours fill it with the
# program's reading of the table, and the question's own cells end up a
# minority. V7 keeps the seeds -- the detector needs them present to be able
# to agree -- but stops expanding from them, so every other slot is filled
# from the question.
V7 = {**V5, "name": "V7 засів без розширення", "seed_row": 0, "seed_twin": 0}
V8 = {**V7, "name": "V8 те саме, cap 20", "cap": 20}
V9 = {**V7, "name": "V9 засів 2, решта з питання", "seed": 2}
VARIANTS = [V0, V5, V6, V7, V8, V9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    args = ap.parse_args()

    code_rows = cache_io.load_rows(os.path.join(CACHE, f"{args.tag}.jsonl"))[0]
    dir_rows = cache_io.load_rows(
        os.path.join(CACHE, f"baseline_table_{args.tag}.jsonl"))[0]
    rows = build(code_rows, dir_rows)
    by_id = {r["id"]: r for r in code_rows}
    recs = {r["id"]: r for r in fq.load_split("test")}

    items = []
    for r in rows:
        x = by_id[r["id"]]
        code = final_code(x)
        tables = fq.to_tables(recs.get(r["id"], {}))
        if not code or not tables:
            continue
        e = examine(code, tables)
        if e["base"] is None:
            continue
        g = cell_operands(x.get("gold_program"))
        if not g:
            continue
        # Operands that are in no cell come from the document text, which no
        # retrieval over the table can reach. Scored separately, not counted
        # as a retrieval failure.
        vals = [c["value"] for c in numeric_cells(next(iter(tables.values())))]
        reachable = all(any(abs(v - u) <= 1e-4 * max(abs(v), 1.0)
                            for u in vals) for v in g)
        items.append({"df": next(iter(tables.values())), "deps": e["deps"],
                      "q": x.get("question"), "gold_ops": g,
                      "correct": r["correct"], "reachable": reachable,
                      "measured_right": covers(
                          [d["value"] for d in e["deps"]], g) is True})

    unreach = [it for it in items if not it["reachable"]]
    items = [it for it in items if it["reachable"]]
    wrong_miss = [it for it in items
                  if not it["correct"] and not it["measured_right"]]
    allw = [it for it in items if not it["correct"]]
    right = [it for it in items if it["correct"]]
    print(f"відкинуто як недосяжні з таблиці (числа з тексту): {len(unreach)}")
    print("=" * 78)
    print(f"КАНДИДАТИ — {args.tag}, нуль викликів API")
    print("=" * 78)
    print(f"питань із числовим gold: {len(items)}  "
          f"(хибних {len(allw)}, з них «не ті клітинки» {len(wrong_miss)}, "
          f"правильних {len(right)})")
    print()
    print(f"  {'варіант':34s} {'не ті клітинки':>15s} {'усі хибні':>11s}"
          f" {'правильні':>11s} {'розмір':>8s} {'якір':>8s} {'код є':>9s}")
    print("  якір = частка кандидатів із рядків, які читав код: чим менше, "
          "тим слабша прив'язка")
    print("  код є = чи всі клітинки коду присутні, без цього детектор "
          "не працює")

    for cfg in VARIANTS:
        def rec(sub):
            if not sub:
                return 0.0
            k = sum(1 for it in sub
                    if covers([c["value"] for c in build_cands(
                        it["df"], it["deps"], it["q"], cfg)],
                        it["gold_ops"]) is True)
            return k / len(sub)

        sizes, anchor, present = [], [], []
        for it in items:
            cs = build_cands(it["df"], it["deps"], it["q"], cfg)
            sizes.append(len(cs))
            dep = {(d["row"], str(d["col"])) for d in it["deps"]}
            deprows = {d["row"] for d in it["deps"]}
            if cs:
                anchor.append(sum(1 for c in cs if c["row"] in deprows)
                              / len(cs))
            present.append(all((c["row"], c["col"]) in
                               {(x["row"], x["col"]) for x in cs}
                               for c in [{"row": r, "col": cl}
                                         for r, cl in dep]) if dep else True)
        print(f"  {cfg['name']:34s} {rec(wrong_miss):15.0%} {rec(allw):11.0%}"
              f" {rec(right):11.0%} {sum(sizes) / len(sizes):8.1f}"
              f" {sum(anchor) / max(len(anchor), 1):8.0%}"
              f" {sum(present) / max(len(present), 1):9.0%}")

    print()
    print("  «не ті клітинки» — це та підмножина, де суддя взагалі має шанс")
    print("  щось виправити. Саме її recall визначає стелю ремонту.")


if __name__ == "__main__":
    main()
