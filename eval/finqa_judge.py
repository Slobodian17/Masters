"""The judge: forced choice among candidate cells, not a yes/no on its own work.

5.36 measured a ceiling. The dependency set a program actually reads separates
right from wrong at recall 72.7% / precision 87.7% -- but the comparison that
produces those numbers looks at the gold operands, so it is a ceiling and not
a detector. One question is left: can a single model call stand in for the gold
comparison?

The call is deliberately NOT "is this cell right?". Self-judging a yes/no is
the shape that failed for others -- P(True) self-evaluation on table QA scores
AUROC 0.42-0.70, essentially a coin flip (arXiv 2604.12491). A forced choice
among concrete alternatives is a different and easier decision, so the model is
shown the measured cells mixed in with their closest neighbours -- the same row
in other periods, rows with similar labels, and cells whose labels match the
question -- in one list, with nothing marking which the program used. It picks
the cells the question needs.

That makes the judgement independent of the program rather than a review of it:
the model cannot anchor on a choice it is not shown. The cost is that the
candidate list has to contain the right cells in the first place, so the
candidate recall is reported as the judge's own ceiling.

Scoring uses the gold operands; the prompt never does.

Run:  python -m eval.finqa_judge --tag confirm300 --limit 30     # pilot
      python -m eval.finqa_judge --tag confirm300
"""
import argparse
import json
import os
import random
import re
import time

from eval import cache_io
from eval.finqa_confirm import build
from eval.finqa_perturb import cell_operands, examine, final_code
from generation import llm_client
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")

MAX_CAND = 14
MAX_TOKENS = 900
SLEEP = 4.2          # flash-lite allows 15 requests per minute
TOL = 0.01

SYSTEM = ("You select which numbers from a table a question needs. "
          "You answer with one line and nothing else.")

TEMPLATE = """Question: {q}

Candidate numbers from the table. Each line is: index | row label | column | value
{cands}

Which candidates does the question need in order to compute its answer?
Choose the smallest set that is sufficient. Think about which row the question
names and which period or column it asks for.

Reply with exactly one line, no explanation:
CELLS: <comma-separated indices>"""


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower()).split()


def numeric_cells(df):
    import pandas as pd
    out = []
    for col in df.columns[1:]:
        s = pd.to_numeric(df[col], errors="coerce")
        for i, v in s.items():
            if pd.notna(v):
                out.append({"row": i, "col": str(col), "value": float(v),
                            "label": str(df.iat[i, 0])})
    return out


def candidates(df, deps, question):
    """The measured cells plus their closest alternatives, in table order.

    Three kinds of neighbour, because the errors of 5.36 are of three kinds:
    the wrong period (same row, other columns), the wrong line item (a row
    with a similar label), and a row the program never looked at but the
    question names (lexical match).
    """
    cells = numeric_cells(df)
    by_key = {(c["row"], c["col"]): c for c in cells}
    picked = {}

    def add(c):
        picked[(c["row"], c["col"])] = c

    seeds = [by_key[(d["row"], str(d["col"]))] for d in deps
             if (d["row"], str(d["col"])) in by_key]
    for c in seeds[:6]:
        add(c)

    # same row, other periods
    for c in seeds[:3]:
        for o in cells:
            if o["row"] == c["row"] and o["col"] != c["col"]:
                add(o)

    # rows whose label looks like the seed's label
    for c in seeds[:2]:
        want = set(norm(c["label"]))
        sib = sorted((o for o in cells if o["row"] != c["row"]
                      and o["col"] == c["col"]),
                     key=lambda o: -len(want & set(norm(o["label"]))))
        for o in sib[:3]:
            add(o)

    # cells the question's own words point at
    qt = set(norm(question))
    lex = sorted(cells, key=lambda o: -(len(qt & set(norm(o["label"])))
                                        + len(qt & set(norm(o["col"])))))
    for o in lex[:4]:
        add(o)

    # Table order, so the ordering cannot hint at which cells the code used.
    out = sorted(picked.values(), key=lambda c: (c["row"], c["col"]))
    return out[:MAX_CAND]


def render(cands):
    return "\n".join(
        f"{i + 1} | {c['label'][:44]} | {c['col'][:24]} | {c['value']:g}"
        for i, c in enumerate(cands))


def parse_choice(raw, n):
    m = re.search(r"CELLS\s*:?\s*([0-9,\s]+)", str(raw or ""), re.I)
    if not m:
        m = re.search(r"([0-9]+(?:\s*,\s*[0-9]+)*)", str(raw or ""))
    if not m:
        return []
    out = []
    for t in re.split(r"[,\s]+", m.group(1).strip()):
        if t.isdigit() and 1 <= int(t) <= n and int(t) not in out:
            out.append(int(t))
    return out


def covers(values, targets, tol=TOL):
    """Is every target present among values?"""
    if not targets:
        return None
    return all(any(abs(t - v) <= tol * max(abs(t), 1.0) for v in values)
               for t in targets)


def same_set(a, b, tol=TOL):
    if len(a) != len(b):
        return False
    rest = list(b)
    for x in a:
        hit = next((y for y in rest if abs(x - y) <= tol * max(abs(x), 1.0)),
                   None)
        if hit is None:
            return False
        rest.remove(hit)
    return True


def _perturb_rows(tag):
    import csv as _csv
    p = os.path.join(HERE, f"finqa_perturb_{tag}.csv")
    if not os.path.exists(p):
        raise SystemExit(f"немає {p}: спершу python -m eval.finqa_perturb")
    return [d for d in _csv.DictReader(open(p, encoding="utf-8"))
            if d["repro"] == "1"]


def _split(tag, n_control):
    """The questions worth spending a call on, and a matched control group.

    `critical` is where the measured cell set does NOT cover the gold cells and
    the answer is wrong: the only questions a judge could repair. The control
    group is correct answers, needed because a judge that disagrees there would
    send a right answer to repair and break it.
    """
    rows = _perturb_rows(tag)
    crit = [d["id"] for d in rows
            if d["correct"] == "0" and d["loc"] in ("HIT_SOME", "HIT_NONE")]
    ok = [d["id"] for d in rows if d["correct"] == "1"]
    random.Random(20261031).shuffle(ok)
    return crit, ok[:n_control]


def critical_ids(tag, n_control):
    crit, ctrl = _split(tag, n_control)
    return set(crit) | set(ctrl)


def keep_counts(tag, n_control):
    crit, ctrl = _split(tag, n_control)
    return f"хибних з не тими клітинками {len(crit)}, контроль {len(ctrl)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--subset", default="",
                    help="critical: лише питання, де код узяв не ті клітинки, "
                         "плюс стільки ж правильних для виміру зламів")
    ap.add_argument("--n-control", type=int, default=55)
    ap.add_argument("--cands", default="V0",
                    help="варіант генератора кандидатів із eval.finqa_cands")
    args = ap.parse_args()

    # V0 is this module's own builder, kept so the baseline run stays
    # reproducible; anything else comes from the tuned sweep.
    if args.cands == "V0":
        def make_cands(df, deps, q):
            return candidates(df, deps, q)
    else:
        from eval import finqa_cands as fc
        cfg = getattr(fc, args.cands)

        def make_cands(df, deps, q):
            return fc.build_cands(df, deps, q, cfg)

    suffix = "" if args.cands == "V0" else f"_{args.cands.lower()}"
    out_path = args.out or os.path.join(
        CACHE, f"judge{suffix}_{args.tag}.jsonl")
    done = {r["id"] for r in cache_io.load_rows(out_path)[0]}

    code_rows = cache_io.load_rows(os.path.join(CACHE, f"{args.tag}.jsonl"))[0]
    dir_rows = cache_io.load_rows(
        os.path.join(CACHE, f"baseline_table_{args.tag}.jsonl"))[0]
    rows = build(code_rows, dir_rows)
    by_id = {r["id"]: r for r in code_rows}
    recs = {r["id"]: r for r in fq.load_split("test")}

    print(f"модель {llm_client.active_model()} / "
          f"{llm_client.active_provider()}, кеш {out_path}")
    print(f"уже зроблено {len(done)}")

    todo = []
    for r in rows:
        x = by_id[r["id"]]
        code = final_code(x)
        tables = fq.to_tables(recs.get(r["id"], {}))
        if not code or not tables:
            continue
        todo.append((r, x, code, tables))
    if args.subset == "critical":
        keep = critical_ids(args.tag, args.n_control)
        todo = [t for t in todo if t[0]["id"] in keep]
        print(f"підмножина critical: {len(keep)} id "
              f"({keep_counts(args.tag, args.n_control)})")
    if args.limit:
        todo = todo[:args.limit]
    print(f"питань до опитування: {len(todo)}")

    with open(out_path, "a", encoding="utf-8") as fh:
        for n, (r, x, code, tables) in enumerate(todo, 1):
            qid = r["id"]
            if qid in done:
                continue
            df = next(iter(tables.values()))
            e = examine(code, tables)
            if e["base"] is None:
                continue
            cands = make_cands(df, e["deps"], x.get("question"))
            if not cands:
                continue
            gold_ops = cell_operands(x.get("gold_program"))
            prompt = TEMPLATE.format(q=x.get("question"),
                                     cands=render(cands))
            llm_client.reset_stats()
            t0 = time.perf_counter()
            raw, err, err_class = "", None, None
            try:
                raw = llm_client.call_llm(prompt, system=SYSTEM,
                                          tag="judge", max_tokens=MAX_TOKENS)
            except Exception as exc:
                err, err_class = f"{type(exc).__name__}: {exc}", type(exc).__name__
            idx = parse_choice(raw, len(cands))
            chosen = [cands[i - 1] for i in idx]
            row = {
                "id": qid,
                "question": x.get("question"),
                "gold": r["gold"],
                "gold_program": x.get("gold_program"),
                "gold_ops": gold_ops,
                "r2": r["r2"],
                "r2_correct": bool(r["correct"]),
                "n_cands": len(cands),
                "cand_values": [c["value"] for c in cands],
                "measured": [d["value"] for d in e["deps"]],
                "measured_cells": [{"label": d["label"], "col": str(d["col"]),
                                    "value": d["value"]} for d in e["deps"]],
                "chosen_idx": idx,
                "chosen": [c["value"] for c in chosen],
                "chosen_cells": [{"label": c["label"], "col": c["col"],
                                  "value": c["value"]} for c in chosen],
                "raw": str(raw)[:400],
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "tokens": (llm_client.get_stats() or {}).get("total_tokens"),
                "error": err,
                "error_class": err_class,
            }
            try:
                cache_io.write_row(fh, row)
            except cache_io.FakeRow as bad:
                print(f"  [{n}/{len(todo)}] {qid}: транспорт, не записано — {bad}")
                break
            mark = "ok" if not err else "ERR"
            print(f"  [{n}/{len(todo)}] {qid} {mark} вибрав {idx} "
                  f"(кандидатів {len(cands)}, код брав {len(e['deps'])})")
            time.sleep(SLEEP)

    report(out_path)


def report(path):
    rows = cache_io.load_rows(path)[0]
    rows = [r for r in rows if not r.get("error")]
    # Recomputed rather than read: rows written before defect 22 stored gold
    # operands that included scale constants, which a cell cannot supply.
    for r in rows:
        r["gold_ops"] = cell_operands(r.get("gold_program"))
    rows = [r for r in rows if r["gold_ops"]]
    if not rows:
        print("немає рядків для звіту")
        return
    wrong = [r for r in rows if not r["r2_correct"]]
    right = [r for r in rows if r["r2_correct"]]

    print()
    print("=" * 78)
    print(f"СУДДЯ ВИБОРОМ — n={len(rows)} (хибних {len(wrong)}, "
          f"правильних {len(right)})")
    print("=" * 78)

    def frac(sub, pred):
        k = sum(1 for r in sub if pred(r))
        return k, (k / len(sub) if sub else 0.0)

    print()
    print("--- СТЕЛЯ САМОГО СУДДІ: чи є gold-операнди серед кандидатів ---")
    for name, sub in (("хибні", wrong), ("правильні", right)):
        k, p = frac(sub, lambda r: covers(r["cand_values"], r["gold_ops"]) is True)
        print(f"  {name:12s} {k:3d}/{len(sub):3d} = {p:5.1%}")

    print()
    print("--- ЧИ ВИБРАВ СУДДЯ ПРАВИЛЬНІ КЛІТИНКИ (покриває gold-операнди) ---")
    for name, sub in (("хибні", wrong), ("правильні", right)):
        k, p = frac(sub, lambda r: covers(r["chosen"], r["gold_ops"]) is True)
        print(f"  {name:12s} {k:3d}/{len(sub):3d} = {p:5.1%}")
    print("  для порівняння — те саме для ВИМІРЯНОЇ множини коду:")
    for name, sub in (("хибні", wrong), ("правильні", right)):
        k, p = frac(sub, lambda r: covers(r["measured"], r["gold_ops"]) is True)
        print(f"  {name:12s} {k:3d}/{len(sub):3d} = {p:5.1%}")

    print()
    print("--- ДЕТЕКТОР: суддя вибрав ІНШІ клітинки, ніж узяв код ---")
    tw, _ = frac(wrong, lambda r: not same_set(r["chosen"], r["measured"]))
    tc, _ = frac(right, lambda r: not same_set(r["chosen"], r["measured"]))
    rec = tw / max(len(wrong), 1)
    pre = tw / max(tw + tc, 1)
    print(f"  позначено хибних    : {tw}/{len(wrong)} → recall    {rec:5.1%}")
    print(f"  позначено правильних: {tc}/{len(right)} → precision {pre:5.1%}")
    print(f"  (стеля з 5.36: recall 72.7%, precision 87.7%)")
    print(f"  (згода маршрутів 5.26: recall 46.5%, precision 93.9%)")

    print()
    print("--- РЕМОНТНА ЦІННІСТЬ: там, де код узяв не ті клітинки ---")
    miss = [r for r in wrong if covers(r["measured"], r["gold_ops"]) is False]
    k, p = frac(miss, lambda r: covers(r["chosen"], r["gold_ops"]) is True)
    print(f"  таких питань                        : {len(miss)}")
    print(f"  суддя назвав ПРАВИЛЬНІ клітинки     : {k} = {p:.1%}")
    print("  це стеля адресного ремонту: якщо клітинки названо правильно,")
    print("  ремонт має шанс; якщо ні — не має.")

    print()
    print("--- ШКОДА: там, де код узяв ПРАВИЛЬНІ клітинки ---")
    ok = [r for r in right if covers(r["measured"], r["gold_ops"]) is True]
    k, p = frac(ok, lambda r: covers(r["chosen"], r["gold_ops"]) is not True)
    print(f"  таких питань                        : {len(ok)}")
    print(f"  суддя зіпсував би вибір             : {k} = {p:.1%}")

    tok = sum(r.get("tokens") or 0 for r in rows)
    print()
    print(f"--- ЦІНА: {len(rows)} викликів, {tok:,} токенів, "
          f"сер. {tok / max(len(rows), 1):.0f} ток/виклик ---")


if __name__ == "__main__":
    main()
