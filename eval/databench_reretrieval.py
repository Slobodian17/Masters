"""Execution-guided table re-retrieval: when the code fails, try the next table.

The retriever gets the table right at rank 1 only 48-49% of the time (5.45),
and a wrong table is not a subtle failure -- the code asks for a column that is
not there, or filters everything away and returns nothing. That signal is free,
deterministic, and available before any answer is scored, so it can be used to
fall back to the next candidate instead of returning a result the system
already has reason to distrust.

TRIGGER V1, chosen on dev and gold-free by construction:

  NO_VALUE   the code returned nothing after crash repair
  EXCEPTION  the last iteration still raised
  EMPTY      the answer is '', '[]', nan or inf
  SHAPE      a list was expected and a scalar came back, or the reverse
             (the expected type is part of the QUESTION, not the answer)

Measured on dev before building: the trigger fires on 21 of 100 questions, the
right table is among ranks 2-3 for 6 of them, and **0 of the 21 were already
correct** -- so the block cannot break a right answer, only spend calls. That
is why the wider trigger was chosen over the narrower column-existence one
(6 fires, 3 fixable).

If every candidate triggers, the top-1 answer is kept, so the block is never
worse than plain top-1 retrieval.

Run:  python -m eval.databench_reretrieval --dev eval/databench_dev.json \
          --tag dev100                      # plan only, no calls
      ... --run                             # spends the budget
"""
import argparse
import json
import os
import time
from math import comb

from eval import cache_io
from eval import databench_run as dr
from generation import llm_client
from ingestion import databench_catalog as cat
from ingestion import databench_loader as db

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "databench_cache")
MAX_RANK = 3


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def triggered(row, qtype):
    """V1. Gold-free: nothing here looks at the reference answer."""
    if row is None or row.get("value") is None:
        return "NO_VALUE"
    steps = row.get("steps") or []
    if steps and steps[-1].get("exception_type"):
        return "EXCEPTION"
    v = str(row["value"]).strip()
    if v in ("", "[]", "nan", "None", "inf", "-inf"):
        return "EMPTY"
    if str(qtype).startswith("list") != v.startswith("["):
        return "SHAPE"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=os.path.join(HERE, "databench_dev.json"))
    ap.add_argument("--tag", default="dev100")
    ap.add_argument("--run", action="store_true",
                    help="витратити квоту; без цього лише план")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    with open(args.dev, encoding="utf-8") as f:
        spec = json.load(f)
    qs = [q for q in spec["questions"] if db.have_table(q["dataset"])]
    r = cat.Retriever()

    B = {x["id"]: x for x in cache_io.load_rows(
        os.path.join(CACHE, f"B_{args.tag}.jsonl"))[0]}
    RG = {x["id"]: x for x in cache_io.load_rows(
        os.path.join(CACHE, f"B-rag_{args.tag}.jsonl"))[0]}
    path = os.path.join(CACHE, f"B-rag2_{args.tag}.jsonl")
    done = {}
    for x in cache_io.load_rows(path)[0]:
        done.setdefault(x["id"], []).append(x)

    print("=" * 76)
    print(f"ПОВТОРНИЙ ПОШУК ЗА СИГНАЛОМ ВИКОНАННЯ (V1) — {args.tag}")
    print("=" * 76)

    plan = []
    for q in qs:
        top = [t for t, _ in r.search(q["question"], MAX_RANK)]
        base = B[q["qid"]] if (top and top[0] == q["dataset"]) else RG.get(q["qid"])
        why = triggered(base, q["type"])
        if why:
            plan.append((q, top, base, why))
    print(f"  питань {len(qs)}, тригер спрацював на {len(plan)}")
    print(f"  правильна таблиця серед рангів 2-3: "
          f"{sum(1 for q, t, b, w in plan if q['dataset'] in t[1:])}")
    print(f"  уже прогнано спроб: {sum(len(v) for v in done.values())}")
    need = sum(max(0, (MAX_RANK - 1) - len(done.get(q['qid'], [])))
               for q, t, b, w in plan)
    print(f"  ПОТРІБНО ВИКЛИКІВ: до {need}"
          + ("" if args.run else "  (не витрачено: без --run)"))

    if args.run and plan:
        with open(path, "a", encoding="utf-8") as fh:
            for i, (q, top, base, why) in enumerate(plan, 1):
                have = {x["rank"] for x in done.get(q["qid"], [])}
                for rank in range(2, MAX_RANK + 1):
                    if rank in have:
                        continue
                    if len(top) < rank:
                        break
                    cand = top[rank - 1]
                    row = dict(q)
                    row["dataset"] = cand
                    llm_client.reset_stats()
                    val, meta, steps = None, {}, []
                    err = err_class = None
                    try:
                        val, meta, steps = dr.run_code(row, db.table(cand), False)
                    except Exception as exc:
                        err = f"{type(exc).__name__}: {exc}"
                        err_class = type(exc).__name__
                    truth = db.truth_of(q, False)
                    ok = False
                    if val is not None and not err:
                        try:
                            ok = db.compare(val, truth, q["type"])
                        except Exception:
                            ok = False
                    out = {"id": q["qid"], "rank": rank, "dataset": cand,
                           "gold_dataset": q["dataset"], "trigger": why,
                           "question": q["question"], "type": q["type"],
                           "truth": str(truth),
                           "value": None if val is None else str(val),
                           "correct": bool(ok), "steps": steps, **meta,
                           "tokens": (llm_client.get_stats() or {}).get("total_tokens"),
                           "error": err, "error_class": err_class}
                    try:
                        cache_io.write_row(fh, out)
                    except cache_io.FakeRow as bad:
                        print(f"  [{i}] {q['qid']}: транспорт — {bad}")
                        return
                    still = triggered(out, q["type"])
                    print(f"  [{i}/{len(plan)}] {q['dataset']} -> ранг{rank} "
                          f"{cand} {'OK' if ok else 'ні'}"
                          f"{'' if not still else ' (знову ' + still + ')'}")
                    time.sleep(dr.SLEEP)
                    if not still:
                        break
        done = {}
        for x in cache_io.load_rows(path)[0]:
            done.setdefault(x["id"], []).append(x)

    # ---- scoring ----
    base_ok = blk_ok = 0
    fixed = broke = 0
    unresolved = 0
    for q in qs:
        top = [t for t, _ in r.search(q["question"], MAX_RANK)]
        base = B[q["qid"]] if (top and top[0] == q["dataset"]) else RG.get(q["qid"])
        b_ok = bool(base and base.get("correct"))
        base_ok += b_ok
        why = triggered(base, q["type"])
        if not why:
            blk_ok += b_ok
            continue
        # first candidate that does not trigger again wins
        tries = sorted(done.get(q["qid"], []), key=lambda x: x["rank"])
        pick = next((x for x in tries if not triggered(x, q["type"])), None)
        if pick is None:
            if not tries:
                unresolved += 1
            blk_ok += b_ok            # keep top-1: never worse
            continue
        n_ok = bool(pick["correct"])
        blk_ok += n_ok
        fixed += (n_ok and not b_ok)
        broke += (b_ok and not n_ok)

    n = len(qs)
    print()
    print(f"  {'конфігурація':44s} {'правильних':>11s} {'точність':>9s}")
    print(f"  {'B-rag, лише top-1':44s} {base_ok:11d} {base_ok / n:9.1%}")
    print(f"  {'B-rag + повторний пошук за виконанням':44s} "
          f"{blk_ok:11d} {blk_ok / n:9.1%}")
    print(f"  різниця {100 * (blk_ok - base_ok) / n:+.1f} п.п.")
    print(f"  виправлено {fixed}, зламано {broke}, "
          f"Макнемар p = {mcnemar(fixed, broke):.4f}")
    if unresolved:
        print(f"  !! {unresolved} спрацювань не прогнано — число є НИЖНЬОЮ "
              f"оцінкою; запустіть із --run")
    tok = sum(x.get("tokens") or 0 for v in done.values() for x in v)
    if tok:
        print(f"  ціна: {sum(len(v) for v in done.values())} викликів, "
              f"{tok:,} токенів")


if __name__ == "__main__":
    main()
