"""Does the system find the right table on its own, and what does it cost?

Until now each question arrived with its table attached, which is the thing the
committee objected to: there was nothing to retrieve. Here all 60 tables are one
corpus and the question has to find its own.

Retrieval is BM25 over the catalog -- zero LLM calls -- so recall can be
measured and tuned for free.

End-to-end accuracy is then measured honestly without paying twice: where
retrieval returns the right table, the retrieved table IS the given table, so
the cached run applies unchanged and no call is made. Only a retrieval MISS
needs a new call, because the code then has to be written against a different
table. `--run-misses` is what spends quota, and it says how much before it does.

Run:  python -m eval.databench_retrieval --dev eval/databench_dev.json
      python -m eval.databench_retrieval --dev ... --run-misses
"""
import argparse
import collections
import json
import os
import time

from eval import cache_io
from ingestion import databench_catalog as cat
from ingestion import databench_loader as db

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "databench_cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=os.path.join(HERE, "databench_dev.json"))
    ap.add_argument("--arm", default="B")
    ap.add_argument("--tag", default="dev100")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--run-misses", action="store_true",
                    help="витратити квоту: прогнати арм на знайдених таблицях")
    args = ap.parse_args()

    with open(args.dev, encoding="utf-8") as f:
        spec = json.load(f)
    qs = [q for q in spec["questions"] if db.have_table(q["dataset"])]

    r = cat.Retriever()
    print("=" * 76)
    print(f"ПОШУК ТАБЛИЦІ В КОРПУСІ — {len(r.ids)} таблиць, BM25, "
          f"нуль викликів LLM")
    print("=" * 76)
    print(f"  питань: {len(qs)}")

    hits = {}
    for q in qs:
        top = r.search(q["question"], max(args.k, 3))
        ids = [t for t, _ in top]
        hits[q["qid"]] = {"gold": q["dataset"], "top": ids,
                          "rank": (ids.index(q["dataset"]) + 1
                                   if q["dataset"] in ids else 0)}

    r1 = sum(1 for h in hits.values() if h["rank"] == 1)
    r3 = sum(1 for h in hits.values() if 1 <= h["rank"] <= 3)
    n = len(qs)
    print()
    print(f"  recall@1 : {r1}/{n} = {r1 / n:.1%}")
    print(f"  recall@3 : {r3}/{n} = {r3 / n:.1%}")
    mrr = sum(1 / h["rank"] for h in hits.values() if h["rank"]) / n
    print(f"  MRR      : {mrr:.3f}")

    print()
    print("--- ЗА ТИПОМ ВІДПОВІДІ ---")
    print(f"  {'тип':16s} {'n':>4s} {'recall@1':>10s} {'recall@3':>10s}")
    for t in db.TYPES:
        sub = [q for q in qs if q["type"] == t]
        if not sub:
            continue
        a = sum(1 for q in sub if hits[q["qid"]]["rank"] == 1)
        b = sum(1 for q in sub if 1 <= hits[q["qid"]]["rank"] <= 3)
        print(f"  {t:16s} {len(sub):4d} {a / len(sub):9.0%} {b / len(sub):9.0%}")

    miss = [q for q in qs if hits[q["qid"]]["rank"] != 1]
    print()
    print(f"--- ПРОМАХИ @1: {len(miss)} ---")
    for q in miss[:12]:
        h = hits[q["qid"]]
        print(f"  потрібна {h['gold']:18s} знайдено {h['top'][0]:18s} "
              f"(ранг {h['rank'] or '>3'})")
        print(f"      {q['question'][:84]}")

    # ---- end to end ----
    path = os.path.join(CACHE, f"{args.arm}_{args.tag}.jsonl")
    given = {x["id"]: x for x in cache_io.load_rows(path)[0]}
    rpath = os.path.join(CACHE, f"{args.arm}-rag_{args.tag}.jsonl")
    retr = {x["id"]: x for x in cache_io.load_rows(rpath)[0]}

    need = [q for q in miss if q["qid"] not in retr]
    print()
    print(f"--- НАСКРІЗНА ТОЧНІСТЬ ---")
    print(f"  влучень @1  : {r1} — кешований прогін застосовується як є, "
          f"нуль викликів")
    print(f"  промахів @1 : {len(miss)}, з них уже прогнано {len(miss) - len(need)}")
    print(f"  ПОТРІБНО ВИКЛИКІВ: {len(need)}"
          + ("" if args.run_misses else "  (не витрачено: без --run-misses)"))

    if args.run_misses and need:
        from eval import databench_run as dr
        from generation import llm_client
        os.makedirs(CACHE, exist_ok=True)
        with open(rpath, "a", encoding="utf-8") as fh:
            for i, q in enumerate(need, 1):
                found = hits[q["qid"]]["top"][0]
                df = db.table(found)
                row = dict(q)
                row["dataset"] = found      # the code sees the RETRIEVED table
                llm_client.reset_stats()
                val, meta, steps = None, {}, []
                err = err_class = None
                try:
                    val, meta, steps = dr.run_code(row, df, False)
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
                out = {"id": q["qid"], "arm": f"{args.arm}-rag",
                       "gold_dataset": q["dataset"], "dataset": found,
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
                    break
                print(f"  [{i}/{len(need)}] {q['dataset']} -> {found} "
                      f"{'OK' if ok else 'ні'}")
                time.sleep(dr.SLEEP)
        retr = {x["id"]: x for x in cache_io.load_rows(rpath)[0]}

    ok_given = ok_rag = scored = 0
    unresolved = 0
    for q in qs:
        g = given.get(q["qid"])
        if g is None:
            continue
        scored += 1
        ok_given += bool(g["correct"])
        if hits[q["qid"]]["rank"] == 1:
            ok_rag += bool(g["correct"])
        elif q["qid"] in retr:
            ok_rag += bool(retr[q["qid"]]["correct"])
        else:
            unresolved += 1

    print()
    print(f"  {'конфігурація':46s} {'правильних':>11s} {'точність':>9s}")
    print(f"  {'B на ДАНІЙ таблиці (як раніше)':46s} {ok_given:11d} "
          f"{ok_given / max(scored, 1):9.1%}")
    lbl = "B на ЗНАЙДЕНІЙ таблиці (справжній RAG)"
    print(f"  {lbl:46s} {ok_rag:11d} {ok_rag / max(scored, 1):9.1%}")
    print(f"  різниця: {100 * (ok_rag - ok_given) / max(scored, 1):+.1f} п.п.")
    if unresolved:
        print(f"  !! {unresolved} промахів не прогнано: вони рахуються як "
              f"хибні, тож число вище є НИЖНЬОЮ оцінкою.")
        print(f"     Прогін на чужій таблиці майже напевно дає хибну "
              f"відповідь, тож оцінка щільна, але це припущення, "
              f"а не вимір — --run-misses його перевіряє.")

    out = os.path.join(HERE, f"databench_retrieval_{args.tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("qid,gold_table,found_table,rank,type,question\n")
        for q in qs:
            h = hits[q["qid"]]
            qq = str(q["question"]).replace(",", ";").replace('"', "'")
            f.write(f"{q['qid']},{h['gold']},{h['top'][0]},{h['rank']},"
                    f"{q['type']},\"{qq}\"\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
