"""Pick the dev questions and fetch their tables. Zero API calls.

The split is by TABLE, not by question: dev comes from `semeval/train`
(datasets 001-049) and the frozen test from `semeval/dev` (050-065), which
share no dataset. So nothing tuned here can have been tuned on a test table.

Also measures what the direct-answer arm is up against: how many rows each
table has, and therefore what fraction of it fits in a prompt. On DataBench
that fraction is the whole reason a code route can win, so it is measured
rather than asserted.

Run:  python -m eval.databench_prep --n 100
"""
import argparse
import json
import os

from ingestion import databench_loader as db

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "databench_dev.json")
TOPK = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = db.questions(args.split)
    rows = [r for r in rows if r.get("answer") not in (None, "")]
    sel = db.stratified(rows, args.n)
    need = sorted({r["dataset"] for r in sel})
    print(f"обрано {len(sel)} питань, {len(need)} таблиць")
    cnt = {t: sum(1 for r in sel if r["type"] == t) for t in db.TYPES}
    print("  за типом відповіді:", cnt)
    print()

    info, failed = {}, []
    for i, ds in enumerate(need, 1):
        try:
            full = db.table(ds, lite=False)
            lite = db.table(ds, lite=True)
        except Exception as exc:
            failed.append((ds, f"{type(exc).__name__}: {exc}"))
            print(f"  [{i}/{len(need)}] {ds:28s} НЕ ЗАВАНТАЖИЛОСЬ")
            continue
        info[ds] = {"rows": int(full.shape[0]), "cols": int(full.shape[1]),
                    "lite_rows": int(lite.shape[0])}
        print(f"  [{i}/{len(need)}] {ds:28s} {full.shape[0]:>8d} x "
              f"{full.shape[1]:>3d}   top{TOPK} = "
              f"{min(1.0, TOPK / max(full.shape[0], 1)):6.1%} таблиці")

    if failed:
        print()
        print(f"не завантажилось {len(failed)}:")
        for ds, why in failed:
            print(f"    {ds}: {why}")
        sel = [r for r in sel if r["dataset"] in info]
        print(f"питань після відкидання: {len(sel)}")

    sizes = sorted(v["rows"] for v in info.values())
    if sizes:
        med = sizes[len(sizes) // 2]
        print()
        print(f"--- РОЗМІРИ ТАБЛИЦЬ ---")
        print(f"  рядків: від {sizes[0]} до {sizes[-1]}, медіана {med}")
        frac = [min(1.0, TOPK / max(v['rows'], 1)) for v in info.values()]
        frac.sort()
        print(f"  частка таблиці у вікні top{TOPK}: медіана "
              f"{frac[len(frac) // 2]:.1%}, найбільша {frac[-1]:.0%}")
        big = sum(1 for v in info.values() if v["rows"] > 200)
        print(f"  таблиць, де пряма відповідь бачить <15% рядків: "
              f"{big}/{len(info)}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"questions": sel, "tables": info, "topk": TOPK}, f,
                  ensure_ascii=False)
    print()
    print("Записано:", args.out)


if __name__ == "__main__":
    main()
