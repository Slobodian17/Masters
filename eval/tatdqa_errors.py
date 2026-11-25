"""Why the gold arm is wrong: bad table, bad scoring, or a bad answer.

The distinction decides who has to change. (a) and (b) are defects of the
harness and must be fixed before the pdf arm runs, or the two arms would be
measured by different code and their difference would mean nothing. Only (c)
is a statement about the method.

Zero API calls: everything is recomputed from the cache and the annotation.

  TABLE   the fact numbers the gold answer used are NOT in the table we built,
          so no program over that table could have reached the answer
  SCORING the value is right up to scale, percent/fraction, sign or rounding
          past 1% -- right in substance, not counted
  MODEL   the table carried the cells, the value is not the answer in any
          reading

Run:  python -m eval.tatdqa_errors
"""
import argparse
import collections
import os
import random
import zipfile

from eval import cache_io, tatdqa_loader as T
from eval.tatdqa_extract import fact_numbers, norm_num

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")


def near(a, b, tol=0.01):
    try:
        return abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1.0)
    except (TypeError, ValueError):
        return False


def table_nums(df):
    out = set()
    if df is None:
        return out
    for row in df.itertuples(index=False):
        for c in row:
            v = norm_num(c)
            if v is not None:
                out.add(v)
                out.add(abs(v))
    return out


def scoring_excuse(pred, gold):
    """Is the value right in substance but not counted?"""
    if pred is None or gold is None:
        return None
    try:
        pred, gold = float(pred), float(gold)
    except (TypeError, ValueError):
        # A non-numeric candidate (a string the sandbox returned) is not a
        # scoring problem: there is nothing to rescale.
        return None
    for f, name in ((1.0, "округлення"), (100.0, "відсоток/частка"),
                    (1e3, "тисячі"), (1e6, "мільйони"), (1e9, "мільярди")):
        for cand, how in ((pred * f, f"×{name}"), (pred / f, f"÷{name}")):
            if near(cand, gold):
                return how
            if near(-cand, gold):
                return f"знак, {how}"
    if near(pred, gold, 0.05):
        return "у межах 5%, але не 1%"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gold")
    ap.add_argument("--model", default="gemini-flash-lite-latest")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from eval.tatdqa_run import gold_table, extracted_table

    path = os.path.join(CACHE_DIR, f"tatdqa_{args.arm}_{args.model}.jsonl")
    rows, _ = cache_io.load_rows(path)
    zf = zipfile.ZipFile(T.DOCS_ZIP)
    sel = T.select(zf=zf)
    random.Random(args.seed).shuffle(sel)
    by_uid = {q["uid"]: q for q in sel}

    wrong = [r for r in rows if not r.get("correct")]
    print("=" * 74)
    print(f"ЧОМУ ХИБНІ: гілка «{args.arm}», {args.model}")
    print("=" * 74)
    print(f"рядків {len(rows)}, хибних {len(wrong)}")
    print()

    kinds = collections.Counter()
    detail = []
    for r in wrong:
        q = by_uid.get(r["id"])
        if q is None:
            continue
        if args.arm == "gold":
            df = gold_table(q, zf)
            nums = table_nums(df)
        else:
            tabs, _ = extracted_table(q)
            nums = set()
            for d in tabs.values():
                nums |= table_nums(d)
        facts = fact_numbers(q)
        gold = T.numeric_answer(q)
        pred = r.get("candidate")
        excuse = scoring_excuse(pred, gold)

        if facts and not facts <= nums:
            k, why = "TABLE", f"бракує клітинок {sorted(facts - nums)[:3]}"
        elif excuse:
            k, why = "SCORING", excuse
        else:
            k, why = "MODEL", f"наш={pred} еталон={gold}"
        kinds[k] += 1
        detail.append((k, r["id"], str(r["question"])[:52], gold,
                       r.get("scale"), pred, why))

    total = sum(kinds.values())
    for k in ("TABLE", "SCORING", "MODEL"):
        v = kinds.get(k, 0)
        print(f"  {k:8s} {v:3d}  {v / max(total, 1):5.1%}  "
              f"{'#' * int(round(30 * v / max(total, 1)))}")
    print()
    for k in ("SCORING", "TABLE", "MODEL"):
        sub = [d for d in detail if d[0] == k]
        if not sub:
            continue
        print(f"--- {k} ({len(sub)}) ---")
        for _, qid, question, gold, scale, pred, why in sub[:8]:
            p = round(pred, 4) if isinstance(pred, float) else pred
            print(f"  {question}")
            print(f"     еталон={gold} шкала={scale} наш={p}  -> {why}")
        print()

    print("ВИСНОВОК:")
    if kinds.get("SCORING", 0) / max(total, 1) > 0.15:
        print("  SCORING помітна — правило оцінки треба виправити ДО прогону pdf")
    if kinds.get("TABLE", 0) / max(total, 1) > 0.25:
        print("  TABLE помітна — побудова таблиці лишається вадою обв'язки")
    if kinds.get("MODEL", 0) / max(total, 1) > 0.5:
        print("  переважає MODEL — обв'язка більше не головна причина")


if __name__ == "__main__":
    main()
