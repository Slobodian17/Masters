"""The arms against each other, paired per question. Zero API calls.

A and B answer the same questions, so the comparison is paired and the test is
exact McNemar on the discordant pairs. A-lite answers the same questions on a
different table (the 20-row sample) and against a different truth
(`sample_answer`), so it is NOT paired with the others and is reported beside
them rather than tested against them: it is there to separate "the model
reasons worse" from "the model cannot see the data".

Run:  python -m eval.databench_compare --tag dev100
"""
import argparse
import os
from math import comb

from eval import cache_io
from ingestion import databench_loader as db

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "databench_cache")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(arm, tag):
    p = os.path.join(CACHE, f"{arm}_{tag}.jsonl")
    return {r["id"]: r for r in cache_io.load_rows(p)[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev100")
    args = ap.parse_args()

    # An arm that was not run for this tag is simply absent: the frozen test
    # has no A-lite, and B-lite exists only on dev. Reporting what is there
    # beats refusing to report anything.
    arms = {}
    for a in ("A", "A-lite", "B", "B-lite"):
        rows = load(a, args.tag)
        if rows:
            arms[a] = rows
    print("=" * 74)
    print(f"DataBench {args.tag} — flash-lite, офіційне порівняння відповідей")
    print("=" * 74)
    print()
    print(f"  {'арм':40s} {'n':>4s} {'правильних':>11s} {'точність':>9s}")
    labels = {"A": "A  пошук + пряма відповідь (top-30)",
              "A-lite": "A-lite  пряма відповідь, вся таблиця",
              "B": "B  код + ремонт падінь",
              "B-lite": "B-lite  код на 20-рядковій таблиці"}
    for a, rows in arms.items():
        ok = sum(1 for r in rows.values() if r["correct"])
        print(f"  {labels[a]:40s} {len(rows):4d} {ok:11d} {ok / len(rows):9.1%}")

    print()
    print("--- ПАРНО, ОДНІ Й ТІ САМІ ПИТАННЯ Й ОДИН ЕТАЛОН ---")
    ids = sorted(set(arms["A"]) & set(arms["B"]))
    b = sum(1 for q in ids if arms["B"][q]["correct"] and not arms["A"][q]["correct"])
    c = sum(1 for q in ids if arms["A"][q]["correct"] and not arms["B"][q]["correct"])
    oa = sum(1 for q in ids if arms["A"][q]["correct"])
    ob = sum(1 for q in ids if arms["B"][q]["correct"])
    n = len(ids)
    print(f"  спільних питань: {n}")
    print(f"  A {oa}/{n} = {oa / n:.1%}   B {ob}/{n} = {ob / n:.1%}")
    print(f"  B правий / A хибний : {b}")
    print(f"  A правий / B хибний : {c}")
    print(f"  різниця {100 * (ob - oa) / n:+.1f} п.п., "
          f"точний Макнемар p = {mcnemar(b, c):.2e}")

    if "A-lite" not in arms:
        print()
        print("  (A-lite для цього набору не прогонявся — розділ пропущено)")
        al = None
    else:
        print()
        print("--- A-LITE ПОРУЧ, НЕ ПРОТИ ---")
        al = arms["A-lite"]
    if al:
        oal = sum(1 for r in al.values() if r["correct"])
        print(f"  A-lite {oal}/{len(al)} = {oal / len(al):.1%}  "
              f"(20-рядкова таблиця, еталон sample_answer)")
        print(f"  A {oa / n:.1%} -> A-lite {oal / len(al):.1%}: "
              f"{100 * (oal / len(al) - oa / n):+.1f} п.п. від того, що модель")
        print(f"  бачить таблицю ЦІЛКОМ. Це не інший метод і не інша модель —")
        print(f"  лише інший обсяг видимих даних.")

    print()
    print("--- ЗА ТИПОМ ВІДПОВІДІ ---")
    print(f"  {'тип':16s} " + " ".join(f"{a:>12s}" for a in arms))
    for t in db.TYPES:
        cells = []
        for a in [x for x in ("A", "A-lite", "B", "B-lite") if x in arms]:
            sub = [r for r in arms[a].values() if r["type"] == t]
            k = sum(1 for r in sub if r["correct"])
            cells.append(f"{k}/{len(sub)} {k / max(len(sub), 1):.0%}" if sub
                         else "-")
        print(f"  {t:16s} " + " ".join(f"{c:>12s}" for c in cells))

    print()
    print("--- СКІЛЬКИ ТАБЛИЦІ БАЧИВ КОЖЕН АРМ ---")
    for a in arms:
        fr = sorted(r.get("frac_seen") or 0 for r in arms[a].values())
        print(f"  {a:8s} медіана {fr[len(fr) // 2]:7.2%}, "
              f"мін {fr[0]:7.3%}, макс {fr[-1]:6.0%}")

    print()
    print("--- ЦІНА ---")
    for a in arms:
        tok = sum(r.get("tokens") or 0 for r in arms[a].values())
        it = [r.get("iterations") or 1 for r in arms[a].values()]
        print(f"  {a:8s} {len(arms[a])} викликів, {tok:>8,} токенів, "
              f"сер. {tok / len(arms[a]):>6,.0f} ток/питання, "
              f"сер. ітерацій {sum(it) / len(it):.2f}")


if __name__ == "__main__":
    main()
