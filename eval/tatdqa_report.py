"""TAT-DQA end-to-end: what the PDF step costs, paired on the same questions.

Both arms run the same code and the same model over the same 40 questions. The
only difference is where the table comes from, so the gap between them is
attributable -- to extraction AND to picking the right table together, which
this design cannot separate and does not claim to.

`gold` is an UPPER BOUND, not a measurement of the method: it is handed the
row group that holds the cells the gold answer used (5.33.2).

Run:  python -m eval.tatdqa_report
"""
import argparse
import collections
import math
import os
import random
import zipfile

from eval import cache_io, tatdqa_loader as T

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(arm, model):
    r, _ = cache_io.load_rows(os.path.join(CACHE, f"tatdqa_{arm}_{model}.jsonl"))
    return {x["id"]: x for x in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-flash-lite-latest")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    zf = zipfile.ZipFile(T.DOCS_ZIP)
    sel = T.select(zf=zf)
    random.Random(args.seed).shuffle(sel)
    sample = sel[:args.n]
    ids = [q["uid"] for q in sample]

    gold, pdf = load("gold", args.model), load("pdf", args.model)

    print("=" * 78)
    print("TAT-DQA, НАСКРІЗНИЙ ПРОГІН")
    print("=" * 78)
    print(f"МОДЕЛЬ: {args.model}")
    print(f"n = {len(ids)} питань, парно, ті самі id (seed {args.seed})")
    print("ТИП: розвідувальне, інший датасет (5.33)")
    print()
    print("  gold — ВЕРХНЯ МЕЖА: таблиця з розмітки + вибір групи з підказкою")
    print("         еталона; НЕ вимір методу")
    print("  pdf  — реальний режим: наш парсер, потрібну таблицю шукає система")
    print()

    # main metric: absent row or absent answer counts as wrong
    def ea(d):
        return sum(1 for q in ids if d.get(q, {}).get("correct")) / len(ids)

    both = [q for q in ids if q in gold and q in pdf]
    print("--- ОСНОВНЕ: EA на всіх %d (відсутня таблиця/відповідь = хибна) ---"
          % len(ids))
    for lab, d in (("gold (верхня межа)", gold), ("pdf (реальний режим)", pdf)):
        k = sum(1 for q in ids if d.get(q, {}).get("correct"))
        miss = sum(1 for q in ids if q not in d)
        print(f"  {lab:24s} {k:2d}/{len(ids)} = {k / len(ids):5.1%}"
              f"   (рядок відсутній: {miss})")
    gap = ea(gold) - ea(pdf)
    print()
    print(f"  РІЗНИЦЯ gold − pdf = {gap * 100:+.1f} п.п.")
    print("  = втрати через витягування з PDF РАЗОМ із вибором таблиці;")
    print("    розділити ці два внески цим планом не можна")

    b = sum(1 for q in both if gold[q].get("correct") and not pdf[q].get("correct"))
    c = sum(1 for q in both if pdf[q].get("correct") and not gold[q].get("correct"))
    print(f"  Макнемар на перетині ({len(both)}): b={b}, c={c}, "
          f"p={mcnemar_exact(b, c):.4f}")

    print()
    print("--- ДОДАТКОВЕ: EA на перетині, де таблиця є в обох ---")
    for lab, d in (("gold", gold), ("pdf", pdf)):
        k = sum(1 for q in both if d[q].get("correct"))
        print(f"  {lab:24s} {k:2d}/{len(both)} = {k / len(both):5.1%}")

    print()
    print("--- ЯКІСТЬ ВИТЯГУВАННЯ ---")
    for lab, d in (("gold", gold), ("pdf", pdf)):
        cov = sum(1 for q in ids if d.get(q, {}).get("covers_facts"))
        nt = [d[q].get("n_tables", 0) for q in ids if q in d]
        print(f"  {lab:24s} містять потрібні клітинки {cov:2d}/{len(ids)} "
              f"= {cov / len(ids):5.0%}   таблиць на сторінку "
              f"{sum(nt) / max(len(nt), 1):.1f}")

    print()
    print("--- СТАТУСИ Й ВАРТІСТЬ ---")
    for lab, d in (("gold", gold), ("pdf", pdf)):
        rs = [d[q] for q in ids if q in d]
        st = collections.Counter(x["status"] for x in rs)
        print(f"  {lab:6s} {dict(st)}   "
              f"винятків {sum(1 for x in rs if x.get('error'))}   "
              f"токенів/питання {sum(x.get('tokens', 0) for x in rs) / max(len(rs), 1):.0f}")

    # where pdf loses: no table, table without the cells, or a wrong answer
    print()
    print("--- ДЕ САМЕ ГУБИТЬСЯ ГІЛКА pdf ---")
    lost = [q for q in ids if gold.get(q, {}).get("correct")
            and not pdf.get(q, {}).get("correct")]
    nocov = sum(1 for q in lost if not pdf.get(q, {}).get("covers_facts"))
    print(f"  правильних у gold і хибних у pdf: {len(lost)}")
    print(f"    з них витягнута таблиця НЕ містила потрібних клітинок: {nocov}")
    print(f"    клітинки були, але відповідь хибна: {len(lost) - nocov}")

    out = os.path.join(HERE, f"tatdqa_report_{args.model}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,gold_correct,pdf_correct,gold_covers,pdf_covers,"
                "gold_status,pdf_status,answer_gold,answer_pdf,gold_value\n")
        for q in ids:
            g, p = gold.get(q, {}), pdf.get(q, {})
            f.write(f"{q},{int(bool(g.get('correct')))},{int(bool(p.get('correct')))},"
                    f"{int(bool(g.get('covers_facts')))},{int(bool(p.get('covers_facts')))},"
                    f"{g.get('status')},{p.get('status')},"
                    f"{g.get('candidate')},{p.get('candidate')},{g.get('gold')}\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
