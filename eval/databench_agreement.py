"""Does the published agreement rule work outside finance? Zero API calls.

The rule confirmed on FinQA (5.26, replicated in 5.43) is: two independent
readings of the same table, accept when they agree, abstain when they do not.
FinQA is one domain and one table shape. DataBench Lite is the first place that
rule can be tested on something else -- five domains, five answer types, and
answers that are booleans, categories and lists rather than only numbers.

Lite is the right place for it and the only honest one: the rule needs BOTH
routes to have seen the same data, and on full DataBench tables the direct
route sees 0.4% of the rows while the code route sees all of them. Comparing
those two would measure the window, not the agreement.

Two choices worth stating:

  * agreement is decided by the organisers' own comparison function, the same
    one that scores correctness, applied to the two PREDICTIONS instead of a
    prediction and the truth. Writing a separate notion of "the same answer"
    would let the rule be tuned, and for lists and booleans a hand-rolled
    comparison is exactly where that would happen.
  * on accept the system returns the CODE route's answer, as on FinQA, so the
    comparison with "code alone" isolates the rule rather than the route.

Run:  python -m eval.databench_agreement --tag dev100
"""
import argparse
import collections
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


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(arm, tag):
    return {r["id"]: r
            for r in cache_io.load_rows(os.path.join(CACHE, f"{arm}_{tag}.jsonl"))[0]}


def agree(a, b, semantic):
    """Do the two routes say the same thing, by the organisers' comparison?"""
    if a is None or b is None:
        return False
    try:
        return bool(db.compare(a, b, semantic))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev100")
    ap.add_argument("--direct", default="A-lite")
    ap.add_argument("--code", default="B-lite")
    args = ap.parse_args()

    D, C = load(args.direct, args.tag), load(args.code, args.tag)
    ids = sorted(set(D) & set(C))
    if not ids:
        print("спільних питань немає")
        return

    print("=" * 76)
    print(f"ПРАВИЛО ЗГОДИ НА DATABENCH LITE — {args.tag}, нуль викликів API")
    print("=" * 76)
    print(f"  {args.direct}: {len(D)} питань, {args.code}: {len(C)}, "
          f"спільних {len(ids)}")
    if len(ids) < max(len(D), len(C)):
        print(f"  !! перетин менший за обидва кеші — один із прогонів "
              f"неповний, і це НЕ випадкова підмножина")

    rows = []
    for q in ids:
        d, c = D[q], C[q]
        ok_c = bool(c["correct"])
        rows.append({
            "id": q, "type": c["type"],
            "accept": agree(d["value"], c["value"], c["type"]),
            "code_ok": ok_c, "direct_ok": bool(d["correct"]),
        })

    n = len(rows)
    acc = [r for r in rows if r["accept"]]
    rej = [r for r in rows if not r["accept"]]
    code_ok = sum(1 for r in rows if r["code_ok"])
    acc_ok = sum(1 for r in acc if r["code_ok"])
    wrong = [r for r in rows if not r["code_ok"]]
    rej_wrong = sum(1 for r in rej if not r["code_ok"])

    print()
    print("--- ПРАВИЛО СПРАЦЮВАЛО ТАК ---")
    print(f"  ACCEPT (маршрути збіглись) : {len(acc)}")
    print(f"  ABSTAIN (розійшлись)       : {len(rej)}")

    print()
    print("--- ПЕРВИННІ МЕТРИКИ (95% CI за Вілсоном) ---")

    def line(name, k, m):
        lo, hi = wilson(k, m)
        print(f"  {name:38s} {k:3d}/{m:3d} = {k / max(m, 1):6.1%}  "
              f"[{lo:5.1%}, {hi:5.1%}]")

    line("покриття", len(acc), n)
    line("точність прийнятих (код)", acc_ok, len(acc))
    line("точність коду при повному покритті", code_ok, n)
    line("rejection recall", rej_wrong, len(wrong))
    line("rejection precision", rej_wrong, len(rej))
    gain = acc_ok / max(len(acc), 1) - code_ok / n
    print(f"  {'приріст точності прийнятих':38s} {100 * gain:+6.1f} п.п.")
    print()
    print(f"  ПОКРИТТЯ ПОРУЧ: точність прийнятих "
          f"{acc_ok / max(len(acc), 1):.1%} при покритті {len(acc) / n:.1%}")

    print()
    print("--- ЧИ ПЕРЕНОСИТЬСЯ ПРАВИЛО З ФІНАНСІВ (пороги 5.26) ---")
    rr = rej_wrong / max(len(wrong), 1)
    rp = rej_wrong / max(len(rej), 1)
    for name, val, thr in (("recall відхилення", rr, 0.40),
                           ("precision відхилення", rp, 0.60),
                           ("приріст точності прийнятих", gain, 0.10)):
        mark = "ТАК " if val >= thr else "НІ  "
        print(f"  {name:34s} >= {thr:.0%}: {mark} ({val:.1%})")

    print()
    print("--- ЗА ТИПОМ ВІДПОВІДІ ---")
    print(f"  {'тип':16s} {'n':>4s} {'ACCEPT':>8s} {'точн.прийн.':>12s} "
          f"{'точн.коду':>10s}")
    for t in db.TYPES:
        sub = [r for r in rows if r["type"] == t]
        if not sub:
            continue
        a = [r for r in sub if r["accept"]]
        print(f"  {t:16s} {len(sub):4d} {len(a):8d} "
              f"{(sum(1 for r in a if r['code_ok']) / max(len(a), 1)):11.0%} "
              f"{(sum(1 for r in sub if r['code_ok']) / len(sub)):10.0%}")

    print()
    print("--- ДЕ ПРАВИЛО БЕЗСИЛЕ ---")
    both_wrong_acc = sum(1 for r in acc
                         if not r["code_ok"] and not r["direct_ok"])
    print(f"  прийнято, але обидва маршрути хибні: {both_wrong_acc}"
          f"  ({both_wrong_acc / max(len(acc), 1):.0%} прийнятих)")
    rej_right = sum(1 for r in rej if r["code_ok"])
    print(f"  відхилено, хоч код був правий    : {rej_right}"
          f"  ({rej_right / max(len(rej), 1):.0%} відхилених)")

    out = os.path.join(HERE, f"databench_agreement_{args.tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,type,accept,code_ok,direct_ok\n")
        for r in rows:
            f.write(f"{r['id']},{r['type']},{int(r['accept'])},"
                    f"{int(r['code_ok'])},{int(r['direct_ok'])}\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
