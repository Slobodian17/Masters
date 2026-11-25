"""Scale and sign across the two routes: accept the disagreement, or resolve it?

`finqa_confirm.decide` treats a power-of-ten gap between the routes as an
ACCEPT with a UNIT flag, and the accepted value is the code's. So when the
direct route has the right magnitude and the code does not, the system accepts
the wrong one. 17 of the 88 wrong answers have a PERFECT measured cell set and
fail only on scale, sign or operand order (5.36), which is exactly this class.

This module asks whether the gap can be resolved by rule instead of accepted:
when the only difference between the routes is a power of ten or a sign, which
route is right, and how often? Nothing here needs a model -- both routes are
already in the cache. The gold answer is used to score the rule, never inside it.

Zero API calls.

Run:  python -m eval.finqa_scale_rule --tag confirm300
"""
import argparse
import collections
import csv
import os
from math import comb

from eval import cache_io
from eval.finqa_confirm import TOL, build, is_correct

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")

POWERS = (1e1, 1e2, 1e3, 1e4, 1e6, 1e9)


def close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(abs(b), 1.0)


def relation(r1, r2):
    """How the two route answers differ. Oracle-free.

    EXACT      the same number within tolerance
    SIGN       the same magnitude, opposite sign
    SCALE      the same digits, a power of ten apart
    SIGNSCALE  both at once
    DIFF       genuinely different numbers
    """
    if r1 is None or r2 is None:
        return "MISSING"
    if close(r1, r2):
        return "EXACT"
    if close(-r1, r2):
        return "SIGN"
    for f in POWERS:
        if close(r1 * f, r2) or close(r1 / f, r2):
            return "SCALE"
    for f in POWERS:
        if close(-r1 * f, r2) or close(-r1 / f, r2):
            return "SIGNSCALE"
    return "DIFF"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    args = ap.parse_args()

    code_rows = cache_io.load_rows(os.path.join(CACHE, f"{args.tag}.jsonl"))[0]
    dir_rows = cache_io.load_rows(
        os.path.join(CACHE, f"baseline_table_{args.tag}.jsonl"))[0]
    rows = build(code_rows, dir_rows)

    # The measured cell set, for the intersection with the 17 "perfect cells".
    loc = {}
    p = os.path.join(HERE, f"finqa_perturb_{args.tag}.csv")
    if os.path.exists(p):
        for d in csv.DictReader(open(p, encoding="utf-8")):
            loc[d["id"]] = (d["loc"], int(d["dep_extra"] or 0))

    for r in rows:
        r["rel"] = relation(r["r1"], r["r2"])

    print("=" * 78)
    print(f"МАСШТАБ І ЗНАК МІЖ МАРШРУТАМИ — {args.tag}, нуль викликів API")
    print("=" * 78)
    print()
    print("--- ЯК РОЗХОДЯТЬСЯ МАРШРУТИ, і хто з них правий ---")
    print(f"  {'відношення':12s} {'питань':>7s} {'лише R1':>9s} {'лише R2':>9s}"
          f" {'обидва':>8s} {'жоден':>7s}")
    order = ["EXACT", "SIGN", "SCALE", "SIGNSCALE", "DIFF", "MISSING"]
    cnt = collections.Counter(r["rel"] for r in rows)
    for k in order:
        sub = [r for r in rows if r["rel"] == k]
        if not sub:
            continue
        only1 = sum(1 for r in sub if r["r1_correct"] and not r["correct"])
        only2 = sum(1 for r in sub if r["correct"] and not r["r1_correct"])
        both = sum(1 for r in sub if r["correct"] and r["r1_correct"])
        none = sum(1 for r in sub if not r["correct"] and not r["r1_correct"])
        print(f"  {k:12s} {len(sub):7d} {only1:9d} {only2:9d} {both:8d} {none:7d}")

    fix = [r for r in rows if r["rel"] in ("SIGN", "SCALE", "SIGNSCALE")]
    print()
    print(f"--- КЛАС, ЯКИЙ ПРАВИЛО МОЖЕ ВИРІШИТИ: {len(fix)} питань ---")
    w1 = sum(1 for r in fix if r["r1_correct"])
    w2 = sum(1 for r in fix if r["correct"])
    print(f"  правий прямий маршрут R1 : {w1}")
    print(f"  правий кодовий маршрут R2: {w2}")
    print(f"  жоден                     : "
          f"{sum(1 for r in fix if not r['correct'] and not r['r1_correct'])}")
    if loc:
        perfect = [r for r in fix
                   if loc.get(r["id"], ("", 0))[0] == "HIT_ALL"
                   and loc.get(r["id"], ("", 0))[1] == 0 and not r["correct"]]
        print(f"  із 17 'клітинки ідеальні' сюди попало: {len(perfect)}")

    print()
    print("--- ПРАВИЛО: коли різниця лише в масштабі/знаку, брати R1 ---")
    print("  (повне покриття, без відмов; точність по ВСІХ питаннях)")

    def acc(pick):
        return sum(1 for r in rows if pick(r))

    base = acc(lambda r: r["correct"])
    ruled = acc(lambda r: r["r1_correct"] if r["rel"] in
                ("SIGN", "SCALE", "SIGNSCALE") else r["correct"])
    b = sum(1 for r in rows
            if r["rel"] in ("SIGN", "SCALE", "SIGNSCALE")
            and r["r1_correct"] and not r["correct"])
    c = sum(1 for r in rows
            if r["rel"] in ("SIGN", "SCALE", "SIGNSCALE")
            and r["correct"] and not r["r1_correct"])
    n = len(rows)
    print(f"  кодовий маршрут як є      : {base}/{n} = {base / n:.1%}")
    print(f"  з правилом масштабу/знаку : {ruled}/{n} = {ruled / n:.1%}"
          f"   ({(ruled - base) / n:+.1%} п.п.)")
    print(f"  виправлено {b}, зламано {c}, Макнемар p = {mcnemar(b, c):.4f}")

    print()
    print("--- ДЛЯ ОРІЄНТИРА: прямий маршрут сам по собі ---")
    r1acc = acc(lambda r: r["r1_correct"])
    print(f"  R1 як є                   : {r1acc}/{n} = {r1acc / n:.1%}")

    print()
    print("--- ЩО РОБИТЬ ЧИННЕ ПРАВИЛО 5.26 З ЦИМ КЛАСОМ ---")
    acc_unit = sum(1 for r in fix if r["correct"])
    print(f"  {len(fix)} питань класу приймаються з UNIT-прапорцем, "
          f"видається R2")
    print(f"  з них R2 правий у {acc_unit}, тобто {len(fix) - acc_unit} "
          f"хибних приймаються")

    out = os.path.join(HERE, f"finqa_scale_rule_{args.tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,relation,gold,r1,r2,r1_correct,r2_correct,loc,dep_extra\n")
        for r in rows:
            l, e = loc.get(r["id"], ("", ""))
            f.write(f"{r['id']},{r['rel']},{r['gold']},{r['r1']},{r['r2']},"
                    f"{int(r['r1_correct'])},{int(r['correct'])},{l},{e}\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
