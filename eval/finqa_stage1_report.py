"""Stage 1 report: four configurations, paired, on the same dev questions.

MODEL: openai/gpt-oss-120b (Groq). Stated because nothing here transfers to
another model without being re-measured.

RUN TYPE: exploratory, dev. The 100 questions were used by earlier analyses, so
every number is an estimate of these 100 and not of the method. The criteria
were fixed in 5.31 before the run and amended once, before any result was seen,
to compare against R1 on the SAME questions rather than against R1 on all 300.

Four answers per question, all on the same questions:

  R1        direct answer, from the existing cache, not re-run
  R2 old    code generated from labels only
  R2 new    code generated with the table's values in the prompt
  system    R1/R2 agreement, with arbitration where they disagree

Run:  python -m eval.finqa_stage1_report
"""
import argparse
import collections
import math
import os
import random

from eval import cache_io
from eval.finqa_arbiter import decide as system_route
from eval.finqa_confirm import build, cluster_of
from eval.finqa_run import is_correct

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")
TOL = 0.01
N_BOOT = 10000
SEED = 20260927


def r2_of(row):
    v = row.get("candidate")
    return row.get("answer") if v is None else v


def mcnemar_exact(b, c):
    """Two-sided exact p for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_ci(rows, fn, n_boot=N_BOOT, seed=SEED):
    cl = collections.defaultdict(list)
    for r in rows:
        cl[cluster_of(r["id"])].append(r)
    keys = list(cl)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        draw = []
        for _ in range(len(keys)):
            draw.extend(cl[keys[rng.randrange(len(keys))]])
        v = fn(draw)
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def pct(x):
    return "н/д" if x != x else f"{x:.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-tag", default="confirm300_values")
    ap.add_argument("--old-tag", default="confirm300")
    ap.add_argument("--r1-tag", default="baseline_table_confirm300")
    ap.add_argument("--arbiter-tag", default=None)
    ap.add_argument("--ids", default=os.path.join(HERE, "finqa_dev100.ids"))
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--model", default="openai/gpt-oss-120b (Groq)",
                    help="назва моделі для заголовка звіту")
    args = ap.parse_args()

    new, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.new_tag}.jsonl"))
    old, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.old_tag}.jsonl"))
    r1r, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.r1_tag}.jsonl"))
    arb_path = os.path.join(
        CACHE_DIR, f"{args.arbiter_tag or ('arbiter_' + args.new_tag)}.jsonl")
    arb, _ = cache_io.load_rows(arb_path)
    arbv = {r["id"]: r.get("arbiter") for r in arb}
    arbpick = {r["id"]: r.get("picked") for r in arb}

    with open(args.ids, encoding="utf-8") as f:
        want = [l.strip() for l in f if l.strip()]

    N, O, R = ({r["id"]: r for r in new}, {r["id"]: r for r in old},
               {r["id"]: r for r in r1r})
    ids = [q for q in want if q in N and q in O and q in R]

    rows = []
    for q in ids:
        gold = O[q].get("gold")
        v1, vo, vn = R[q].get("answer"), r2_of(O[q]), r2_of(N[q])
        route, val = system_route(v1, vn)
        if route == "AGREE":
            sysv = val
        elif route == "R1":
            sysv = v1
        elif route == "R2":
            sysv = vn
        else:
            sysv = arbv.get(q, None)
        rows.append({
            "id": q, "gold": gold, "r1": v1, "old": vo, "new": vn,
            "route": route, "sys": sysv,
            "c_r1": bool(is_correct(v1, gold)) if v1 is not None else False,
            "c_old": bool(is_correct(vo, gold)) if vo is not None else False,
            "c_new": bool(is_correct(vn, gold)) if vn is not None else False,
            "c_sys": bool(is_correct(sysv, gold)) if sysv is not None else False,
        })

    n = len(rows)
    print("=" * 78)
    print("ФІНАЛЬНІ МЕТРИКИ")
    print("=" * 78)
    print(f"МОДЕЛЬ: {args.model}")
    print(f"n = {n} питань dev, парно, ті самі id "
          f"({os.path.basename(args.ids)})")
    print("ТИП: розвідувальний, dev. Числа — оцінки цих n питань, не методу.")
    missing = [q for q in want if q not in N]
    if missing:
        print(f"УВАГА: новий R2 неповний, бракує {len(missing)} питань — "
              f"звіт на перетині")
    need_arb = sum(1 for r in rows if r["route"] == "ARBITER")
    have_arb = sum(1 for r in rows if r["route"] == "ARBITER"
                   and r["id"] in arbv)
    print(f"розбіжностей, що потребують розбору: {need_arb}, "
          f"пораховано розбором: {have_arb}")
    print()

    print("--- EA ПРИ ПОВНОМУ ПОКРИТТІ, парно ---")
    print(f"{'конфігурація':34s} {'EA':>8s} {'правильних':>12s}  95% CI")
    print("-" * 74)
    for label, k in (("R1 — пряма відповідь", "c_r1"),
                     ("R2 старий — код без значень", "c_old"),
                     ("R2 новий — код зі значеннями", "c_new"),
                     ("СИСТЕМА — згода + розбір", "c_sys")):
        v = sum(1 for r in rows if r[k]) / n
        lo, hi = boot_ci(rows, lambda rr, k=k:
                         sum(1 for x in rr if x[k]) / len(rr), args.boot)
        print(f"{label:34s} {pct(v):>8s} {sum(1 for r in rows if r[k]):>7d}/{n:<4d} "
              f"[{pct(lo)}, {pct(hi)}]")

    print()
    print("--- ПАРНІ ПОРІВНЯННЯ (Макнемар, точний) ---")
    def mc(a, b, la, lb):
        bb = sum(1 for r in rows if r[a] and not r[b])
        cc = sum(1 for r in rows if r[b] and not r[a])
        p = mcnemar_exact(bb, cc)
        d = (sum(1 for r in rows if r[a]) - sum(1 for r in rows if r[b])) / n
        print(f"  {la} проти {lb}: {d * 100:+.1f} п.п.   "
              f"розбіжних пар b={bb}, c={cc}   p={p:.4f}")
        return d, bb, cc, p
    mc("c_sys", "c_r1", "СИСТЕМА", "R1")
    mc("c_new", "c_old", "R2 новий", "R2 старий")
    mc("c_new", "c_r1", "R2 новий", "R1")

    print()
    print("--- «КОД ПРАВИЙ, ПРЯМА ХИБНА» ---")
    for label, k in (("R2 старий", "c_old"), ("R2 новий", "c_new")):
        a = sum(1 for r in rows if r[k] and not r["c_r1"])
        b = sum(1 for r in rows if r["c_r1"] and not r[k])
        print(f"  {label:12s} код правий/пряма хибна: {a:3d}   "
              f"пряма права/код хибний: {b:3d}")

    print()
    print("--- PRECISION/RECALL ЗГОДИ НА НОВОМУ R2 (умова 5.31.5) ---")
    for label, k, v in (("старий R2", "c_old", "old"), ("новий R2", "c_new", "new")):
        acc = [r for r in rows if r[v] is not None and r["r1"] is not None
               and abs(r["r1"] - r[v]) <= TOL * max(abs(r[v]), 1.0)]
        rej = [r for r in rows if r not in acc and r[v] is not None]
        wrong = [r for r in rows if r[v] is not None and not r[k]]
        rec = (sum(1 for r in wrong if r in rej) / len(wrong)) if wrong else float("nan")
        prec = (sum(1 for r in rej if not r[k]) / len(rej)) if rej else float("nan")
        cov = len(acc) / n
        print(f"  {label:10s} recall {pct(rec)}   precision {pct(prec)}   "
              f"покриття {pct(cov)}   (прийнято {len(acc)}, відхилено {len(rej)})")

    if have_arb:
        print()
        print("--- ЩО ОБРАВ РОЗБІР ---")
        picks = collections.Counter(arbpick.get(r["id"])
                                    for r in rows if r["route"] == "ARBITER")
        for k, v in picks.most_common():
            print(f"  {k}: {v}")
        ok = sum(1 for r in rows if r["route"] == "ARBITER" and r["c_sys"])
        tot = sum(1 for r in rows if r["route"] == "ARBITER")
        best = sum(1 for r in rows if r["route"] == "ARBITER"
                   and (r["c_r1"] or r["c_new"]))
        print(f"  правильних після розбору: {ok}/{tot}")
        print(f"  стеля (хоч один маршрут правий): {best}/{tot}")

    out = os.path.join(HERE, "finqa_stage1_gptoss120b.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,gold,r1,r2_old,r2_new,route,system,"
                "c_r1,c_old,c_new,c_sys\n")
        for r in rows:
            f.write(f"{r['id']},{r['gold']},{r['r1']},{r['old']},{r['new']},"
                    f"{r['route']},{r['sys']},{int(r['c_r1'])},{int(r['c_old'])},"
                    f"{int(r['c_new'])},{int(r['c_sys'])}\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
