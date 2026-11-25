"""Honest comparison against the direct route, and a taxonomy of ITS errors.

SECONDARY POST-HOC ANALYSIS OF THE DEV SET, not pre-registered. 5.26 fixed
the delivered answer as R2's and measured the agreement rule against that.
Step 0 then showed R2 is the weaker answerer (59.3% against 72.3% at full
coverage), which makes one question unavoidable: how much of the confirmed
result survives when the answer is taken from the STRONGER route instead.

Stated plainly, because it is the thing a reader will ask: agreement
reliably detects errors OF THE CODE ROUTE. Against the direct answer the
effect is smaller, and this module measures how much smaller.

Zero API calls.

Run:  python -m eval.finqa_vs_r1
"""
import argparse
import collections
import os
import random

from eval import cache_io
from eval.finqa_ceilings import classify
from eval.finqa_confirm import build, cluster_of
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")
N_BOOT = 10000
SEED = 20260926


def sel_metrics(rows, who):
    """Selective-prediction metrics when the delivered answer is `who`."""
    ok = "correct" if who == "r2" else "r1_correct"
    val = "r2" if who == "r2" else "r1"
    n = len(rows)
    acc_rows = [r for r in rows if r["verdict"] == "ACCEPT"]
    full = sum(1 for r in rows if r[ok]) / n if n else 0.0
    among = (sum(1 for r in acc_rows if r[ok]) / len(acc_rows)
             if acc_rows else float("nan"))
    with_v = [r for r in rows if r[val] is not None]
    wrong = [r for r in with_v if not r[ok]]
    rej_v = [r for r in rows if r["verdict"] == "ABSTAIN" and r[val] is not None]
    rec = (sum(1 for r in wrong if r["verdict"] == "ABSTAIN") / len(wrong)
           if wrong else float("nan"))
    prec = (sum(1 for r in rej_v if not r[ok]) / len(rej_v)
            if rej_v else float("nan"))
    # The cost of abstaining: correct answers thrown away.
    lost = sum(1 for r in rows if r["verdict"] == "ABSTAIN" and r[ok])
    return {"full": full, "among": among, "gain": among - full,
            "coverage": len(acc_rows) / n if n else 0.0,
            "recall": rec, "precision": prec,
            "lost_correct": lost, "n_wrong": len(wrong)}


def boot(rows, who, key, n_boot=N_BOOT, seed=SEED):
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
        v = sel_metrics(draw, who)[key]
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
    ap.add_argument("--tag", default="confirm300")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--show-other", type=int, default=20,
                    help="скільки прикладів кошика OTHER виписати для розбору")
    args = ap.parse_args()

    code, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.tag}.jsonl"))
    dirr, _ = cache_io.load_rows(
        os.path.join(CACHE_DIR, f"baseline_table_{args.tag}.jsonl"))
    rows = build(code, dirr)
    by_id = {r["id"]: r for r in code}
    recs = {r["id"]: r for r in fq.load_split("test")}
    for r in rows:
        r["gold_program"] = by_id[r["id"]].get("gold_program")
        r["question"] = by_id[r["id"]].get("question")

    print("=" * 78)
    print("ЧЕСНЕ ПОРІВНЯННЯ З ПРЯМОЮ ВІДПОВІДДЮ (вторинний пост-хок аналіз dev)")
    print("=" * 78)
    print("НЕ передреєстровано. Первинний вердикт 5.26 лишається як є;")
    print("це відповідь на питання, яке Крок 0 зробив неминучим.")
    print()

    m2, m1 = sel_metrics(rows, "r2"), sel_metrics(rows, "r1")
    print(f"{'':38s} {'R2 (код)':>12s} {'R1 (пряма)':>12s}")
    print("-" * 66)
    print(f"{'точність при повному покритті':38s} "
          f"{pct(m2['full']):>12s} {pct(m1['full']):>12s}")
    print(f"{'точність прийнятих':38s} "
          f"{pct(m2['among']):>12s} {pct(m1['among']):>12s}")
    print(f"{'приріст від відмови':38s} "
          f"{m2['gain'] * 100:11.1f}п {m1['gain'] * 100:11.1f}п")
    print(f"{'покриття (однакове, правило те саме)':38s} "
          f"{pct(m2['coverage']):>12s} {pct(m1['coverage']):>12s}")
    print(f"{'rejection recall':38s} "
          f"{pct(m2['recall']):>12s} {pct(m1['recall']):>12s}")
    print(f"{'rejection precision':38s} "
          f"{pct(m2['precision']):>12s} {pct(m1['precision']):>12s}")
    print(f"{'правильних відкинуто відмовою':38s} "
          f"{m2['lost_correct']:12d} {m1['lost_correct']:12d}")
    print(f"{'хибних усього':38s} "
          f"{m2['n_wrong']:12d} {m1['n_wrong']:12d}")
    print()

    print("--- кластерні 95% CI для R1 ---")
    for label, k in (("точність при повному покритті", "full"),
                     ("точність прийнятих", "among"),
                     ("приріст", "gain"),
                     ("rejection recall", "recall"),
                     ("rejection precision", "precision")):
        lo, hi = boot(rows, "r1", k, args.boot)
        pt = m1[k]
        if k == "gain":
            print(f"  {label:34s} {pt * 100:6.1f}п  [{lo * 100:6.1f}п, {hi * 100:6.1f}п]")
        else:
            print(f"  {label:34s} {pct(pt):>6s}  [{pct(lo):>7s}, {pct(hi):>7s}]")

    print()
    print("ФОРМУЛЮВАННЯ ДЛЯ РОБОТИ:")
    print("  Згода маршрутів надійно виявляє помилки КОДОВОГО маршруту")
    print(f"  (recall {pct(m2['recall'])}, precision {pct(m2['precision'])}).")
    print(f"  Відносно прямої відповіді ефект менший: recall {pct(m1['recall'])},")
    print(f"  precision {pct(m1['precision'])}, приріст {m1['gain'] * 100:.1f} п.п.")
    print(f"  проти {m2['gain'] * 100:.1f} п.п. Відмова при цьому викидає "
          f"{m1['lost_correct']} правильних")
    print(f"  відповідей R1 проти {m2['lost_correct']} правильних R2.")

    # ---- taxonomy of R1 errors ----
    print()
    print("=" * 78)
    print("ТАКСОНОМІЯ ПОМИЛОК ПРЯМОГО МАРШРУТУ")
    print("=" * 78)
    wrong1 = [r for r in rows if not r["r1_correct"]]
    print(f"хибних відповідей R1: {len(wrong1)} з {len(rows)}")
    lab1 = {}
    for r in wrong1:
        rr = dict(r)
        rr["r2"] = r["r1"]              # classify() reads the r2 field
        lab1[r["id"]] = classify(rr, recs.get(r["id"]))
    c1 = collections.Counter(lab1.values())
    wrong2 = [r for r in rows if not r["correct"]]
    lab2 = {r["id"]: classify(r, recs.get(r["id"])) for r in wrong2}
    c2 = collections.Counter(lab2.values())

    print()
    print(f"{'тип':16s} {'R1':>10s} {'R2':>10s}")
    print("-" * 40)
    for lab in sorted(set(c1) | set(c2), key=lambda k: -c1.get(k, 0)):
        a, b = c1.get(lab, 0), c2.get(lab, 0)
        print(f"{lab:16s} {a:4d} {a / max(len(wrong1), 1):5.1%} "
              f"{b:4d} {b / max(len(wrong2), 1):5.1%}")

    print()
    print(f"--- КОШИК OTHER У R1: {c1.get('OTHER', 0)} шт, "
          f"перші {args.show_other} для ручного розбору ---")
    others = [r for r in wrong1 if lab1[r["id"]] == "OTHER"]
    for r in others[:args.show_other]:
        rec = recs.get(r["id"])
        shape = by_id[r["id"]].get("table_shape")
        print()
        print(f"  {r['id']}  таблиця {shape}  gold_program={r['gold_program']}")
        print(f"    питання : {str(r['question'])[:110]}")
        print(f"    еталон  : {r['gold']}   R1={r['r1']}   R2={r['r2']} "
              f"(R2 правильний: {r['correct']})")
        if r["gold"] not in (None, 0) and r["r1"] is not None:
            try:
                print(f"    відношення R1/еталон = {r['r1'] / r['gold']:.4f}")
            except ZeroDivisionError:
                pass

    out = os.path.join(HERE, f"finqa_r1_errors_{args.tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,gold,r1,r2,r1_error_type,r2_error_type,verdict,question\n")
        for r in rows:
            q = str(r["question"]).replace(",", " ").replace("\n", " ")
            f.write(f"{r['id']},{r['gold']},{r['r1']},{r['r2']},"
                    f"{lab1.get(r['id'], '')},{lab2.get(r['id'], '')},"
                    f"{r['verdict']},{q}\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
