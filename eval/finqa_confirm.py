"""Confirmatory analysis of the pre-registered run (ARCHITECTURE.md 5.26).

Nothing here is chosen after seeing a number. The decision rule, the metrics,
the thresholds and the way a grey result is presented were fixed in 5.26
before the run started; this module only evaluates them.

Decision rule (5.26.2), two routes of ONE model:

    R1  direct answer, table in the prompt
    R2  generated pandas, executed in the sandbox

    |R1 - R2| <= 1% * max(|R2|, 1)      -> ACCEPT
    R1 and R2 differ by exactly 10^k    -> ACCEPT, flag UNIT
    otherwise                           -> ABSTAIN
    either route gave no number         -> ABSTAIN

The delivered answer is R2's. A UNIT case is scored on R2 with no leniency.

Confidence intervals are CLUSTER bootstrap over unique tables, not over
questions: 2.05 questions share a table (5.16.5), and resampling questions
would understate every interval.

Run:  python -m eval.finqa_confirm
"""
import argparse
import collections
import json
import os
import random

from eval import cache_io
from eval.finqa_run import is_correct

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")

TOL = 0.01
N_BOOT = 10000
SEED = 20260926

# 5.26.4, fixed before the run
CONF_RECALL_POINT, CONF_RECALL_LO = 0.40, 0.15
CONF_PREC_POINT, CONF_PREC_LO = 0.60, 0.40
CONF_GAIN_POINT = 0.10
REFUTE_RECALL_CONTAINS = 0.10
REFUTE_GAIN_MAX = 0.05


# ---- decision rule ----

def scale_ratio(a, b):
    """True when a and b differ by exactly a power of ten (within TOL)."""
    for f in (1e1, 1e2, 1e3, 1e6, 1e9):
        for q in (a * f, a / f):
            if abs(q - b) <= TOL * max(abs(b), 1.0):
                return True
    return False


def decide(r1, r2):
    """('ACCEPT'|'ABSTAIN', unit_flag, distance).

    `distance` is the relative gap the rule thresholds on, and doubles as the
    confidence score for the risk-coverage curve: it is the rule's own
    continuous quantity, not a separate construct chosen afterwards.
    """
    if r1 is None or r2 is None:
        return "ABSTAIN", False, float("inf")
    d = abs(r1 - r2) / max(abs(r2), 1.0)
    if d <= TOL:
        return "ACCEPT", False, d
    if scale_ratio(r1, r2):
        return "ACCEPT", True, d
    return "ABSTAIN", False, d


def cluster_of(qid):
    """Unique table: everything before the trailing question index."""
    return qid.rsplit("-", 1)[0]


# ---- metrics on one sample ----

def metrics(rows):
    n = len(rows)
    acc_rows = [r for r in rows if r["verdict"] == "ACCEPT"]
    rej_rows = [r for r in rows if r["verdict"] == "ABSTAIN"]

    coverage = len(acc_rows) / n if n else 0.0
    acc_among = (sum(1 for r in acc_rows if r["correct"]) / len(acc_rows)
                 if acc_rows else float("nan"))

    # Accuracy if every question were answered from R2: the full-coverage
    # comparison condition 3 is stated against. No number counts as wrong,
    # because a system that delivers nothing delivered no correct answer.
    full = sum(1 for r in rows if r["correct"]) / n if n else 0.0

    # Literal reading of 5.26.3: a rejection can only be "right" if there was
    # a wrong candidate to reject. Questions with no candidate at all are
    # counted separately rather than folded in either direction.
    with_cand = [r for r in rows if r["r2"] is not None]
    wrong_cand = [r for r in with_cand if not r["correct"]]
    rej_with_cand = [r for r in rej_rows if r["r2"] is not None]

    rec = (sum(1 for r in wrong_cand if r["verdict"] == "ABSTAIN")
           / len(wrong_cand)) if wrong_cand else float("nan")
    prec = (sum(1 for r in rej_with_cand if not r["correct"])
            / len(rej_with_cand)) if rej_with_cand else float("nan")

    return {
        "n": n,
        "coverage": coverage,
        "acc_accepted": acc_among,
        "acc_full": full,
        "gain": acc_among - full if acc_among == acc_among else float("nan"),
        "rej_recall": rec,
        "rej_precision": prec,
        "aurc": aurc(rows),
        "n_accept": len(acc_rows),
        "n_reject": len(rej_rows),
        "n_unit": sum(1 for r in acc_rows if r["unit"]),
        "n_no_candidate": sum(1 for r in rows if r["r2"] is None),
        "n_rej_no_candidate": sum(1 for r in rej_rows if r["r2"] is None),
        "n_wrong_candidates": len(wrong_cand),
    }


def aurc(rows):
    """Area under the risk-coverage curve, ordered by the rule's own distance.

    Risk at coverage k/n is the error rate among the k questions whose routes
    agree most closely. Lower is better; 0.5 is what a useless ordering gives
    on a half-wrong set.
    """
    order = sorted(rows, key=lambda r: r["distance"])
    errs = 0
    areas = []
    for i, r in enumerate(order, start=1):
        if not r["correct"]:
            errs += 1
        areas.append(errs / i)
    return sum(areas) / len(areas) if areas else float("nan")


# ---- cluster bootstrap ----

def boot_ci(rows, key, n_boot=N_BOOT, seed=SEED):
    clusters = collections.defaultdict(list)
    for r in rows:
        clusters[cluster_of(r["id"])].append(r)
    keys = list(clusters)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        draw = []
        for _ in range(len(keys)):
            draw.extend(clusters[keys[rng.randrange(len(keys))]])
        v = metrics(draw)[key]
        if v == v:                      # drop NaN draws
            vals.append(v)
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return lo, hi


def boot_ci_diff(rows, n_boot=N_BOOT, seed=SEED):
    """CI for (accuracy among accepted - accuracy at full coverage)."""
    clusters = collections.defaultdict(list)
    for r in rows:
        clusters[cluster_of(r["id"])].append(r)
    keys = list(clusters)
    rng = random.Random(seed + 1)
    vals = []
    for _ in range(n_boot):
        draw = []
        for _ in range(len(keys)):
            draw.extend(clusters[keys[rng.randrange(len(keys))]])
        m = metrics(draw)
        if m["gain"] == m["gain"]:
            vals.append(m["gain"])
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return lo, hi


def needed_n(n, point, lo, threshold):
    """n at which the CI bound would clear `threshold`, holding the effect.

    Half-width shrinks as 1/sqrt(n), so n scales with the square of the ratio
    of the observed half-width to the one that would just reach the threshold.
    """
    half = point - lo
    need = point - threshold
    if need <= 0 or half <= 0:
        return None
    return int(round(n * (half / need) ** 2))


# ---- assembly ----

def build(code_rows, dir_rows):
    d1 = {r["id"]: r for r in dir_rows}
    out = []
    for r in code_rows:
        q = r["id"]
        gold = r.get("gold")
        r2 = r.get("candidate")
        if r2 is None:
            r2 = r.get("answer")
        r1 = d1[q].get("answer") if q in d1 else None
        verdict, unit, dist = decide(r1, r2)
        out.append({
            "id": q, "gold": gold, "r1": r1, "r2": r2,
            "verdict": verdict, "unit": unit, "distance": dist,
            "correct": bool(is_correct(r2, gold)) if r2 is not None else False,
            "r1_correct": bool(is_correct(r1, gold)) if r1 is not None else False,
            "confidence": r.get("confidence"),
            "n_ops": r.get("n_ops"),
        })
    return out


def pct(x):
    return "н/д" if x != x else f"{x:.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    # Sensitivity only. 5.26.1 defined the population as the table subset with
    # no numeric filter, so the registered analysis keeps every question; this
    # switch says what the same numbers look like without the ones whose gold
    # answer is "yes"/"no" and which a numeric pipeline cannot answer at all.
    ap.add_argument("--drop-nogold", action="store_true",
                    help="ВІДХИЛЕННЯ від 5.26.1: виключити питання без "
                         "числового еталона (аналіз чутливості)")
    # The size was hardcoded to 300 for confirm300. The check is what matters,
    # not the number: both routes must have answered the SAME questions, and
    # the set must be the complete pre-registered one, because the criteria are
    # computed on pairs and a partial run would quietly change the denominator.
    ap.add_argument("--expect", type=int, default=0,
                    help="скільки питань мусить бути; 0 = взяти з кеша коду")
    args = ap.parse_args()

    code, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.tag}.jsonl"))
    dirr, _ = cache_io.load_rows(
        os.path.join(CACHE_DIR, f"baseline_table_{args.tag}.jsonl"))
    expect = args.expect or len(code)
    assert len(code) == len(dirr) == expect, (len(code), len(dirr), expect)
    assert {r["id"] for r in code} == {r["id"] for r in dirr},         "маршрути відповідали на РІЗНІ питання — пара недійсна"
    print(f"пара повна: {len(code)} питань, обидва маршрути на тих самих id")


    rows = build(code, dirr)
    n_nogold = sum(1 for r in rows if r["gold"] is None)
    if args.drop_nogold:
        rows = [r for r in rows if r["gold"] is not None]
        print("!!! АНАЛІЗ ЧУТЛИВОСТІ, НЕ ЗАРЕЄСТРОВАНИЙ: виключено "
              f"{n_nogold} питань без числового еталона")
        print()
    m = metrics(rows)
    clusters = len({cluster_of(r["id"]) for r in rows})

    print("=" * 78)
    print("ПІДТВЕРДЖУВАЛЬНИЙ АНАЛІЗ за передреєстрацією 5.26")
    print("=" * 78)
    print(f"питань: {m['n']}   унікальних таблиць (кластерів): {clusters}   "
          f"{m['n'] / clusters:.2f} питання на таблицю")
    print(f"bootstrap: {args.boot} ресемплів КЛАСТЕРІВ, не питань")
    print()

    print("--- правило рішення спрацювало так ---")
    print(f"  ACCEPT  : {m['n_accept']}   з них UNIT (різниця 10^k): {m['n_unit']}")
    print(f"  ABSTAIN : {m['n_reject']}")
    print(f"  без числа від R2: {m['n_no_candidate']}   "
          f"з них у відмовах: {m['n_rej_no_candidate']}")
    print(f"  хибних кандидатів усього: {m['n_wrong_candidates']}")
    print()

    ci = {}
    for k in ("coverage", "acc_accepted", "acc_full", "rej_recall",
              "rej_precision", "aurc"):
        ci[k] = boot_ci(rows, k, args.boot)
    ci["gain"] = boot_ci_diff(rows, args.boot)

    print("--- ПЕРВИННІ МЕТРИКИ (усі, включно з провальними) ---")
    print(f"{'метрика':34s} {'оцінка':>9s}  {'95% CI (кластерний)':>26s}")
    print("-" * 74)
    order = [("покриття", "coverage"),
             ("точність прийнятих", "acc_accepted"),
             ("точність при повному покритті", "acc_full"),
             ("приріст точності", "gain"),
             ("rejection recall", "rej_recall"),
             ("rejection precision", "rej_precision")]
    for label, k in order:
        lo, hi = ci[k]
        print(f"{label:34s} {pct(m[k]):>9s}  [{pct(lo):>9s}, {pct(hi):>9s}]")
    lo, hi = ci["aurc"]
    print(f"{'AURC (менше — краще)':34s} {m['aurc']:9.3f}  "
          f"[{lo:9.3f}, {hi:9.3f}]")
    print()
    print(f"  ПОКРИТТЯ ПОРУЧ: точність прийнятих {pct(m['acc_accepted'])} "
          f"при покритті {pct(m['coverage'])} "
          f"({m['n_accept']} з {m['n']} питань)")
    print()

    # ---- verdict ----
    c1 = m["rej_recall"] >= CONF_RECALL_POINT and ci["rej_recall"][0] > CONF_RECALL_LO
    c2 = m["rej_precision"] >= CONF_PREC_POINT and ci["rej_precision"][0] > CONF_PREC_LO
    c3 = m["gain"] >= CONF_GAIN_POINT and ci["gain"][0] > 0
    r1_ = ci["rej_recall"][0] <= REFUTE_RECALL_CONTAINS <= ci["rej_recall"][1]
    r2_ = m["gain"] < REFUTE_GAIN_MAX

    print("--- УМОВИ ПІДТВЕРДЖЕННЯ (5.26.4, зафіксовані до прогону) ---")
    print(f"  1. recall >= 40% і нижня межа CI > 15%        : "
          f"{'ТАК' if c1 else 'НІ'}  "
          f"({pct(m['rej_recall'])}, нижня {pct(ci['rej_recall'][0])})")
    print(f"  2. precision >= 60% і нижня межа CI > 40%     : "
          f"{'ТАК' if c2 else 'НІ'}  "
          f"({pct(m['rej_precision'])}, нижня {pct(ci['rej_precision'][0])})")
    print(f"  3. приріст точності >= 10 п.п. і нижня CI > 0 : "
          f"{'ТАК' if c3 else 'НІ'}  "
          f"({m['gain'] * 100:.1f} п.п., нижня {ci['gain'][0] * 100:.1f} п.п.)")
    print()
    print("--- УМОВИ СПРОСТУВАННЯ (досить однієї) ---")
    print(f"  1. CI для recall містить 10%                  : "
          f"{'ТАК' if r1_ else 'НІ'}  "
          f"(CI [{pct(ci['rej_recall'][0])}, {pct(ci['rej_recall'][1])}])")
    print(f"  2. приріст точності < 5 п.п.                  : "
          f"{'ТАК' if r2_ else 'НІ'}  ({m['gain'] * 100:.1f} п.п.)")
    print()

    if c1 and c2 and c3:
        verdict = "ПІДТВЕРДЖЕНО"
    elif r1_ or r2_:
        verdict = "СПРОСТОВАНО"
    else:
        verdict = "СІРА ЗОНА — не підтверджено й не спростовано"

    print("=" * 78)
    print(f"ВЕРДИКТ: {verdict}")
    print("=" * 78)

    if verdict.startswith("СІРА"):
        print()
        print("Якої потужності бракує (5.26.4, пункт 3-4):")
        for label, k, thr in (("recall", "rej_recall", CONF_RECALL_LO),
                              ("precision", "rej_precision", CONF_PREC_LO),
                              ("приріст", "gain", 0.0)):
            point, lo = m[k], ci[k][0]
            if point != point:
                continue
            need = needed_n(m["n"], point, lo, thr)
            if lo > thr:
                print(f"  {label}: нижня межа {pct(lo)} вже вище порога "
                      f"{pct(thr)} — потужності достатньо")
            elif need:
                print(f"  {label}: точкова {pct(point)}, нижня {pct(lo)} "
                      f"проти порога {pct(thr)} -> потрібно n ≈ {need}")
            else:
                print(f"  {label}: точкова оцінка {pct(point)} сама не "
                      f"задовольняє критерій — ширина CI тут не причина")

    out = os.path.join(HERE, f"finqa_confirm_{args.tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,gold,r1,r2,verdict,unit,distance,correct,r1_correct,n_ops\n")
        for r in rows:
            f.write(f"{r['id']},{r['gold']},{r['r1']},{r['r2']},{r['verdict']},"
                    f"{int(r['unit'])},{r['distance']:.6g},{int(r['correct'])},"
                    f"{int(r['r1_correct'])},{r['n_ops']}\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
