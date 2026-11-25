"""Agreement between independent readings as a confidence signal.

The cascade's four surrogate signals do not separate right from wrong on real
data (5.11, 5.15), and 5.18 says why: a check that uses the same information
the same way as the generator inherits its blind spots. Agreement between
INDEPENDENT readings has no such flaw by construction -- two sources that fail
the same way are not independent, and the degree to which they fail
differently is exactly what makes the signal informative.

Three levels of independence are available from the caches already paid for:

  path    same model, different route: direct answer vs generated code
  size    same family, different size: gpt-oss-120b vs gpt-oss-20b
  family  different training lineage: gpt-oss vs qwen

The hypothesis is that the gap between accuracy-under-agreement and
accuracy-under-disagreement widens with independence.

PRE-REGISTERED, fixed before any number was computed:
  * agreement tolerance 1% relative, the same as execution accuracy;
  * two answers differing by exactly a power of ten count as agreeing ON THE
    ROW and are reported separately -- the sources read the same cell and
    differ only in the unit, which is a real distinction for a signal meant
    to detect reading the wrong thing;
  * a source with no numeric answer counts as absent, not as disagreement.

Zero API calls: everything is read from finqa_cache.

Run:  python -m eval.finqa_agreement
"""
import argparse
import collections
import itertools
import json
import os

import numpy as np
import pandas as pd

from core import calibration, conformal
from eval.finqa_replay import load_cache

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")

TOL = 0.01
SCALE_FACTORS = (1e1, 1e2, 1e3, 1e6, 1e9)

# (label, family, size, kind, cache, is_baseline)
SOURCES = [
    ("120b-direct", "gpt-oss", 120, "direct", "baseline_table_ab_yes_v2", True),
    ("120b-code", "gpt-oss", 120, "code", "ab_yes_v2", False),
    ("27b-direct", "qwen", 27, "direct", "baseline_table_m_qwen27", True),
    ("27b-code", "qwen", 27, "code", "m_qwen27", False),
    ("20b-direct", "gpt-oss", 20, "direct", "baseline_table_m_gptoss20", True),
    ("20b-code", "gpt-oss", 20, "code", "m_gptoss20", False),
]


def load_source(cache: str, is_baseline: bool) -> dict:
    path = os.path.join(CACHE_DIR, f"{cache}.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    if is_baseline:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    out[r["id"]] = {"answer": r["answer"], "gold": r["gold"],
                                    "correct": bool(r["correct"])}
    else:
        for r in load_cache(cache):
            out[r["id"]] = {"answer": r["answer"] if r["status"] == "ACCEPT" else None,
                            "gold": r["gold"],
                            "correct": r["answer_correct"] is True,
                            "signals": r.get("signals")}
    return out


def close(a, b, tol: float = TOL) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(abs(b), 1.0)


def agreement(a, b) -> str:
    """-> 'agree' | 'scale' | 'disagree' | 'absent'."""
    if a is None or b is None:
        return "absent"
    if close(a, b) or close(b, a):
        return "agree"
    for k in SCALE_FACTORS:
        if close(a * k, b) or close(a / k, b) or close(b * k, a) or close(b / k, a):
            return "scale"
    return "disagree"


def independence(s1, s2) -> str:
    _, fam1, size1, kind1, _, _ = s1
    _, fam2, size2, kind2, _, _ = s2
    if fam1 == fam2 and size1 == size2:
        return "1-шлях"
    if fam1 == fam2:
        return "2-розмір"
    return "3-сімейство"


def pair_stats(A: dict, B: dict, ids: list) -> dict:
    buckets = collections.Counter()
    ok_agree = ok_dis_a = ok_dis_b = 0
    n_agree = n_dis = 0
    for i in ids:
        a, b = A.get(i), B.get(i)
        if not a or not b:
            continue
        verdict = agreement(a["answer"], b["answer"])
        buckets[verdict] += 1
        if verdict in ("agree", "scale"):
            n_agree += 1
            ok_agree += a["correct"] or b["correct"] if verdict == "scale" else a["correct"]
        elif verdict == "disagree":
            n_dis += 1
            ok_dis_a += a["correct"]
            ok_dis_b += b["correct"]
    n = sum(buckets.values())
    return {
        "n": n, "agree": buckets["agree"], "scale": buckets["scale"],
        "disagree": buckets["disagree"], "absent": buckets["absent"],
        "coverage": (buckets["agree"] + buckets["scale"]) / n if n else None,
        "acc_agree": ok_agree / n_agree if n_agree else None,
        "acc_dis_a": ok_dis_a / n_dis if n_dis else None,
        "acc_dis_b": ok_dis_b / n_dis if n_dis else None,
        "n_agree": n_agree, "n_dis": n_dis,
    }


def _f(x):
    return "—" if x is None else f"{x:.0%}"


def majority(sources: dict, ids: list) -> dict:
    """Most common answer across all sources, within tolerance."""
    hits, covered, rows = 0, 0, []
    for i in ids:
        vals = [(k, s[i]["answer"]) for k, s in sources.items()
                if i in s and s[i]["answer"] is not None]
        if not vals:
            continue
        clusters = []
        for k, v in vals:
            placed = False
            for c in clusters:
                if close(v, c["rep"]):
                    c["members"].append(k)
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": v, "members": [k]})
        clusters.sort(key=lambda c: -len(c["members"]))
        best = clusters[0]
        gold = next(s[i]["gold"] for s in sources.values() if i in s)
        ok = close(best["rep"], gold)
        hits += ok
        covered += 1
        rows.append({"id": i, "votes": len(best["members"]), "of": len(vals),
                     "answer": best["rep"], "gold": gold, "correct": ok})
    return {"hits": hits, "n": covered, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    sources = {}
    for spec in SOURCES:
        label, _, _, _, cache, is_base = spec
        d = load_source(cache, is_base)
        if d:
            sources[label] = d

    ids = sorted(set.intersection(*[set(s) for s in sources.values()])) \
        if sources else []

    print("=" * 78)
    print("ЗГОДА НЕЗАЛЕЖНИХ ПРОЧИТАНЬ ЯК СИГНАЛ ДОВІРИ")
    print("=" * 78)
    print(f"джерел: {len(sources)} — {', '.join(sources)}")
    print(f"спільних питань: {len(ids)}")
    print(f"ПОПЕРЕДНІЙ РЕЗУЛЬТАТ на {len(ids)} питаннях, менше ніж {len(ids)} кластерів.")
    print("Поріг згоди 1% зафіксовано до підрахунку; збіг з точністю до 10^k")
    print("рахується як «згода щодо рядка» й позначається окремо.")
    print()

    if not ids:
        print("Немає спільних питань.")
        return

    print("--- Точність кожного джерела окремо ---")
    for k, s in sources.items():
        ok = sum(1 for i in ids if s[i]["correct"])
        print(f"  {k:12s} {ok}/{len(ids)} = {ok/len(ids):.0%}")

    print()
    print("--- Пари, за рівнем незалежності ---")
    rows = []
    spec_by_label = {s[0]: s for s in SOURCES}
    for a, b in itertools.combinations(sources, 2):
        st = pair_stats(sources[a], sources[b], ids)
        if not st["n"]:
            continue
        gap = (st["acc_agree"] - max(st["acc_dis_a"] or 0, st["acc_dis_b"] or 0)) \
            if st["acc_agree"] is not None and st["n_dis"] else None
        rows.append({
            "рівень": independence(spec_by_label[a], spec_by_label[b]),
            "пара": f"{a} / {b}",
            "згода": st["agree"], "лише масштаб": st["scale"],
            "розбіжність": st["disagree"],
            "покриття": _f(st["coverage"]),
            "(а) правильних при згоді": _f(st["acc_agree"]),
            "(б) правильних при розбіжності": f"{_f(st['acc_dis_a'])} / {_f(st['acc_dis_b'])}",
            "розрив (а)−(б)": _f(gap) if gap is not None else "—",
        })
    df = pd.DataFrame(rows).sort_values("рівень")
    print(df.to_string(index=False))

    print()
    print("--- Чи росте розрив із незалежністю ---")
    by_level = collections.defaultdict(list)
    for r in rows:
        g = r["розрив (а)−(б)"]
        if g != "—":
            by_level[r["рівень"]].append(float(g.rstrip("%")) / 100)
    for lvl in sorted(by_level):
        v = by_level[lvl]
        print(f"  {lvl:12s} середній розрив {np.mean(v):+.0%}  (пар: {len(v)})")
    levels = sorted(by_level)
    if len(levels) >= 2:
        means = [np.mean(by_level[l]) for l in levels]
        if all(means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1)) \
                and means[-1] - means[0] >= 0.08:
            print("  РОСТЕ з незалежністю — узгоджується з гіпотезою.")
        else:
            print("  НЕ РОСТЕ монотонно або приріст менший за 8 п.п.")
            print("  На цій вибірці гіпотеза не підтверджується.")

    print()
    print("--- (г) Голосування більшістю по всіх джерелах ---")
    m = majority(sources, ids)
    best_single = max(sum(1 for i in ids if s[i]["correct"]) for s in sources.values())
    print(f"  більшість: {m['hits']}/{m['n']} = {m['hits']/m['n']:.0%}")
    print(f"  найкраще одиночне джерело: {best_single}/{len(ids)} = {best_single/len(ids):.0%}")
    vd = collections.Counter(r["votes"] for r in m["rows"])
    print(f"  розподіл голосів за переможця: {dict(sorted(vd.items()))}")
    for k in sorted(vd):
        sub = [r for r in m["rows"] if r["votes"] == k]
        ok = sum(1 for r in sub if r["correct"])
        print(f"    {k} голос(ів): {ok}/{len(sub)} правильних = {ok/len(sub):.0%}")

    print()
    print("--- (д) Ризик-покриття: згода проти φ(v) ---")
    votes = {r["id"]: r["votes"] for r in m["rows"]}
    eps_agree = [{"signals": None, "correct": r["correct"], "score": r["votes"]}
                 for r in m["rows"]]
    order = sorted(eps_agree, key=lambda e: -e["score"])
    risks, pts = [], []
    err = 0
    for k, e in enumerate(order, start=1):
        err += 0 if e["correct"] else 1
        risks.append(err / k)
        pts.append((k / len(order), err / k))
    print(f"  згода (к-сть голосів як оцінка): AURC = {np.mean(risks):.4f}")
    for cov in (0.25, 0.5, 0.75, 1.0):
        c = min(pts, key=lambda p: abs(p[0] - cov))
        print(f"     покриття {c[0]:.0%} -> ризик {c[1]:.0%}")

    method = sources.get("120b-code", {})
    eps_phi = [{"signals": method[i]["signals"], "correct": method[i]["correct"]}
               for i in ids if method.get(i, {}).get("signals")]
    if eps_phi:
        rc = conformal.risk_coverage_curve(eps_phi)
        print(f"  φ(v) на тих самих питаннях: AURC = {rc['aurc']:.4f}")
        for cov in (0.5, 1.0):
            c = min(rc["points"], key=lambda p: abs(p["coverage"] - cov))
            print(f"     покриття {c['coverage']:.0%} -> ризик {c['risk']:.0%}")
    print()
    print("  Нижчий AURC кращий. Базова частота помилок = ризик при покритті 100%.")

    out = os.path.join(HERE, "finqa_agreement.csv")
    df.to_csv(out, index=False)
    print()
    print("Збережено:", out)


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    main()
