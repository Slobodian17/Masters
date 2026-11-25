"""What the cascade could reject if it were tuned, measured on cached runs.

Two of the three wrong answers the cascade accepted on FinQA had
`v_semantic_ok` already False. The signal fired and the weighted score
overruled it, because that signal carries 0.10 of the weight and 0.90 of
support is enough to clear a 0.70 threshold. So before adding a new signal it
is worth knowing how much of the gap is a tuning problem rather than a missing
mechanism.

Everything here reads the cache written by `eval.finqa_run`. No LLM call, no
budget: the signal vectors and the correctness labels are already recorded,
and a threshold sweep is arithmetic over them.

Two caveats that belong next to the numbers, not under them:

  * the arms share question ids, so pooling them gives 50 decisions but not 50
    independent ones. Per-arm figures are reported alongside;
  * a threshold chosen on the same data it is evaluated on is not a held-out
    result. This sweep says what is reachable, not what would generalise.

Run:  python -m eval.finqa_threshold --tags ab_no,ab_yes
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from core import calibration, conformal
from eval.finqa_replay import load_cache

HERE = os.path.dirname(os.path.abspath(__file__))

THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


def episodes(rows: list) -> list:
    """Cached decisions that carry both a signal vector and a verdict."""
    out = []
    for r in rows:
        if not r.get("signals") or r.get("candidate_correct") is None:
            continue
        out.append({"signals": r["signals"],
                    "correct": bool(r["candidate_correct"]),
                    "id": r["id"]})
    return out


def rates(eps: list, threshold: float, weights: dict = None) -> dict:
    saved = dict(calibration._HEURISTIC_WEIGHTS)
    if weights:
        calibration._HEURISTIC_WEIGHTS.clear()
        calibration._HEURISTIC_WEIGHTS.update(weights)
    try:
        conf = [calibration.confidence(e["signals"]) for e in eps]
    finally:
        calibration._HEURISTIC_WEIGHTS.clear()
        calibration._HEURISTIC_WEIGHTS.update(saved)

    rejected = [c < threshold for c in conf]
    wrong = [not e["correct"] for e in eps]

    n_wrong = sum(wrong)
    n_rej = sum(rejected)
    caught = sum(1 for r, w in zip(rejected, wrong) if r and w)
    kept_right = sum(1 for r, w in zip(rejected, wrong) if not r and not w)

    return {
        "threshold": threshold,
        "n": len(eps),
        "wrong": n_wrong,
        "rejected": n_rej,
        "caught": caught,
        "false_rejections": n_rej - caught,
        "recall": caught / n_wrong if n_wrong else None,
        "precision": caught / n_rej if n_rej else None,
        "kept_right": kept_right,
    }


def _f(x):
    return "—" if x is None else f"{x:.1%}"


def sweep(eps: list, weights: dict = None) -> pd.DataFrame:
    rows = [rates(eps, t, weights) for t in THRESHOLDS]
    return pd.DataFrame([{
        "поріг": f"{r['threshold']:.2f}",
        "відхилено": r["rejected"],
        "з них хибних": r["caught"],
        "хибних відхилень": r["false_rejections"],
        "recall": _f(r["recall"]),
        "precision": _f(r["precision"]),
    } for r in rows])


# Weight variants. The only change is how much a failing semantic or schema
# signal is allowed to matter; nothing is re-derived from data, so these are
# candidates to measure, not a fitted model.
WEIGHT_VARIANTS = {
    "поточні": None,
    "semantic 0.30": {"v_type_ok": 0.30, "v_schema_ok": 0.15,
                      "v_range_ok": 0.25, "v_cross_table": 0.10,
                      "v_semantic_ok": 0.30},
    "рівні": {k: 0.20 for k in calibration._FEATURES},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="ab_no,ab_yes")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    per_tag = {t: episodes(load_cache(t)) for t in tags}
    pooled = [e for t in tags for e in per_tag[t]]

    print("=" * 78)
    print("ЩО КАСКАД МІГ БИ ВІДХИЛЯТИ ПРИ ІНШОМУ ПОРОЗІ — на кешах, без LLM")
    print("=" * 78)
    print(f"кеші: {', '.join(tags)}")
    for t in tags:
        e = per_tag[t]
        print(f"  {t}: рішень з вердиктом {len(e)}, з них хибних {sum(1 for x in e if not x['correct'])}")
    print(f"  разом: {len(pooled)}, хибних {sum(1 for x in pooled if not x['correct'])}")
    print()
    print("Орієнтир FinAgent-RAG: recall 58.4%, precision 76.2%.")
    print("УВАГА: питання в гілках ті самі, тож 50 рішень НЕ є 50 незалежними.")
    print("І поріг, підібраний на цих же даних, не є held-out результатом.")
    print()

    print("--- Поточні ваги, об'єднано ---")
    print(sweep(pooled).to_string(index=False))

    print()
    print("--- Робочий поріг 0.70 по гілках окремо ---")
    rows = []
    for t in tags:
        r = rates(per_tag[t], 0.70)
        rows.append({"кеш": t, "хибних": r["wrong"], "відхилено": r["rejected"],
                     "спіймано": r["caught"], "recall": _f(r["recall"]),
                     "precision": _f(r["precision"])})
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("--- Варіанти ваг, поріг 0.70, об'єднано ---")
    rows = []
    for name, w in WEIGHT_VARIANTS.items():
        r = rates(pooled, 0.70, w)
        rows.append({"ваги": name, "відхилено": r["rejected"],
                     "спіймано": r["caught"],
                     "хибних відхилень": r["false_rejections"],
                     "recall": _f(r["recall"]), "precision": _f(r["precision"])})
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("--- Крива ризик-покриття (core/conformal, вперше підключений) ---")
    rc = conformal.risk_coverage_curve(pooled)
    print(f"  AURC = {rc['aurc']:.4f}   (нижче — краще; 0 означало б ідеальне впорядкування)")
    pts = rc["points"]
    show = [p for p in pts if abs(p["coverage"] * 10 - round(p["coverage"] * 10)) < 0.06]
    seen = set()
    rows = []
    for p in show:
        k = round(p["coverage"], 1)
        if k in seen:
            continue
        seen.add(k)
        rows.append({"покриття": f"{p['coverage']:.0%}",
                     "ризик серед прийнятих": f"{p['risk']:.1%}",
                     "поріг": f"{p['threshold']:.3f}"})
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("  Якщо ризик не падає зі зменшенням покриття — оцінка довіри не")
    print("  впорядковує правильні відповіді перед хибними, і жоден поріг не")
    print("  допоможе. Це перевірка самої φ(v), а не її налаштування.")

    out = os.path.join(HERE, "finqa_threshold_sweep.csv")
    sweep(pooled).to_csv(out, index=False)
    print()
    print("Збережено:", out)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
