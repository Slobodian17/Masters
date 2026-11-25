"""Would switching off v_semantic change any decision the code route made?

v_semantic costs one API call per question -- 488 tokens of 1923, 25% of the
budget of every run. The new decision rule is agreement between R1 and R2, so
the cascade's own verdict no longer decides anything. What v_semantic can
still change is the REPAIR TRAJECTORY: if confidence crosses the threshold
differently, the pipeline regenerates differently and the candidate value
differs.

That is answerable from cache. `_heuristic_confidence` renormalises over the
signals that are applicable, so omitting v_semantic is exactly what happens
when it is None -- no new code path, no approximation.

If no step's accept/reject flips, the trajectory would have been identical and
the call is free to remove.

Run:  python -m eval.finqa_semantic_ablation
"""
import argparse
import collections
import os

from core import calibration
from eval import cache_io

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")
THRESHOLD = 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    if calibration.load_model() is not None:
        print("УВАГА: завантажено навчений калібратор — абляція сигналу не")
        print("зводиться до перерахунку ваг, потрібен інший аналіз. Зупиняюсь.")
        return 1

    rows, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.tag}.jsonl"))
    print("=" * 74)
    print("АБЛЯЦІЯ v_semantic: чи змінює вона хоч одне рішення (нуль викликів)")
    print("=" * 74)
    print(f"питань: {len(rows)}   поріг: {args.threshold}")
    print()

    n_steps = flips = 0
    drift = []
    flip_rows = set()
    present = 0
    for r in rows:
        for st in (r.get("steps") or []):
            sig = st.get("signals")
            if not sig:
                continue
            n_steps += 1
            if "v_semantic_ok" in sig:
                present += 1
            full = calibration.confidence(sig)
            without = calibration.confidence(
                {k: v for k, v in sig.items() if k != "v_semantic_ok"})
            drift.append(abs(full - without))
            if (full >= args.threshold) != (without >= args.threshold):
                flips += 1
                flip_rows.add(r["id"])

    print(f"кроків із вектором сигналів        : {n_steps}")
    print(f"з них мають v_semantic_ok          : {present}")
    print(f"РІШЕНЬ, ЩО ЗМІНИЛИСЬ (перетин порога): {flips}")
    print(f"питань, яких це торкнулось         : {len(flip_rows)}")
    if drift:
        drift.sort()
        print(f"зсув довіри: медіана {drift[len(drift) // 2]:.4f}   "
              f"макс {drift[-1]:.4f}")

    # what the signal was actually saying
    vals = collections.Counter(
        st.get("signals", {}).get("v_semantic_ok")
        for r in rows for st in (r.get("steps") or []) if st.get("signals"))
    print(f"розподіл самого сигналу: {dict(vals)}")

    print()
    if flips == 0:
        print("ВИСНОВОК: жодне рішення не змінюється. Траєкторія ремонту була б")
        print("тією самою, значення кандидата — тим самим. v_semantic можна")
        print("вимкнути для нових прогонів без впливу на результат, і це")
        print("звільняє ~25% токенного бюджету.")
    else:
        print(f"ВИСНОВОК: {flips} рішень змінюються — вимикати НЕ можна,")
        print("нові прогони були б не порівнянні зі старими.")
    for qid in sorted(flip_rows)[:10]:
        print("   ", qid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
