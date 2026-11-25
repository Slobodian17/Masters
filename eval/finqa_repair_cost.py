"""What blind repair cost and bought, and what targeted repair would cost.

Blind repair is not a projection: it already ran on all 300 questions and the
cache records every iteration. So the comparison has one measured side and one
estimated side, and this module keeps them visibly apart -- the measured
columns come from `confirm300.jsonl`, the estimate from the signal set that
`eval.finqa_perturb` wrote.

Zero API calls.

Run:  python -m eval.finqa_repair_cost --tag confirm300
"""
import argparse
import collections
import csv
import os

from eval import cache_io
from eval.finqa_confirm import is_correct

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")


def first_result(row):
    steps = row.get("steps") or []
    if not steps:
        return None, None
    s = steps[0]
    return s.get("result"), s.get("exception_type")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    args = ap.parse_args()

    rows = cache_io.load_rows(os.path.join(CACHE, f"{args.tag}.jsonl"))[0]

    print("=" * 78)
    print(f"РЕМОНТ: НАОСЛІП (зміряно) ПРОТИ АДРЕСНОГО (оцінка) — {args.tag}")
    print("=" * 78)

    rescued, broken, kept_wrong, no_repair = [], [], [], 0
    extra_calls = extra_tokens = 0.0
    iters = collections.Counter()
    for r in rows:
        gold = r.get("gold")
        final = r.get("candidate")
        if final is None:
            final = r.get("answer")
        f0, exc = first_result(r)
        n_cg = r.get("calls_codegen") or 1
        n_all = r.get("calls_total") or 1
        iters[r.get("iterations") or 0] += 1
        if n_cg > 1:
            extra_calls += n_cg - 1
            extra_tokens += (r.get("tokens") or 0) / max(n_all, 1) * (n_cg - 1)
        else:
            no_repair += 1
            continue
        ok0 = f0 is not None and bool(is_correct(f0, gold))
        ok1 = final is not None and bool(is_correct(final, gold))
        if not ok0 and ok1:
            rescued.append(r["id"])
        elif ok0 and not ok1:
            broken.append(r["id"])
        elif not ok0 and not ok1:
            kept_wrong.append(r["id"])

    n_rep = len(rows) - no_repair
    print()
    print("--- РЕМОНТ НАОСЛІП, зміряно в кеші ---")
    print(f"  питань усього                    : {len(rows)}")
    print(f"  ремонт НЕ запускався (1 ітерація): {no_repair}")
    print(f"  ремонт запускався                : {n_rep}")
    print(f"  з них ВРЯТОВАНО                  : {len(rescued)}")
    print(f"  з них ЗЛАМАНО                    : {len(broken)}")
    print(f"  лишились хибними                 : {len(kept_wrong)}")
    print(f"  зайвих викликів генерації        : {extra_calls:.0f}")
    print(f"  зайвих токенів                   : {extra_tokens:,.0f}")
    if extra_calls:
        print(f"  ціна одного врятованого питання  : "
              f"{extra_calls / max(len(rescued), 1):.1f} викликів, "
              f"{extra_tokens / max(len(rescued), 1):,.0f} токенів")
    print(f"  розподіл ітерацій                : "
          f"{dict(sorted(iters.items()))}")

    csv_path = os.path.join(HERE, f"finqa_perturb_{args.tag}.csv")
    if not os.path.exists(csv_path):
        print()
        print(f"Немає {csv_path} — спершу `python -m eval.finqa_perturb`.")
        return

    flag_w = flag_c = tot_w = tot_c = 0
    addressable = 0
    with open(csv_path, encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if d["repro"] != "1":
                continue
            has = bool(d["signals"].strip())
            if d["correct"] == "1":
                tot_c += 1
                flag_c += has
            else:
                tot_w += 1
                flag_w += has
                if has and d["loc"] in ("HIT_SOME", "HIT_NONE"):
                    addressable += 1

    flagged = flag_w + flag_c
    print()
    print("--- АДРЕСНИЙ РЕМОНТ, оцінка з виміряних сигналів ---")
    print(f"  позначено сигналом               : {flagged}"
          f"  (хибних {flag_w} з {tot_w}, правильних {flag_c} з {tot_c})")
    print(f"  виявлення (recall)               : {flag_w / max(tot_w, 1):.1%}")
    print(f"  точність позначення (precision)  : "
          f"{flag_w / max(flagged, 1):.1%}")
    print(f"  позначених із адресою на клітинку: {addressable}")
    print()
    print("  Схема витрат на питання: 1 перевірка LLM + 1 ремонт, лише для")
    print("  позначених. Верифікація виконанням — нуль викликів.")
    for tok in (900, 1200):
        calls = flagged * 2
        print(f"    {flagged} позначених x 2 виклики x {tok} ток = "
              f"{calls * tok:,} ток"
              f" = {calls * tok / 200000:.2f} доби Groq"
              f" / {calls / 500:.2f} доби flash-lite")
    print()
    print("  Для порівняння — ремонт наослі­п витратив "
          f"{extra_calls:.0f} викликів на {len(rescued)} врятованих.")
    print("  Адресний має побити цю ціну, інакше він не вартий роботи.")


if __name__ == "__main__":
    main()
