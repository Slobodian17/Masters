"""Paired comparison of two cached FinQA runs.

Used for the dataset-conventions ablation: the same questions, the same model,
one prompt with the corpus conventions stated and one without. Because both
arms cover the same question ids, the comparison is paired and McNemar's exact
test applies.

The test is exact (binomial) rather than the chi-square approximation, because
at this sample size the number of discordant pairs is small and the
approximation is not trustworthy there.

Run:  python -m eval.finqa_ab --a ab_no --b ab_yes
"""
import argparse
import json
import os
from math import comb

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")


def load(tag: str) -> dict:
    path = os.path.join(CACHE_DIR, f"{tag}.jsonl")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def mcnemar_exact(b01: int, b10: int) -> float:
    """Two-sided exact McNemar p-value. 1.0 when there are no discordant pairs."""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(2.0 * tail, 1.0)


def summarise(d: dict, ids: list, name: str) -> dict:
    n = len(ids)
    ea = sum(1 for i in ids if d[i]["answer_correct"] is True)
    accepted_wrong = sum(1 for i in ids
                         if d[i]["status"] == "ACCEPT" and d[i]["answer_correct"] is False)
    abstain = sum(1 for i in ids if d[i]["status"] == "ABSTAIN")
    scale = sum(1 for i in ids if d[i].get("candidate_scale_error"))
    calls = sum(d[i]["calls_codegen"] + d[i]["calls_semantic"] for i in ids)
    return {"конфігурація": name, "n": n, "EA": f"{ea}/{n} = {ea / n:.0%}",
            "прийнято хибних": accepted_wrong, "ABSTAIN": abstain,
            "похибка масштабу": scale, "викликів": calls,
            "_ea": ea}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="ab_no", help="cache tag of arm A")
    ap.add_argument("--b", default="ab_yes", help="cache tag of arm B")
    ap.add_argument("--name-a", default="без конвенцій датасету")
    ap.add_argument("--name-b", default="з конвенціями датасету")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    ids = sorted(set(A) & set(B))

    print("=" * 78)
    print("ПАРНЕ ПОРІВНЯННЯ ДВОХ ПРОГОНІВ FinQA")
    print("=" * 78)
    print(f"A = {args.a} ({len(A)} питань), B = {args.b} ({len(B)})")
    print(f"спільних питань: {len(ids)}")
    if not ids:
        print("Перетину немає — порівнювати нічого.")
        return
    print()

    rows = [summarise(A, ids, args.name_a), summarise(B, ids, args.name_b)]
    show = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    print(show.to_string(index=False))

    b01 = sum(1 for i in ids
              if A[i]["answer_correct"] is not True and B[i]["answer_correct"] is True)
    b10 = sum(1 for i in ids
              if A[i]["answer_correct"] is True and B[i]["answer_correct"] is not True)
    p = mcnemar_exact(b01, b10)

    print()
    print("--- Тест Макнемара (точний, двобічний) ---")
    print(f"  A хибно -> B правильно: {b01}")
    print(f"  A правильно -> B хибно: {b10}")
    print(f"  розбіжних пар: {b01 + b10}")
    print(f"  p = {p:.4f}")
    print()
    if b01 + b10 == 0:
        print("  Розбіжних пар немає: на цій вибірці конфігурації не відрізняються.")
    elif p < 0.05:
        print("  Різниця значуща на рівні 0.05.")
    else:
        print(f"  Різниця НЕ значуща (p = {p:.3f}). На {len(ids)} питаннях і")
        print(f"  {b01 + b10} розбіжних парах тест не має потужності підтвердити")
        print("  ефект такого розміру — це не те саме, що відсутність ефекту.")

    print()
    print("  ЗАСТЕРЕЖЕННЯ: LLM недетермінована, тож частина розбіжностей")
    print("  породжена не промптом, а дисперсією генерації. Щоб розділити ці")
    print("  два джерела, потрібен повтор кожної гілки — він не робився.")

    out = os.path.join(HERE, f"finqa_ab_{args.a}_vs_{args.b}.csv")
    pd.DataFrame([{"id": i,
                   "gold": A[i]["gold"],
                   "A_answer": A[i]["answer"], "A_correct": A[i]["answer_correct"],
                   "A_status": A[i]["status"],
                   "B_answer": B[i]["answer"], "B_correct": B[i]["answer_correct"],
                   "B_status": B[i]["status"],
                   "A_scale": A[i].get("candidate_scale_error"),
                   "B_scale": B[i].get("candidate_scale_error")}
                  for i in ids]).to_csv(out, index=False)
    print()
    print("Збережено:", out)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
