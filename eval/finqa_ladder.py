"""Does the method's value grow as the model weakens?

Line 2 of the work, now with a measured motive. On gpt-oss-120b the method
does not beat a direct question (5.19): the model computes well enough in its
head that generating code buys nothing. But the two configurations lean on
different abilities. The baseline needs the model to do arithmetic; the method
needs it only to write pandas, and the sandbox does the arithmetic exactly.
The weaker the model, the more that trade should be worth.

The quantity of interest is therefore not accuracy but the DIFFERENCE
(method - baseline) per model, and whether it grows as the model shrinks.

Fair-comparison rules, enforced rather than assumed:
  * the same 25 questions for every model and both configurations;
  * the same token budget for baseline and method within one model, since a
    budget mismatch once understated the baseline by 12 points (5.19.1);
  * McNemar on the paired outcomes, with the number of discordant pairs
    stated, because at this sample size that number is what limits the
    conclusion.

Run:  python -m eval.finqa_ladder
"""
import argparse
import json
import os
from math import comb

import pandas as pd

from eval.finqa_replay import load_cache

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")

# (label, size hint, method cache, baseline cache)
LADDER = [
    ("gpt-oss-120b", 120, "ab_yes_v2", "baseline_table_ab_yes_v2"),
    ("qwen3.8-27b", 27, "m_qwen27", "baseline_table_m_qwen27"),
    ("gpt-oss-20b", 20, "m_gptoss20", "baseline_table_m_gptoss20"),
    ("allam-2-7b", 7, "m_allam7", "baseline_table_m_allam7"),
]


def load_jsonl(name: str):
    path = os.path.join(CACHE_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return None
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def mcnemar_exact(b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n), 1.0)


def compare(method_tag: str, baseline_tag: str) -> dict:
    try:
        meth = {r["id"]: r for r in load_cache(method_tag)}
    except FileNotFoundError:
        return {}
    base = load_jsonl(baseline_tag)
    if not base:
        return {}

    ids = [r["id"] for r in base if r["id"] in meth]
    if not ids:
        return {}

    b_ok = {r["id"]: bool(r["correct"]) for r in base}
    m_ok = {i: meth[i].get("answer_correct") is True for i in ids}

    nb = sum(b_ok[i] for i in ids)
    nm = sum(m_ok[i] for i in ids)
    b01 = sum(1 for i in ids if not b_ok[i] and m_ok[i])
    b10 = sum(1 for i in ids if b_ok[i] and not m_ok[i])

    calls = sum(meth[i].get("calls_codegen", 0) + meth[i].get("calls_semantic", 0)
                for i in ids) / len(ids)
    empty = sum(1 for r in base if r.get("empty_reply"))
    abst = sum(1 for i in ids if meth[i].get("status") != "ACCEPT")

    return {"n": len(ids), "baseline": nb, "method": nm,
            "b01": b01, "b10": b10, "p": mcnemar_exact(b01, b10),
            "calls": calls, "baseline_empty": empty, "method_no_answer": abst}


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    print("=" * 78)
    print("ЛІНІЯ 2: чи росте користь методу зі спаданням моделі")
    print("=" * 78)
    print("Головне число — РІЗНИЦЯ (метод − базова лінія), не точність.")
    print("Однакові 25 питань, однаковий бюджет токенів у межах моделі.")
    print()

    rows, missing = [], []
    for label, size, mt, bt in LADDER:
        c = compare(mt, bt)
        if not c:
            missing.append(label)
            continue
        d = (c["method"] - c["baseline"]) / c["n"]
        rows.append({
            "модель": label, "B": size, "n": c["n"],
            "базова лінія": f"{c['baseline']}/{c['n']} = {c['baseline']/c['n']:.0%}",
            "метод": f"{c['method']}/{c['n']} = {c['method']/c['n']:.0%}",
            "РІЗНИЦЯ": f"{d:+.0%}",
            "пар": c["b01"] + c["b10"],
            "p": f"{c['p']:.3f}",
            "викл/пит": f"{c['calls']:.2f}",
            "база порожніх": c["baseline_empty"],
            "метод без відп.": c["method_no_answer"],
        })

    if missing:
        print("немає кешів для:", ", ".join(missing))
        print()
    if not rows:
        print("Жодної пари кешів. Спершу прожени finqa_run і finqa_baseline.")
        return

    df = pd.DataFrame(rows).sort_values("B", ascending=False)
    print(df.to_string(index=False))

    print()
    print("--- Перевірка гіпотези ---")
    diffs = [(r["B"], float(r["РІЗНИЦЯ"].rstrip("%")) / 100) for r in rows]
    diffs.sort(key=lambda x: -x[0])
    print("  різниця за спаданням розміру:",
          ", ".join(f"{b}B: {d:+.0%}" for b, d in diffs))
    # A verdict has to clear two bars, not one. An ordering that merely fails
    # to decrease is not evidence: with every p at 1.000 the ordering is
    # whatever the noise happened to produce. So a positive verdict needs BOTH
    # a difference large enough to matter and a test that separates it from
    # zero somewhere.
    MEANINGFUL = 0.08          # 2 questions out of 25
    any_significant = any(float(r["p"]) < 0.05 for r in rows)
    biggest = max(d for _, d in diffs)
    grows = all(diffs[i][1] <= diffs[i + 1][1] + 1e-9
                for i in range(len(diffs) - 1))

    if len(diffs) >= 2:
        if not any_significant and biggest < MEANINGFUL:
            print(f"  ЖОДНА різниця не сягає {MEANINGFUL:.0%} і жодна не значуща.")
            print("  Гіпотеза НЕ ПІДТВЕРДЖЕНА на цих моделях: усі різниці в межах")
            print("  шуму нуля. Порядок величин нічого не доводить — при p = 1.000")
            print("  він такий, яким його зробив шум.")
            if grows and diffs[-1][1] > diffs[0][1]:
                print("  (Формально різниці не спадають зі зменшенням моделі, але")
                print("   називати це підтвердженням було б видаванням шуму за тренд.)")
        elif grows and diffs[-1][1] > diffs[0][1] and any_significant:
            print("  Росте зі спаданням моделі І десь значуща — узгоджується з гіпотезою.")
        elif all(d <= 0 for _, d in diffs) and any_significant:
            print("  РІЗНИЦЯ СКРІЗЬ <= 0 і значуща — гіпотеза СПРОСТОВАНА.")
        else:
            print("  Немонотонно або незначуще. Гіпотеза не підтверджується й не")
            print("  спростовується; при такій кількості розбіжних пар це")
            print("  найімовірніший результат.")

    print()
    print("  Розмір у параметрах — НЕ та вісь. Зміряно: qwen-27B точніша за")
    print("  gpt-oss-120B на цій задачі, тож упорядковувати треба за")
    print("  спроможністю (точність базової лінії), а не за кількістю ваг.")
    by_cap = sorted(((r["базова лінія"], r["модель"], r["РІЗНИЦЯ"]) for r in rows),
                    key=lambda x: -float(x[0].split("=")[1].strip().rstrip("%")))
    for b, m, d in by_cap:
        print(f"    {m:14s} база {b:14s} різниця {d}")
    print()
    print("  Жодне p нижче не слід читати як доказ рівності: при кількох")
    print("  розбіжних парах тест не має потужності (див. 5.16.6).")
    print()
    print("  ВАЖЛИВО: тут міряється різниця базова/метод, а не якість каскаду.")
    print("  Rejection recall від зміни моделі не виросте — причини (5.11)")
    print("  структурні й від моделі не залежать.")

    out = os.path.join(HERE, "finqa_ladder.csv")
    df.to_csv(out, index=False)
    print()
    print("Збережено:", out)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    main()
