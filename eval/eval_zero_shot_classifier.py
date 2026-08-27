"""Measures the zero-shot LLM baseline for query-type classification.

Reports accuracy, latency and cost, so the trained classifier can be judged on
all three rather than on accuracy alone.

Uses the labelled queries from both test sets, deduplicated. Adversarial
queries are excluded: scope is the gate's task and is measured separately.

This is not the final phase comparison; there is no held-out split here, since
these queries later serve as the final test set.
"""
import collections
import json
import os
import time

import pandas as pd

from generation import llm_client
from generation.query_classifier_zeroshot import TAXONOMY, classify, describe

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = ["test_queries_full.json", "test_queries_phase1.json"]

# The free tier costs nothing, so cost is counted in tokens with a paid-tier
# equivalent for reference; otherwise the metric would be a trivial zero.
GROQ_PAID_USD_PER_MTOK_IN = 0.05
GROQ_PAID_USD_PER_MTOK_OUT = 0.08


def load_labelled() -> list:
    seen, out = set(), []
    for name in SETS:
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            for item in json.load(f):
                q = item["query"]
                if q in seen:
                    continue
                seen.add(q)
                if item["type"] == "ADVERSARIAL":
                    # поза областю визначення -> це задача scope_gate, не М3a
                    continue
                out.append({"query": q, "gold": item["type"], "source": name})
    return out


def run():
    provider = llm_client.active_provider()
    if provider == "mock":
        print("=" * 78)
        print("ZERO-SHOT BASELINE (М3a) — НЕ ВИКОНАНО")
        print("=" * 78)
        print("GROQ_API_KEY не встановлено, тож активний провайдер — mock.")
        print("Zero-shot узагальнення неможливо змоделювати в MOCK: саме воно")
        print("й вимірюється. Отримай безкоштовний ключ на console.groq.com і:")
        print("    GROQ_API_KEY=gsk_... python -m eval.eval_zero_shot_classifier")
        print("\nРезультат: N/A (чесно, а не підмінено числом).")
        return None

    data = load_labelled()
    info = describe()
    print("=" * 78)
    print("ZERO-SHOT BASELINE класифікації типу запиту (М3a, Фаза 2)")
    print("=" * 78)
    print(f"Провайдер: {info['provider']}   модель: {info['model']}   класів: {info['classes']}")
    print(f"Запитів (унікальних, з двох наборів): {len(data)}")
    print()

    llm_client.reset_stats()
    rows, started = [], time.time()
    for i, item in enumerate(data, start=1):
        try:
            pred = classify(item["query"])
            err = None
        except Exception as e:
            pred, err = "ERROR", f"{type(e).__name__}: {e}"
        rows.append({**item, "predicted": pred, "correct": pred == item["gold"], "error": err})
        mark = "+" if pred == item["gold"] else "x"
        print(f"  [{i:2d}/{len(data)}] {mark} gold={item['gold']:<19s} pred={pred:<19s} {item['query'][:44]}")
        if err:
            print(f"          ПОМИЛКА: {err}")

    wall = time.time() - started
    df = pd.DataFrame(rows)
    stats = llm_client.get_stats()

    acc = df["correct"].mean()
    print("\n--- Точність ---")
    print(f"  accuracy (zero-shot): {acc:.1%}   ({df['correct'].sum()}/{len(df)})")
    print(f"  поріг готовності Фази 2 для НАВЧЕНОГО класифікатора: 90%")

    print("\n--- Точність за класами ---")
    per = df.groupby("gold").agg(n=("correct", "size"), hit=("correct", "sum"))
    per["accuracy"] = (per["hit"] / per["n"]).map(lambda v: f"{v:.0%}")
    print(per.to_string())

    print("\n--- Найчастіші плутанини ---")
    conf = collections.Counter((r.gold, r.predicted) for r in df.itertuples() if not r.correct)
    if conf:
        for (g, p), c in conf.most_common(8):
            print(f"  {g} -> {p}: {c}")
    else:
        print("  немає")

    print("\n--- Швидкість ---")
    print(f"  сумарний час стіни:      {wall:.1f} с")
    print(f"  середня латентність:     {stats['avg_latency_s']:.2f} с/запит")
    print(f"  ретраїв через ліміт 429: {stats['retries']}")

    tok_in, tok_out = stats["prompt_tokens"], stats["completion_tokens"]
    paid = tok_in / 1e6 * GROQ_PAID_USD_PER_MTOK_IN + tok_out / 1e6 * GROQ_PAID_USD_PER_MTOK_OUT
    print("\n--- Вартість ---")
    print(f"  токенів вхід/вихід:      {tok_in} / {tok_out}  (разом {stats['total_tokens']})")
    print(f"  фактична вартість:       $0.00 (безкоштовний тариф Groq)")
    print(f"  еквівалент на платному:  ${paid:.6f} за {len(df)} запитів "
          f"(~${paid / max(len(df), 1) * 1000:.4f} за 1000 запитів)")
    print("  локальний класифікатор (М3a): $0 і без мережі — саме це й треба довести")

    out = os.path.join(HERE, "zero_shot_classifier_results.csv")
    df.to_csv(out, index=False)
    print(f"\nЗбережено: {out}")
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 150)
    run()
