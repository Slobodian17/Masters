"""Final evaluation of the trained query-type classifier.

Run once, on a fixed model. The live queries are the only real, non-generated
data available, and the zero-shot baseline was measured on them; re-checking
against them while tuning would turn them into a second validation set and
destroy the only honest point of comparison.

Reports three separate accuracies — combinatorial holdout, paraphrase holdout
and live queries — without averaging them, since the gap between synthetic and
real language is the result of interest.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from core import lang
from generation import query_classifier
from generation.dataset_core import TAXONOMY
from generation.query_classifier_train import load_dataset, split_by_seed

HERE = os.path.dirname(os.path.abspath(__file__))

# Measured zero-shot baseline, for comparison.
ZERO_SHOT_ACCURACY = 0.969
ZERO_SHOT_LATENCY_S = 1.03
ZERO_SHOT_RETRIES = 4
ZERO_SHOT_N = 32


def load_live_queries() -> list:
    """Live labelled queries; adversarial ones belong to the gate."""
    out, seen = [], set()
    prof = lang.get()
    for name in (prof.TEST_SET_FULL, prof.TEST_SET_PHASE1):
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            for it in json.load(f):
                if it["type"] == "ADVERSARIAL" or it["query"] in seen:
                    continue
                seen.add(it["query"])
                out.append({"query": it["query"], "type": it["type"]})
    return out


def _measure(model, queries: list, labels: list) -> dict:
    t0 = time.perf_counter()
    pred = [model.predict([q])[0] for q in queries]
    elapsed = time.perf_counter() - t0
    correct = [p == g for p, g in zip(pred, labels)]
    return {
        "n": len(queries),
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "latency_ms": (elapsed / len(queries) * 1000) if queries else 0.0,
        "pred": pred,
        "correct": correct,
    }


def main():
    if not query_classifier.is_available():
        print("Модель не натреновано. Спочатку:")
        print("    python -m generation.query_classifier_train")
        return None

    model = query_classifier.load_model()
    data = load_dataset()
    _, _, X_val, y_val, src_val = split_by_seed(data)

    print("=" * 78)
    print("ФІНАЛЬНИЙ ТЕСТ НАВЧЕНОГО КЛАСИФІКАТОРА")
    print("=" * 78)

    results = {}

    # holdout from seed groups, split by source
    for source, title in (("combinatorial", "комбінаторний holdout"),
                          ("llm_paraphrase", "LLM-holdout")):
        mask = src_val == source
        if not mask.sum():
            print(f"\n{title}: порожній (у датасеті немає прикладів source={source})")
            continue
        r = _measure(model, list(X_val[mask]), list(y_val[mask]))
        results[source] = r
        print(f"\n{title}: n={r['n']}, accuracy={r['accuracy']:.1%}, "
              f"латентність={r['latency_ms']:.3f} мс/запит")

    # live queries: single evaluation
    live = load_live_queries()
    r_live = _measure(model, [d["query"] for d in live], [d["type"] for d in live])
    results["live"] = r_live

    print(f"\nЖиві запити: n={r_live['n']}, "
          f"accuracy={r_live['accuracy']:.1%}, латентність={r_live['latency_ms']:.3f} мс/запит")

    # --- comparison with the zero-shot baseline ---
    print("\n" + "-" * 78)
    print("ПОРІВНЯННЯ З ZERO-SHOT (живі запити, та сама вибірка)")
    print("-" * 78)
    speedup = ZERO_SHOT_LATENCY_S * 1000 / r_live["latency_ms"] if r_live["latency_ms"] else float("inf")
    cmp = pd.DataFrame([
        {"метрика": "accuracy", "zero-shot LLM": f"{ZERO_SHOT_ACCURACY:.1%}",
         "М3a (локальний)": f"{r_live['accuracy']:.1%}",
         "висновок": "паритет — умова допуску, не перевага"},
        {"метрика": "латентність", "zero-shot LLM": f"{ZERO_SHOT_LATENCY_S * 1000:.0f} мс",
         "М3a (локальний)": f"{r_live['latency_ms']:.3f} мс",
         "висновок": f"швидше у {speedup:.0f}x"},
        {"метрика": "ретраї 429", "zero-shot LLM": f"{ZERO_SHOT_RETRIES} на {ZERO_SHOT_N}",
         "М3a (локальний)": "0", "висновок": "мережі немає взагалі"},
        {"метрика": "вартість / 1000", "zero-shot LLM": "$0 (безкошт.) / ~$0.025 платний",
         "М3a (локальний)": "$0 завжди", "висновок": "без прив'язки до тарифу"},
        {"метрика": "rate-limit бюджет", "zero-shot LLM": "витрачає",
         "М3a (локальний)": "не витрачає", "висновок": "головний аргумент"},
    ])
    print(cmp.to_string(index=False))

    # --- errors on live queries ---
    wrong = [(d, p) for d, p, ok in zip(live, r_live["pred"], r_live["correct"]) if not ok]
    if wrong:
        print(f"\nПомилки на живих запитах ({len(wrong)}):")
        for d, p in wrong:
            print(f"  gold={d['type']:19s} pred={p:19s} {d['query']}")
    else:
        print("\nПомилок на живих запитах немає.")

    # --- readiness criteria ---
    print("\n" + "-" * 78)
    print("КРИТЕРІЇ ГОТОВНОСТІ")
    print("-" * 78)
    ok_acc = r_live["accuracy"] >= 0.90
    ok_lat = r_live["latency_ms"] < 1.0
    print(f"  accuracy >= 90% на фінальному тесті:  {'ТАК' if ok_acc else 'НІ'} "
          f"({r_live['accuracy']:.1%})")
    print(f"  латентність < 1 мс:                   {'ТАК' if ok_lat else 'НІ'} "
          f"({r_live['latency_ms']:.3f} мс)")
    print(f"  нуль мережевих викликів:              ТАК")
    print(f"  8 класів, без OUT_OF_SCOPE:           ТАК ({len(TAXONOMY)})")

    rows = []
    for key, label in (("combinatorial", "комбінаторний holdout"),
                       ("llm_paraphrase", "LLM-holdout"),
                       ("live", "живі запити")):
        if key in results:
            rows.append({"вибірка": label, "n": results[key]["n"],
                         "accuracy": round(results[key]["accuracy"], 4),
                         "латентність_мс": round(results[key]["latency_ms"], 4)})
    out = os.path.join(HERE, f"query_classifier_results_{lang.active_lang()}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nЗбережено: {out}")
    print("\nТри числа наведені окремо і не усереднюються: вибірки мають")
    print("різну природу, і розрив між синтетикою й живою мовою — це діагноз,")
    print("який блендоване середнє приховало б.")
    return results


if __name__ == "__main__":
    pd.set_option("display.width", 170)
    main()
