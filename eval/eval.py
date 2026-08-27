"""Configuration A (LLM alone) against configuration E (full pipeline).

Also collects episodes and fits the calibrator on them.

Results go to a provider-specific file for real runs, so the frozen MOCK
baseline cannot be overwritten by accident.
"""
import json
import os

import pandas as pd

from core import calibration
from core.pipeline import run
from generation import llm_client
from generation.llm_client import call_llm
from load_data import load_tables

# Test set lives next to this module, not in the working directory.
TEST_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_queries_full.json")


def load_test_set(path=None):
    with open(path or TEST_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def baseline_answer(query, tables):
    schema = {name: list(df.columns) for name, df in tables.items()}
    prompt = f"Схема таблиць: {schema}\nЗапит: {query}\nДай ТІЛЬКИ число-відповідь, без пояснень і без одиниць виміру."
    return call_llm(prompt)


def is_correct(predicted, expected, tol=0.01):
    if expected is None:
        return None  # adversarial query: no reference answer
    try:
        p, e = float(predicted), float(expected)
        return abs(p - e) <= tol * max(abs(e), 1)
    except (TypeError, ValueError):
        return False


def output_path(provider: str, model: str) -> str:
    """Output file name, keyed by provider.

    The frozen baseline was taken in MOCK mode; once a real key is present the
    same command would otherwise overwrite it with different numbers and
    silently destroy the regression check.
    """
    if provider == "mock":
        return "eval_results.csv"
    slug = model.replace("/", "-").replace(".", "-")
    return f"eval_results_{provider}_{slug}.csv"


def run_eval(pdf_path="sample_report.pdf", data_path=None, threshold=0.7):
    provider = llm_client.active_provider()
    model = llm_client.active_model()
    has_key = provider != "mock"
    out_path = output_path(provider, model)

    print(f"Провайдер: {provider}   модель: {model}")
    if not has_key:
        print('MOCK-режим (немає ключа провайдера або VKSK_LLM_PROVIDER=mock).')
        print('Порівняння з baseline (конфігурація A) позначається як N/A:')
        print('MOCK не є реальною LLM, тож наївне вгадування неможливо чесно')
        print('змоделювати. Метрики методу лишаються змістовними, бо перевіряють')
        print('sandbox / verifier / escalation.')
        print('ЦЕ Й Є ЗАМОРОЖЕНИЙ BASELINE ФАЗИ 0.')
    else:
        print(f'РЕАЛЬНИЙ ПРОГІН -> {out_path}')
        print('Заморожений eval_results.csv не чіпається.')
    print()

    tables = load_tables(pdf_path)
    test_set = load_test_set(data_path)

    rows, episodes = [], []
    for item in test_set:
        q, expected = item["query"], item["expected"]

        if has_key:
            b = baseline_answer(q, tables)
            b_ok = is_correct(b, expected)
        else:
            b, b_ok = "N/A (потрібен ключ)", None

        m = run(q, tables, threshold=threshold, verbose=False, source_file=pdf_path)
        m_answer = m.get("answer") if m["status"] == "ACCEPT" else "ABSTAIN"
        m_ok = is_correct(m_answer, expected) if m["status"] == "ACCEPT" else (expected is None)

        if m["status"] == "ACCEPT" and expected is not None:
            episodes.append({"signals": m["trace"][-1]["signals"], "correct": bool(m_ok)})

        rows.append({
            "type": item["type"], "query": q, "expected": expected,
            "baseline": b, "baseline_correct": b_ok,
            "method": m_answer, "method_correct": m_ok,
            "status": m["status"], "iterations": len(m["trace"]),
            "used_llm_repair": any(s.get("level") in ("L1", "L2", "L3") for s in m["trace"]),
        })

    return pd.DataFrame(rows), episodes, has_key, out_path


if __name__ == "__main__":
    results, episodes, has_key, out_path = run_eval()

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 45)
    cols = ["type", "query", "expected", "baseline", "baseline_correct", "method", "method_correct", "status", "iterations"]
    print(results[cols].to_string(index=False))

    scored = results[results["expected"].notna()]
    print("\n--- Агрегатні метрики (конфігурації A vs E, розділ 9 специфікації) ---")
    if has_key:
        print(f"Baseline accuracy (A):      {scored['baseline_correct'].mean():.0%}")
    else:
        print('Baseline accuracy (A):      N/A — MOCK-режим; для конфігурації A потрібен GROQ_API_KEY')
    print(f"Method accuracy (E):        {scored['method_correct'].mean():.0%}")
    print(f"% вирішено без LLM-репейру: {(~results['used_llm_repair']).mean():.0%}   (H2: вартість)")
    print(f"Середня к-сть ітерацій:     {results['iterations'].mean():.1f}")
    print(f"ABSTAIN:                    {(results['status'] == 'ABSTAIN').sum()} / {len(results)}  (в т.ч. 1 навмисний adversarial-запит)")

    print("\n--- Тренування калібрації φ(v) на зібраних епізодах (H3) ---")
    print(f"Зібрано епізодів: {len(episodes)}")
    metrics = calibration.fit(episodes)
    print(metrics)

    results.to_csv(out_path, index=False)
    print()
    print('Збережено:', out_path)
