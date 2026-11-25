"""Configuration A (LLM alone) against configuration E (full pipeline).

Also collects episodes and fits the calibrator on them.

Results go to a provider-specific file for real runs, so the frozen MOCK
baseline cannot be overwritten by accident.
"""
import json
import os

import pandas as pd

from core import calibration
from core import lang
from core.pipeline import run
from generation import llm_client
from generation.llm_client import call_llm
from load_data import load_tables

HERE = os.path.dirname(os.path.abspath(__file__))


def test_set_path() -> str:
    """Test set for the active language, next to this module."""
    return os.path.join(HERE, lang.get().TEST_SET_FULL)


def load_test_set(path=None):
    with open(path or test_set_path(), encoding="utf-8") as f:
        return json.load(f)


def baseline_answer(query, tables):
    schema = {name: list(df.columns) for name, df in tables.items()}
    prompt = f"Схема таблиць: {schema}\nЗапит: {query}\nДай ТІЛЬКИ число-відповідь, без пояснень і без одиниць виміру."
    return call_llm(prompt, tag="baseline")


def is_correct(predicted, expected, tol=0.01):
    if expected is None:
        return None  # adversarial query: no reference answer
    try:
        p, e = float(predicted), float(expected)
        return abs(p - e) <= tol * max(abs(e), 1)
    except (TypeError, ValueError):
        return False


def output_path(provider: str, model: str, code: str = None) -> str:
    """Output file name, keyed by provider and language.

    Two separate ways to destroy the frozen baseline are closed here. A real
    key turns the same command into a different experiment, and so does a
    different language profile; either would otherwise overwrite
    eval_results.csv with numbers that are not comparable to it.
    """
    code = code or lang.active_lang()
    if provider == "mock":
        return "eval_results.csv" if code == "uk" else f"eval_results_{code}.csv"
    slug = model.replace("/", "-").replace(".", "-")
    return f"eval_results_{code}_{provider}_{slug}.csv"


def _call_row(item: dict, result: dict, stats: dict) -> dict:
    """One row of the per-query API breakdown.

    Successful calls and failed ones are counted separately: a call that raised
    still reached the provider and still consumed the request budget.
    """
    by = stats.get("by_tag", {})

    def g(tag, field, default=0):
        return by.get(tag, {}).get(field, default)

    codegen = g("codegen", "calls")
    semantic = g("semantic", "calls")
    baseline = g("baseline", "calls")
    failed = sum(v.get("failed_calls", 0) for v in by.values())

    # A budget retry is a full call to the model. It carries the "codegen"
    # tag, so it is already inside `codegen`; counting it again here would
    # double it. Reported separately because the escalation level it belongs
    # to (LB) is deliberately not a repair.
    truncation_retries = sum(1 for st in result.get("trace", [])
                             if st.get("level") == "LB")

    return {
        "type": item["type"],
        "query": item["query"],
        "status": result["status"],
        "iterations": len(result["trace"]),
        "calls_codegen": codegen,
        "calls_truncation_retry": truncation_retries,
        "calls_semantic": semantic,
        "calls_method_total": codegen + semantic,
        "calls_baseline": baseline,
        "calls_failed": failed,
        "retries": stats.get("retries", 0),
        "api_requests_total": codegen + semantic + baseline + failed + stats.get("retries", 0),
        "prompt_tokens": stats.get("prompt_tokens", 0),
        "completion_tokens": stats.get("completion_tokens", 0),
        "elapsed_s": round(stats.get("elapsed_s", 0.0), 3),
    }


def run_eval(pdf_path="sample_report.pdf", data_path=None, threshold=0.7):
    provider = llm_client.active_provider()
    model = llm_client.active_model()
    has_key = provider != "mock"
    out_path = output_path(provider, model)

    print(f"Мова: {lang.active_lang()}   провайдер: {provider}   модель: {model}")
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

    rows, episodes, call_rows = [], [], []
    for item in test_set:
        q, expected = item["query"], item["expected"]

        # Per-query API accounting: reset before the query, read after, so each
        # row shows exactly what that one query cost.
        llm_client.reset_stats()

        if has_key:
            b = baseline_answer(q, tables)
            b_ok = is_correct(b, expected)
        else:
            b, b_ok = "N/A (потрібен ключ)", None

        m = run(q, tables, threshold=threshold, verbose=False, source_file=pdf_path)
        m_answer = m.get("answer") if m["status"] == "ACCEPT" else "ABSTAIN"
        m_ok = is_correct(m_answer, expected) if m["status"] == "ACCEPT" else (expected is None)

        call_rows.append(_call_row(item, m, llm_client.get_stats()))

        if m["status"] == "ACCEPT" and expected is not None:
            episodes.append({"signals": m["trace"][-1]["signals"], "correct": bool(m_ok)})

        rows.append({
            "type": item["type"], "query": q, "expected": expected,
            "baseline": b, "baseline_correct": b_ok,
            "method": m_answer, "method_correct": m_ok,
            "status": m["status"], "iterations": len(m["trace"]),
            "used_llm_repair": any(s.get("level") in ("L1", "L2", "L3") for s in m["trace"]),
        })

    return pd.DataFrame(rows), episodes, has_key, out_path, pd.DataFrame(call_rows)


if __name__ == "__main__":
    results, episodes, has_key, out_path, calls = run_eval()

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

    # Per-query API breakdown goes to its own file, so the frozen baseline CSV
    # keeps its exact schema.
    print()
    print("--- Звернення до LLM API по кожному запиту ---")
    show = ["type", "status", "iterations", "calls_codegen", "calls_semantic",
            "calls_method_total", "calls_baseline", "calls_failed", "retries",
            "api_requests_total"]
    print(calls[show].to_string())
    print()
    print(f"СУМА по {len(calls)} запитах:")
    print(f"  генерація коду:            {int(calls.calls_codegen.sum())}")
    print(f"  семантична перевірка:      {int(calls.calls_semantic.sum())}")
    print(f"  метод (E) разом:           {int(calls.calls_method_total.sum())}")
    print(f"  baseline (A):              {int(calls.calls_baseline.sum())}")
    print(f"  невдалі виклики:           {int(calls.calls_failed.sum())}")
    print(f"  ретраї (429/5xx):          {int(calls.retries.sum())}")
    print(f"  УСЬОГО HTTP-запитів:       {int(calls.api_requests_total.sum())}")
    print()
    print("СЕРЕДНЄ на запит:")
    print(f"  метод (E):                 {calls.calls_method_total.mean():.2f}")
    print(f"  усього HTTP-запитів:       {calls.api_requests_total.mean():.2f}")

    calls_path = os.path.join(
        HERE,
        f"llm_calls_{lang.active_lang()}_{llm_client.active_provider()}_"
        f"{llm_client.active_model().replace('/', '-').replace('.', '-')}.csv")
    calls.to_csv(calls_path, index=False)
    print()
    print('Збережено:', calls_path)
