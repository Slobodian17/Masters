"""Can a smaller model write runnable pandas at all?

A cheap gate before spending budget on a full paired run. The hypothesis being
tested downstream is that the method's value grows as the model weakens --
the baseline needs the model to do arithmetic in its head, the method only
needs it to write code and lets the sandbox compute. That hypothesis is only
testable on a model that can write the code; a model that cannot produce a
runnable program has nothing to trade.

Five questions per model, one call each. The check is deliberately weak: it
asks whether the code parses, runs and yields a number, NOT whether the number
is right. Correctness is what the full run measures.

Run:  python -m eval.finqa_model_check --models openai/gpt-oss-20b,qwen/qwen3.8-27b
"""
import argparse
import os
import time

import pandas as pd

from core.pipeline import generate_code
from core.sandbox import execute
from core.verifier import as_scalar
from eval.finqa_run import FINQA_INSTRUCTIONS, stratified_sample, table_only
from generation import llm_client
from ingestion import finqa_loader as fq

N_QUESTIONS = 5
SEED = 42


def check_model(model: str, records: list, max_tokens: int) -> dict:
    os.environ["VKSK_LLM_MODEL"] = model
    rows = []
    for rec in records:
        tables = fq.to_tables(rec)
        if not tables:
            continue
        llm_client.reset_stats()
        t0 = time.perf_counter()
        code, truncated, err = "", False, None
        result, exc = None, None
        try:
            code, truncated = generate_code(
                fq.question(rec), tables, instructions=FINQA_INSTRUCTIONS,
                max_tokens=max_tokens)
            result, exc = execute(code, tables)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        value = as_scalar(result) if result is not None else None
        rows.append({
            "model": model,
            "empty": not str(code).strip(),
            "truncated": truncated,
            "ran": exc is None and err is None,
            "exception": type(exc).__name__ if exc else (err or ""),
            "numeric": value is not None,
            "value": value,
            "gold": fq.gold_answer(rec),
            "tokens": llm_client.get_stats().get("total_tokens", 0),
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "code_head": (code or "").strip().split("\n")[0][:70],
        })
        time.sleep(2.0)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="openai/gpt-oss-20b,qwen/qwen3.8-27b,allam-2-7b")
    ap.add_argument("--max-tokens", type=int, default=2400)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if llm_client.active_provider() == "mock":
        print("MOCK не містить питань FinQA — потрібен GROQ_API_KEY.")
        return

    saved = os.environ.get("VKSK_LLM_MODEL")
    records = stratified_sample(table_only(fq.load_split(args.split)),
                                N_QUESTIONS, SEED)

    print("=" * 78)
    print("SANITY-CHECK: чи генерує менша модель робочий pandas-код")
    print("=" * 78)
    print(f"питань на модель: {len(records)}   max_tokens: {args.max_tokens}")
    print("Перевіряється ЗАПУСКАНІСТЬ, не правильність.")
    print()

    summary = []
    try:
        for m in models:
            try:
                df = check_model(m, records, args.max_tokens)
            except Exception as e:
                print(f"  {m}: НЕДОСТУПНА — {type(e).__name__}: {str(e)[:90]}")
                summary.append({"модель": m, "код видано": "—", "запустився": "—",
                                "число": "—", "вердикт": "недоступна"})
                continue
            n = len(df)
            gave = int((~df.empty_ if hasattr(df, "empty_") else ~df["empty"]).sum())
            ran = int(df["ran"].sum())
            num = int(df["numeric"].sum())
            verdict = ("годиться" if num >= 3 else
                       "сумнівно" if num >= 1 else "НЕ ГОДИТЬСЯ")
            summary.append({"модель": m, "код видано": f"{gave}/{n}",
                            "запустився": f"{ran}/{n}", "число": f"{num}/{n}",
                            "обірвано": int(df["truncated"].sum()),
                            "сер. токенів": int(df["tokens"].mean()),
                            "вердикт": verdict})
            print(f"--- {m} ---")
            for _, r in df.iterrows():
                mark = "OK " if r["numeric"] else "нi "
                print(f"  {mark} {r['exception'] or 'ok':16s} val={r['value']} "
                      f"gold={r['gold']}")
                print(f"      {r['code_head']}")
            print()
    finally:
        if saved is None:
            os.environ.pop("VKSK_LLM_MODEL", None)
        else:
            os.environ["VKSK_LLM_MODEL"] = saved

    print("=" * 78)
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print("«НЕ ГОДИТЬСЯ» означає, що гіпотезу Лінії 2 на цій моделі не можна")
    print("перевірити: їй нема чим торгувати, бо коду вона не пише.")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
