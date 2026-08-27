"""Evaluates table stitching and hybrid retrieval.

Measures: recall@1 of the fused ranking, recall@1 of each index alone, the
stitching ablation on both page-break variants, and an end-to-end run with and
without retrieval.

Sections 1 and 2 use no LLM at all and give identical numbers with or without
a key; only the end-to-end section depends on the provider.
"""
import json
import os

import pandas as pd

from core.pipeline import run
from generation import llm_client
from ingestion.ingest_pdf import extract_tables
from ingestion.synthetic_data import build_cashflow
from load_data import load_tables
from retrieval import structural_index, temporal_index, vector_index
from retrieval.retrieve import explain, select_tables

PDF = "sample_report_multipage.pdf"
TEST_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_queries_phase1.json")

SINGLE_INDEXES = {
    "structural": {"structural": structural_index.rank},
    "temporal": {"temporal": temporal_index.rank},
    "vector": {"vector": vector_index.rank},
}


def load_test_set(path=None):
    with open(path or TEST_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def recall_at_1(test_set, tables, indexes=None) -> tuple:
    """Returns (recall@1, misses); queries without a reference table are
    excluded."""
    scored = [it for it in test_set if it["correct_tables"]]
    hits, misses = 0, []
    for it in scored:
        picked = select_tables(it["query"], tables, indexes=indexes)
        if picked and picked[0] in it["correct_tables"]:
            hits += 1
        else:
            misses.append({"query": it["query"], "picked": picked,
                           "expected": it["correct_tables"]})
    return hits / len(scored), misses


STITCH_FIXTURES = {
    "розрив із повтореною шапкою": "sample_report_multipage.pdf",
    "розрив без шапки": "sample_report_multipage_noheader.pdf",
}


def stitching_ablation() -> pd.DataFrame:
    """Stitching ablation on both break variants.

    Two distinct harms are measured. Rows are physically lost only when the
    header is not repeated. Reachability is lost in both cases: without
    stitching the table splits in two, and labels in the tail are invisible to
    a query that sees only the head.
    """
    ref_labels = set(build_cashflow().iloc[:, 0])
    test_set = load_test_set()
    rows = []

    for label, path in STITCH_FIXTURES.items():
        off = extract_tables(path, stitch=False)
        on = extract_tables(path, stitch=True)

        stitched = max(on, key=lambda t: t.shape[0])
        head = max(off, key=lambda t: t.shape[0])
        head_labels = set(head.iloc[:, 0])

        # queries referring to a label absent from the largest raw fragment
        unreachable = 0
        for it in test_set:
            if it["correct_tables"] != ["cashflow"]:
                continue
            referenced = [lbl for lbl in ref_labels if str(lbl).lower() in it["query"].lower()]
            if referenced and not any(lbl in head_labels for lbl in referenced):
                unreachable += 1

        rows.append({
            "варіант розриву": label,
            "фрагментів без М1": len(off),
            "таблиць після М1": len(on),
            "рядків без М1": int(sum(t.shape[0] for t in off)),
            "рядків після М1": int(sum(t.shape[0] for t in on)),
            "рядків втрачено без М1": int(sum(t.shape[0] for t in on) - sum(t.shape[0] for t in off)),
            "запитів недосяжно без М1": unreachable,
        })
    return pd.DataFrame(rows)


def pipeline_ablation(test_set, tables, threshold=0.7) -> pd.DataFrame:
    """End-to-end run with retrieval (narrowed scope) and without."""
    rows = []
    for it in test_set:
        q, expected = it["query"], it["expected"]

        for mode in ("з retrieval", "без retrieval"):
            names = select_tables(q, tables) if mode == "з retrieval" else None
            r = run(q, tables, retrieved_names=names, threshold=threshold,
                    verbose=False, source_file=PDF)
            answer = r.get("answer") if r["status"] == "ACCEPT" else "ABSTAIN"
            ok = _is_correct(answer, expected) if r["status"] == "ACCEPT" else (expected is None)
            rows.append({
                "mode": mode, "type": it["type"], "query": q,
                "scope": ",".join(names) if names else "(усі)",
                "expected": expected, "answer": answer, "correct": ok,
                "status": r["status"], "iterations": len(r["trace"]),
                "used_llm_repair": any(s.get("level") in ("L1", "L2", "L3") for s in r["trace"]),
            })
    return pd.DataFrame(rows)


def _is_correct(predicted, expected, tol=0.01):
    if expected is None:
        return None
    try:
        p, e = float(predicted), float(expected)
        return abs(p - e) <= tol * max(abs(e), 1)
    except (TypeError, ValueError):
        return False


def main():
    test_set = load_test_set()
    tables = load_tables(PDF)

    print("=" * 78)
    print("ФАЗА 1 — ОЦІНКА М1 (Table Stitching) + М2 (гібридний retrieval)")
    print("=" * 78)
    print(f"Документ: {PDF}")
    print(f"Таблиць у документі: {len(tables)} -> {({k: v.shape for k, v in tables.items()})}")
    print(f"Запитів у наборі: {len(test_set)} (з них з еталонною таблицею: "
          f"{sum(1 for i in test_set if i['correct_tables'])})")
    print(f"Векторний бекенд: {vector_index.backend_name()}")
    prov = llm_client.active_provider()
    print(f"LLM-провайдер: {prov}   модель: {llm_client.active_model()}")
    if prov != "mock":
        print("  Увага: наскрізний розділ піде в реальну LLM. Еталонні числа")
        print("  зміряні в MOCK: VKSK_LLM_PROVIDER=mock python -m eval.eval_retrieval")

    print("\n--- 1. М1: ablation зшивання (обидва варіанти розриву) ---")
    print(stitching_ablation().to_string(index=False))

    print("\n--- 2. М2: recall@1 ---")
    r_fused, misses = recall_at_1(test_set, tables)
    per_index = {}
    for label, idx in SINGLE_INDEXES.items():
        r, _ = recall_at_1(test_set, tables, indexes=idx)
        per_index[label] = r
        print(f"  тільки {label:11s}: recall@1 = {r:.0%}")
    print(f"  RRF-фузія (усі 3): recall@1 = {r_fused:.0%}   <-- критерій готовності Фази 1")

    if misses:
        print("\n  Промахи retrieval:")
        for m in misses:
            print(f"    - {m['query']}")
            print(f"      обрано={m['picked']}, очікувалось={m['expected']}")
    else:
        print("\n  Промахів немає.")

    print("\n--- 3. Наскрізний ВКСК: з retrieval проти без ---")
    df = pipeline_ablation(test_set, tables)
    scored = df[df["expected"].notna()]
    summary = []
    for mode in ("без retrieval", "з retrieval"):
        s = scored[scored["mode"] == mode]
        d = df[df["mode"] == mode]
        summary.append({
            "режим": mode,
            "accuracy": f"{s['correct'].mean():.0%}",
            "% без LLM-репейру": f"{(~d['used_llm_repair']).mean():.0%}",
            "сер. ітерацій": round(d["iterations"].mean(), 2),
            "ABSTAIN": f"{(d['status'] == 'ABSTAIN').sum()}/{len(d)}",
        })
    print(pd.DataFrame(summary).to_string(index=False))

    wrong = scored[scored["correct"] == False]  # noqa: E712
    if len(wrong):
        print("\n  Неправильні відповіді:")
        for _, r in wrong.iterrows():
            print(f"    [{r['mode']}] {r['query']}")
            print(f"      scope={r['scope']}, отримано={r['answer']}, еталон={r['expected']}")

    provider = llm_client.active_provider()
    slug = "" if provider == "mock" else f"_{provider}_{llm_client.active_model().replace('/', '-').replace('.', '-')}"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"eval_phase1_results{slug}.csv")
    df.to_csv(out, index=False)
    print(f"\nЗбережено: {out}")

    print("\n--- Підсумок за критерієм готовності Фази 1 ---")
    ok_tables = len(tables) >= 2
    ok_queries = sum(1 for i in test_set if i["correct_tables"]) >= 20
    ok_recall = r_fused >= 1.0
    print(f"  таблиць у документі >= 2:        {'ТАК' if ok_tables else 'НІ'} ({len(tables)})")
    print(f"  запитів з еталоном >= 20:        {'ТАК' if ok_queries else 'НІ'}")
    print(f"  retrieval знаходить потрібну:    {'ТАК' if ok_recall else 'НІ'} (recall@1={r_fused:.0%})")
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 170)
    pd.set_option("display.max_colwidth", 50)
    main()
