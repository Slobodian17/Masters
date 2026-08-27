"""End-to-end demonstration: PDF -> stitching -> retrieval -> repair loop.

Runs without an API key in MOCK mode. Part one shows the single-page document
with the deterministic repair, escalation and cross-table verification cases.
Part two uses the three-table document with a table split across pages, so the
retrieval decision is visible.
"""
from core.pipeline import run
from core import provenance
from load_data import load_tables
from retrieval.retrieve import explain, select_tables

PHASE0_QUERIES = [
    "Яка сума фінансових витрат за 2023 рік?",      # wrong row label -> deterministic repair
    "Яка маржа чистого прибутку (%) за Q4 2024?",   # inverted formula -> escalation
    "Які грошові кошти на кінець Q2 2024?",          # value present in both tables
]

PHASE1_QUERIES = [
    "Скільки виплачено дивідендів акціонерам за Q2 2023?",   # row from the stitched tail
    "Яка сума витрат на маркетинг за 2024 рік?",              # repair on the stitched table
    "Який власний капітал на Q2 2024?",                       # should route to balance
    "Скільки кварталів мали чистий рух грошових коштів за період більше 80000?",  # only reachable after stitching
]


def _show(query, tables, use_retrieval):
    names = select_tables(query, tables) if use_retrieval else None
    result = run(query, tables, retrieved_names=names, max_iter=4, threshold=0.7,
                 verbose=False, source_file="demo")
    scope = ",".join(names) if names else "(усі таблиці)"
    print("=" * 78)
    print("Запит:", query)
    print(f"  retrieval -> {scope}")
    print("  підсумок:", {k: v for k, v in result.items() if k not in ("trace", "provenance")})
    print("  трейс:")
    for step in result["trace"]:
        print("   ", step)
    return result


if __name__ == "__main__":
    print("#" * 78)
    print("# ЧАСТИНА 1: дві таблиці на одній сторінці")
    print("#" * 78)
    tables0 = load_tables("sample_report.pdf")
    print(f"Таблиці: { {k: v.shape for k, v in tables0.items()} }\n")
    for q in PHASE0_QUERIES:
        _show(q, tables0, use_retrieval=False)

    print("\n" + "#" * 78)
    print("# ЧАСТИНА 2: три таблиці, одна розірвана між сторінками")
    print("#" * 78)
    tables1 = load_tables("sample_report_multipage.pdf")
    print(f"Таблиці після зшивання: { {k: v.shape for k, v in tables1.items()} }\n")

    last = None
    for q in PHASE1_QUERIES:
        last = _show(q, tables1, use_retrieval=True)

    print("\n" + "=" * 78)
    print("Розкладка рішення retrieval на прикладі останнього запиту:")
    e = explain(PHASE1_QUERIES[-1], tables1)
    print(f"  обрано: {e['selected']}   (векторний бекенд: {e['vector_backend']})")
    print(f"  RRF: {[(n, round(s, 4)) for n, s in e['fused']]}")
    for source, detail in e["per_index"].items():
        print(f"  {source:11s}: " + ", ".join(f"{n}(ранг {d['rank']}, скор {d['score']})"
                                              for n, d in detail.items()))

    print("\n" + "=" * 78)
    print("Приклад Provenance-запису (останній запит):")
    print(provenance.to_json(last["provenance"]))
