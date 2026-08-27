"""Generates the three-table test set.

Differs from the first set in three ways: three tables instead of two, a
`correct_tables` list per query, and 24 queries covering all eight types.

`correct_tables` is a list because some queries are legitimately answerable
from more than one table, and counting those as retrieval errors would be
wrong.

Reference answers are computed programmatically, as in the first set.
"""
import json
import os

from ingestion.synthetic_data import build_tables, build_cashflow

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_queries_phase1.json")

income, balance = build_tables()
cashflow = build_cashflow()

Q2023 = ["Q1_2023", "Q2_2023", "Q3_2023", "Q4_2023"]
Q2024 = ["Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024"]
ALL_Q = Q2023 + Q2024

_TABLES = {"income": income, "balance": balance, "cashflow": cashflow}


def val(table: str, row: str, col: str) -> float:
    df = _TABLES[table]
    key_col = df.columns[0]
    return float(df.loc[df[key_col] == row, col].values[0])


items = []


def add(query, qtype, expected, correct_tables, note=""):
    items.append({
        "query": query,
        "type": qtype,
        "expected": round(float(expected), 2) if expected is not None else None,
        "correct_tables": correct_tables,
        "note": note,
    })


# ---- income (8 запитів, усі 8 типів представлені в наборі загалом) ----
add("Яка виручка за Q3 2024?", "SCALAR_LOOKUP", val("income", "Виручка", "Q3_2024"), ["income"])
add("Який чистий прибуток за Q1 2023?", "SCALAR_LOOKUP", val("income", "Чистий прибуток", "Q1_2023"), ["income"])
add("Яка сукупна виручка за 2023 рік?", "AGGREGATE_SUM",
    sum(val("income", "Виручка", q) for q in Q2023), ["income"])
add("Який середній чистий прибуток за 2023 рік?", "AGGREGATE_MEAN",
    sum(val("income", "Чистий прибуток", q) for q in Q2023) / 4, ["income"])
add("Який максимальний валовий прибуток серед усіх кварталів?", "FIND_MAX_MIN",
    max(val("income", "Валовий прибуток", q) for q in ALL_Q), ["income"])
add("Яка різниця між чистим прибутком Q1 2024 і Q1 2023?", "COMPARE_TWO",
    val("income", "Чистий прибуток", "Q1_2024") - val("income", "Чистий прибуток", "Q1_2023"), ["income"])
add("На скільки відсотків зросла виручка з Q1 2023 по Q1 2024?", "GROWTH_RATE",
    (val("income", "Виручка", "Q1_2024") - val("income", "Виручка", "Q1_2023"))
    / val("income", "Виручка", "Q1_2023") * 100, ["income"])
add("Яка маржа чистого прибутку (%) за Q4 2024?", "RATIO",
    val("income", "Чистий прибуток", "Q4_2024") / val("income", "Виручка", "Q4_2024") * 100, ["income"],
    note="навмисно провокує L2: перша спроба у MOCK-режимі інвертує формулу")

# ---- balance (6 запитів) ----
add("Який власний капітал на Q2 2024?", "SCALAR_LOOKUP", val("balance", "Власний капітал", "Q2_2024"), ["balance"])
add("Скільки всього активів на Q4 2023?", "SCALAR_LOOKUP", val("balance", "Всього активів", "Q4_2023"), ["balance"])
add("На скільки зросли всього активів з Q1 2023 по Q4 2024?", "COMPARE_TWO",
    val("balance", "Всього активів", "Q4_2024") - val("balance", "Всього активів", "Q1_2023"), ["balance"])
add("Яка частка власного капіталу у всього активів (%) на Q3 2024?", "RATIO",
    val("balance", "Власний капітал", "Q3_2024") / val("balance", "Всього активів", "Q3_2024") * 100, ["balance"])
add("Яка максимальна дебіторська заборгованість серед усіх кварталів?", "FIND_MAX_MIN",
    max(val("balance", "Дебіторська заборгованість", q) for q in ALL_Q), ["balance"])
add("Яка середня величина всього зобов'язань за 2024 рік?", "AGGREGATE_MEAN",
    sum(val("balance", "Всього зобов'язань", q) for q in Q2024) / 4, ["balance"])

# ---- cashflow (8 запитів; таблиця, розірвана між сторінками -> залежить від М1) ----
add("Скільки виплачено дивідендів акціонерам за Q2 2023?", "SCALAR_LOOKUP",
    val("cashflow", "Виплачені дивіденди акціонерам", "Q2_2023"), ["cashflow"])
add("Який чистий рух коштів від операційної діяльності за Q4 2024?", "SCALAR_LOOKUP",
    val("cashflow", "Чистий рух коштів від операційної діяльності", "Q4_2024"), ["cashflow"])
add("Яка сума витрат на маркетинг за 2024 рік?", "AGGREGATE_SUM",
    sum(val("cashflow", "Витрати на маркетинг", q) for q in Q2024), ["cashflow"])
add("Скільки всього сплачено відсотків за кредитами за 2023 рік?", "AGGREGATE_SUM",
    sum(val("cashflow", "Сплачені відсотки за кредитами", q) for q in Q2023), ["cashflow"])
add("Які найбільші надходження від реалізації продукції серед усіх кварталів?", "FIND_MAX_MIN",
    max(val("cashflow", "Надходження від реалізації продукції", q) for q in ALL_Q), ["cashflow"])
add("Скільки кварталів мали чистий рух грошових коштів за період більше 80000?", "CONDITIONAL_FILTER",
    sum(1 for q in ALL_Q if val("cashflow", "Чистий рух грошових коштів за період", q) > 80000), ["cashflow"],
    note="показник лежить у РЯДКУ 27 — доступний лише якщо М1 зшив таблицю")
add("Який середній чистий рух коштів від інвестиційної діяльності за 2023 рік?", "AGGREGATE_MEAN",
    sum(val("cashflow", "Чистий рух коштів від інвестиційної діяльності", q) for q in Q2023) / 4, ["cashflow"])
add("На скільки відсотків зросли надходження від реалізації продукції з Q1 2023 по Q4 2024?", "GROWTH_RATE",
    (val("cashflow", "Надходження від реалізації продукції", "Q4_2024")
     - val("cashflow", "Надходження від реалізації продукції", "Q1_2023"))
    / val("cashflow", "Надходження від реалізації продукції", "Q1_2023") * 100, ["cashflow"])

# ---- крос-табличний та adversarial ----
add("Які грошові кошти на кінець Q2 2024?", "SCALAR_LOOKUP",
    val("balance", "Грошові кошти", "Q2_2024"), ["income", "balance", "cashflow"],
    note="те саме число присутнє в усіх трьох таблицях — саме тому correct_tables це СПИСОК; "
         "перевіряє і v_cross_table")
add("Яка рентабельність по сегменту B2B за МСФЗ 15 у Q2 2023?", "ADVERSARIAL", None, [],
    note="таких даних немає в жодній таблиці — перевірка чесного ABSTAIN; "
         "у recall@1 не враховується (немає правильної відповіді)")


if __name__ == "__main__":
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Записано {len(items)} запитів у {OUT_PATH}")
    by_table = {}
    for it in items:
        key = ",".join(it["correct_tables"]) or "(немає)"
        by_table[key] = by_table.get(key, 0) + 1
    print("Розподіл за очікуваними таблицями:", by_table)
    types = sorted({it["type"] for it in items})
    print(f"Таксономічних типів: {len(types)} -> {types}")
