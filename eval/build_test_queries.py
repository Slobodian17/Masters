"""Generates the first test set covering all eight operation types.

Reference answers are computed programmatically from the same data the
pipeline sees, rather than written by hand, so the test set cannot itself
contain arithmetic errors.
"""
import json
import os

from ingestion.synthetic_data import build_tables

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_queries_full.json")

income, balance = build_tables()
Q2023 = ["Q1_2023", "Q2_2023", "Q3_2023", "Q4_2023"]
Q2024 = ["Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024"]
ALL_Q = Q2023 + Q2024


def scalar(row, col, table=income):
    return float(table.loc[table["Показник"] == row, col].values[0])


items = []


def add(query, qtype, expected, note=""):
    items.append({
        "query": query,
        "type": qtype,
        "expected": round(float(expected), 2) if expected is not None else None,
        "note": note,
    })


add("Яка виручка за Q3 2024?", "SCALAR_LOOKUP", scalar("Виручка", "Q3_2024"))
add("Який чистий прибуток за Q1 2023?", "SCALAR_LOOKUP", scalar("Чистий прибуток", "Q1_2023"))
add("Яка сукупна виручка за 2023 рік?", "AGGREGATE_SUM", sum(scalar("Виручка", q) for q in Q2023))
add("Яка сумарна собівартість реалізації за 2024 рік?", "AGGREGATE_SUM", sum(scalar("Собівартість реалізації", q) for q in Q2024))
add("Який середній чистий прибуток за 2023 рік?", "AGGREGATE_MEAN", sum(scalar("Чистий прибуток", q) for q in Q2023) / 4)
add("Яка середня виручка за всі 8 кварталів?", "AGGREGATE_MEAN", sum(scalar("Виручка", q) for q in ALL_Q) / 8)
add("Який максимальний чистий прибуток серед усіх кварталів?", "FIND_MAX_MIN", max(scalar("Чистий прибуток", q) for q in ALL_Q))
add("Яка мінімальна виручка серед усіх кварталів?", "FIND_MAX_MIN", min(scalar("Виручка", q) for q in ALL_Q))
add("Наскільки виручка Q4 2024 більша за Q4 2023 в абсолютних цифрах?", "COMPARE_TWO", scalar("Виручка", "Q4_2024") - scalar("Виручка", "Q4_2023"))
add("Яка різниця між чистим прибутком Q1 2024 і Q1 2023?", "COMPARE_TWO", scalar("Чистий прибуток", "Q1_2024") - scalar("Чистий прибуток", "Q1_2023"))
add("На скільки відсотків зросла виручка з Q1 2023 по Q1 2024?", "GROWTH_RATE",
    (scalar("Виручка", "Q1_2024") - scalar("Виручка", "Q1_2023")) / scalar("Виручка", "Q1_2023") * 100)
add("Який приріст чистого прибутку з Q3 2023 по Q3 2024 у відсотках?", "GROWTH_RATE",
    (scalar("Чистий прибуток", "Q3_2024") - scalar("Чистий прибуток", "Q3_2023")) / scalar("Чистий прибуток", "Q3_2023") * 100)
add("Скільки кварталів мали чистий прибуток більше 150000?", "CONDITIONAL_FILTER",
    sum(1 for q in ALL_Q if scalar("Чистий прибуток", q) > 150000))
add("Яка сумарна виручка за квартали, де чистий прибуток перевищив 200000?", "CONDITIONAL_FILTER",
    sum(scalar("Виручка", q) for q in ALL_Q if scalar("Чистий прибуток", q) > 200000))
add("Які грошові кошти на кінець Q2 2024?", "SCALAR_LOOKUP", scalar("Грошові кошти на кінець кварталу", "Q2_2024"),
    note="перевіряє v_cross_table — те саме число присутнє і в таблиці balance")
add("Яка маржа чистого прибутку (%) за Q4 2024?", "RATIO",
    scalar("Чистий прибуток", "Q4_2024") / scalar("Виручка", "Q4_2024") * 100,
    note="навмисно провокує L2: перша спроба у MOCK-режимі інвертує формулу")
add("Яка сума фінансових витрат за 2023 рік?", "AGGREGATE_SUM", sum(scalar("Фінансові витрати", q) for q in Q2023),
    note="навмисно провокує L0: перша спроба у MOCK-режимі містить помилку в назві рядка")
add("Яка рентабельність по сегменту B2B за МСФЗ 15 у Q2 2023?", "ADVERSARIAL", None,
    note="таких даних у таблиці немає — перевірка чесного ABSTAIN")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print(f"Записано {len(items)} запитів у {OUT_PATH}")
for it in items:
    print(f"  [{it['type']}] {it['query']} -> {it['expected']}")
