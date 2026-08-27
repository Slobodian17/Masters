"""Synthetic financial dataset: income statement, balance sheet, cash flow.

Values are computed rather than written by hand, so the reference answers in
the test sets are arithmetically consistent by construction. Tables use a plain
RangeIndex to match what the PDF parser returns.
"""
import numpy as np
import pandas as pd

QUARTERS = ["Q1_2023", "Q2_2023", "Q3_2023", "Q4_2023", "Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024"]


def _base_series(seed: int = 42) -> dict:
    """Shared source of numbers for every table in the document.

    Cash flow is derived from the same series as income and balance, otherwise
    the cross-table verification signal would be meaningless.
    """
    rng = np.random.default_rng(seed)

    base_revenue = 1_000_000
    growth = np.array([1.00, 1.04, 1.07, 1.15, 1.18, 1.23, 1.27, 1.36])  # Q4 seasonal peak
    noise = rng.normal(1.0, 0.01, size=8)
    revenue = np.round(base_revenue * growth * noise, -3)

    cogs_ratio = 0.62
    opex_ratio = 0.16
    fin_expenses = np.round(np.linspace(18_000, 26_000, 8), -2)
    tax_rate = 0.18

    cogs = np.round(revenue * cogs_ratio, -3)
    gross_profit = revenue - cogs
    opex = np.round(revenue * opex_ratio, -3)
    op_profit = gross_profit - opex
    pretax_profit = op_profit - fin_expenses
    tax = np.round(pretax_profit * tax_rate, -2)
    net_profit = pretax_profit - tax

    cash_start = 500_000
    cash = [cash_start]
    for np_ in net_profit:
        cash.append(np.round(cash[-1] + np_ * 0.5, -3))  # half of profit retained
    cash_end = np.array(cash[1:])

    return {
        "revenue": revenue, "cogs": cogs, "gross_profit": gross_profit, "opex": opex,
        "op_profit": op_profit, "fin_expenses": fin_expenses, "pretax_profit": pretax_profit,
        "tax": tax, "net_profit": net_profit, "cash_end": cash_end,
    }


def build_tables(seed: int = 42):
    s = _base_series(seed)
    revenue, cogs, gross_profit, opex = s["revenue"], s["cogs"], s["gross_profit"], s["opex"]
    op_profit, fin_expenses, pretax_profit = s["op_profit"], s["fin_expenses"], s["pretax_profit"]
    tax, net_profit, cash_end = s["tax"], s["net_profit"], s["cash_end"]

    income_stmt = pd.DataFrame(
        {
            "Показник": [
                "Виручка", "Собівартість реалізації", "Валовий прибуток",
                "Операційні витрати", "Прибуток від операційної діяльності",
                "Фінансові витрати", "Прибуток до оподаткування",
                "Податок на прибуток", "Чистий прибуток",
                "Грошові кошти на кінець кварталу",
            ]
        }
    )
    values = [revenue, cogs, gross_profit, opex, op_profit, fin_expenses, pretax_profit, tax, net_profit, cash_end]
    for i, q in enumerate(QUARTERS):
        income_stmt[q] = [v[i] for v in values]

    receivables = np.round(revenue * 0.20, -3)
    inventory = np.round(cogs * 0.10, -3)
    fixed_assets = np.round(np.linspace(2_000_000, 2_300_000, 8), -3)
    total_assets = cash_end + receivables + inventory + fixed_assets

    current_liab = np.round(total_assets * 0.18, -3)
    lt_liab = np.round(total_assets * 0.22, -3)
    total_liab = current_liab + lt_liab
    equity = total_assets - total_liab

    balance = pd.DataFrame(
        {
            "Показник": [
                "Грошові кошти", "Дебіторська заборгованість", "Всього активів",
                "Всього зобов'язань", "Власний капітал",
            ]
        }
    )
    bal_values = [cash_end, receivables, total_assets, total_liab, equity]
    for i, q in enumerate(QUARTERS):
        balance[q] = [v[i] for v in bal_values]

    # RangeIndex on purpose, matching what the PDF parser produces.
    return income_stmt, balance


# Cash flow statement. It deliberately shares vocabulary with the income
# statement, which makes retrieval non-trivial, and it is long enough to split
# across pages, which makes it the stitching fixture.

_CASHFLOW_ROWS = [
    # (label, base series key, multiplier)
    ("Надходження від реалізації продукції", "revenue", 0.95),
    ("Надходження від оренди", "revenue", 0.012),
    ("Інші операційні надходження", "revenue", 0.008),
    ("Оплата постачальникам сировини", "cogs", -0.72),
    ("Оплата постачальникам послуг", "cogs", -0.18),
    ("Виплати персоналу", "opex", -0.55),
    ("Відрахування на соціальні заходи", "opex", -0.12),
    ("Оплата оренди приміщень", "opex", -0.09),
    ("Витрати на маркетинг", "opex", -0.14),
    ("Сплата податку на прибуток", "tax", -1.0),
    ("Сплачені відсотки за кредитами", "fin_expenses", -1.0),
    ("Чистий рух коштів від операційної діяльності", None, None),  # підсумок
    ("Придбання основних засобів", "revenue", -0.11),
    ("Придбання нематеріальних активів", "revenue", -0.015),
    ("Надходження від продажу обладнання", "revenue", 0.006),
    ("Надання позик пов'язаним особам", "revenue", -0.02),
    ("Повернення раніше наданих позик", "revenue", 0.011),
    ("Отримані дивіденди", "net_profit", 0.03),
    ("Чистий рух коштів від інвестиційної діяльності", None, None),  # підсумок
    ("Надходження від довгострокових кредитів", "revenue", 0.08),
    ("Погашення довгострокових кредитів", "revenue", -0.055),
    ("Надходження від короткострокових кредитів", "revenue", 0.03),
    ("Погашення короткострокових кредитів", "revenue", -0.028),
    ("Виплачені дивіденди акціонерам", "net_profit", -0.25),
    ("Витрати на викуп власних акцій", "net_profit", -0.04),
    ("Чистий рух коштів від фінансової діяльності", None, None),  # підсумок
    ("Чистий рух грошових коштів за період", None, None),         # підсумок підсумків
    ("Залишок коштів на кінець періоду", "cash_end", 1.0),
]

_SUBTOTAL_SPANS = {
    "Чистий рух коштів від операційної діяльності": (0, 11),
    "Чистий рух коштів від інвестиційної діяльності": (12, 18),
    "Чистий рух коштів від фінансової діяльності": (19, 25),
}


def build_cashflow(seed: int = 42) -> pd.DataFrame:
    """Cash flow statement; subtotal rows are real sums of their sections."""
    s = _base_series(seed)
    labels = [r[0] for r in _CASHFLOW_ROWS]
    values = {}

    for label, key, mult in _CASHFLOW_ROWS:
        if key is not None:
            values[label] = np.round(s[key] * mult, -2)

    for label, (lo, hi) in _SUBTOTAL_SPANS.items():
        members = [labels[i] for i in range(lo, hi) if labels[i] in values]
        values[label] = np.sum([values[m] for m in members], axis=0)

    values["Чистий рух грошових коштів за період"] = np.sum(
        [values[k] for k in _SUBTOTAL_SPANS], axis=0
    )

    cashflow = pd.DataFrame({"Стаття руху коштів": labels})
    for i, q in enumerate(QUARTERS):
        cashflow[q] = [float(values[lbl][i]) for lbl in labels]
    return cashflow


if __name__ == "__main__":
    inc, bal = build_tables()
    pd.set_option("display.width", 160)
    print("=== Звіт про фінансові результати ===")
    print(inc.to_string(index=False))
    print("\n=== Баланс (витяг) ===")
    print(bal.to_string(index=False))
    print("\n=== Рух грошових коштів ===")
    print(build_cashflow().to_string(index=False))
