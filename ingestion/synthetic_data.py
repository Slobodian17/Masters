"""Synthetic financial dataset: income statement, balance sheet, cash flow.

Values are computed rather than written by hand, so the reference answers in
the test sets are arithmetically consistent by construction. Tables use a plain
RangeIndex to match what the PDF parser returns.
"""
import numpy as np
import pandas as pd

from core import lang

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

    prof = lang.get()
    income_stmt = pd.DataFrame({prof.INCOME_HEADER: list(prof.INCOME_LABELS)})
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

    balance = pd.DataFrame({prof.BALANCE_HEADER: list(prof.BALANCE_LABELS)})
    bal_values = [cash_end, receivables, total_assets, total_liab, equity]
    for i, q in enumerate(QUARTERS):
        balance[q] = [v[i] for v in bal_values]

    # RangeIndex on purpose, matching what the PDF parser produces.
    return income_stmt, balance


# Cash flow statement. It deliberately shares vocabulary with the income
# statement, which makes retrieval non-trivial, and it is long enough to split
# across pages, which makes it the stitching fixture.

# Cash flow rows as (base series key, multiplier); None marks a subtotal that
# is computed from its section rather than generated. Labels are positional and
# come from the language profile, so both profiles produce the same numbers
# under different names — which is what makes the two runs comparable.
_CASHFLOW_ROWS = (
    ("revenue", 0.95),
    ("revenue", 0.012),
    ("revenue", 0.008),
    ("cogs", -0.72),
    ("cogs", -0.18),
    ("opex", -0.55),
    ("opex", -0.12),
    ("opex", -0.09),
    ("opex", -0.14),
    ("tax", -1.0),
    ("fin_expenses", -1.0),
    (None, None),                 # net cash from operating activities
    ("revenue", -0.11),
    ("revenue", -0.015),
    ("revenue", 0.006),
    ("revenue", -0.02),
    ("revenue", 0.011),
    ("net_profit", 0.03),
    (None, None),                 # net cash from investing activities
    ("revenue", 0.08),
    ("revenue", -0.055),
    ("revenue", 0.03),
    ("revenue", -0.028),
    ("net_profit", -0.25),
    ("net_profit", -0.04),
    (None, None),                 # net cash from financing activities
    (None, None),                 # net change in cash for the period
    ("cash_end", 1.0),
)

# {subtotal row index: (first member index, one past the last)}. Index-based on
# purpose: a keyword rule would be one more thing to translate.
_SUBTOTAL_SPANS = {
    11: (0, 11),
    18: (12, 18),
    25: (19, 25),
}

_GRAND_TOTAL_INDEX = 26


def build_cashflow(seed: int = 42) -> pd.DataFrame:
    """Cash flow statement; subtotal rows are real sums of their sections."""
    prof = lang.get()
    s = _base_series(seed)
    labels = list(prof.CASHFLOW_LABELS)
    assert len(labels) == len(_CASHFLOW_ROWS)

    values = {}
    for i, (key, mult) in enumerate(_CASHFLOW_ROWS):
        if key is not None:
            values[i] = np.round(s[key] * mult, -2)

    for target, (lo, hi) in _SUBTOTAL_SPANS.items():
        members = [i for i in range(lo, hi) if i in values]
        values[target] = np.sum([values[m] for m in members], axis=0)

    values[_GRAND_TOTAL_INDEX] = np.sum(
        [values[t] for t in _SUBTOTAL_SPANS], axis=0
    )

    cashflow = pd.DataFrame({prof.CASHFLOW_HEADER: labels})
    for i, q in enumerate(QUARTERS):
        cashflow[q] = [float(values[j][i]) for j in range(len(labels))]
    return cashflow


if __name__ == "__main__":
    inc, bal = build_tables()
    pd.set_option("display.width", 160)
    print("=== income ===")
    print(inc.to_string(index=False))
    print("\n=== balance ===")
    print(bal.to_string(index=False))
    print("\n=== cashflow ===")
    print(build_cashflow().to_string(index=False))
