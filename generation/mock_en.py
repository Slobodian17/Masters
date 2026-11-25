"""MOCK lookup tables for the English profile.

A mirror of the Ukrainian maps in `llm_client`, query for query, so both
profiles exercise the same paths: the same three deliberate failures, the same
scope-dependent variants, the same repair levels. Where the two runs differ,
the cause is the language and not the fixture.

Not a general LLM simulation — a lookup keyed on the exact query text of the
English test sets.
"""

INCOME = "income['Line item']"
BALANCE = "balance['Line item']"
CASHFLOW = "cashflow['Cash flow item']"

Y2023 = "['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023']"
Y2024 = "['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']"
ALL8 = "['Q1_2023','Q2_2023','Q3_2023','Q4_2023','Q1_2024','Q2_2024','Q3_2024','Q4_2024']"

CODE_MAP = {
    "What was revenue in Q3 2024?":
        f"result = income.loc[{INCOME} == 'Revenue', 'Q3_2024'].values[0]",

    "What was net profit in Q1 2023?":
        f"result = income.loc[{INCOME} == 'Net profit', 'Q1_2023'].values[0]",

    "What was total revenue for 2023?":
        f"cols = {Y2023}\nresult = income.loc[{INCOME} == 'Revenue', cols].values.sum()",

    "What was total cost of sales for 2024?":
        f"cols = {Y2024}\nresult = income.loc[{INCOME} == 'Cost of sales', cols].values.sum()",

    "What was the average net profit for 2023?":
        f"cols = {Y2023}\nresult = income.loc[{INCOME} == 'Net profit', cols].values.mean()",

    "What was the average revenue across all 8 quarters?":
        f"cols = {ALL8}\nresult = income.loc[{INCOME} == 'Revenue', cols].values.mean()",

    "What was the maximum net profit across all quarters?":
        f"cols = {ALL8}\nresult = income.loc[{INCOME} == 'Net profit', cols].values.max()",

    "What was the minimum revenue across all quarters?":
        f"cols = {ALL8}\nresult = income.loc[{INCOME} == 'Revenue', cols].values.min()",

    "How much larger was revenue in Q4 2024 than in Q4 2023 in absolute terms?":
        f"a = income.loc[{INCOME} == 'Revenue', 'Q4_2024'].values[0]\n"
        f"b = income.loc[{INCOME} == 'Revenue', 'Q4_2023'].values[0]\nresult = a - b",

    "What is the difference between net profit in Q1 2024 and Q1 2023?":
        f"a = income.loc[{INCOME} == 'Net profit', 'Q1_2024'].values[0]\n"
        f"b = income.loc[{INCOME} == 'Net profit', 'Q1_2023'].values[0]\nresult = a - b",

    "By what percent did revenue grow from Q1 2023 to Q1 2024?":
        f"a = income.loc[{INCOME} == 'Revenue', 'Q1_2024'].values[0]\n"
        f"b = income.loc[{INCOME} == 'Revenue', 'Q1_2023'].values[0]\nresult = (a - b) / b * 100",

    "What was the growth of net profit from Q3 2023 to Q3 2024 in percent?":
        f"a = income.loc[{INCOME} == 'Net profit', 'Q3_2024'].values[0]\n"
        f"b = income.loc[{INCOME} == 'Net profit', 'Q3_2023'].values[0]\nresult = (a - b) / b * 100",

    "How many quarters had net profit above 150000?":
        f"cols = {ALL8}\nrow = income.loc[{INCOME} == 'Net profit', cols].values[0]\n"
        "result = (row > 150000).sum()",

    "What was the total revenue for quarters where net profit exceeded 200000?":
        f"cols = {ALL8}\nprofit = income.loc[{INCOME} == 'Net profit', cols].values[0]\n"
        f"revenue = income.loc[{INCOME} == 'Revenue', cols].values[0]\n"
        "result = revenue[profit > 200000].sum()",

    # Present in all three tables under different labels; the variant is
    # picked from whichever table retrieval put in scope.
    "What was cash at end of quarter in Q2 2024?": {
        "income": f"result = income.loc[{INCOME} == 'Cash at end of quarter', 'Q2_2024'].values[0]",
        "balance": f"result = balance.loc[{BALANCE} == 'Cash and cash equivalents', 'Q2_2024'].values[0]",
        "cashflow": f"result = cashflow.loc[{CASHFLOW} == 'Cash at end of period', 'Q2_2024'].values[0]",
    },

    "What was the net profit margin (%) in Q4 2024?":
        f"a = income.loc[{INCOME} == 'Net profit', 'Q4_2024'].values[0]\n"
        f"b = income.loc[{INCOME} == 'Revenue', 'Q4_2024'].values[0]\nresult = a / b * 100",

    "What was the total of finance costs for 2023?":
        f"cols = {Y2023}\nresult = income.loc[{INCOME} == 'Finance costs', cols].sum(axis=1).values[0]",

    # Adversarial: the document holds nothing of the sort. The generated code
    # is plausible-looking and addresses a row that does not exist.
    "What was the profitability of the B2B segment under IFRS 15 in Q2 2023?": {
        "income": f"result = income.loc[{INCOME} == 'B2B segment IFRS 15', 'Q2_2023'].values[0]",
        "balance": f"result = balance.loc[{BALANCE} == 'B2B segment IFRS 15', 'Q2_2023'].values[0]",
        "cashflow": f"result = cashflow.loc[{CASHFLOW} == 'B2B segment IFRS 15', 'Q2_2023'].values[0]",
    },

    "What was the maximum gross profit across all quarters?":
        f"cols = {ALL8}\nresult = income.loc[{INCOME} == 'Gross profit', cols].values.max()",

    "What was total equity at Q2 2024?":
        f"result = balance.loc[{BALANCE} == 'Total equity', 'Q2_2024'].values[0]",

    "What were total assets at Q4 2023?":
        f"result = balance.loc[{BALANCE} == 'Total assets', 'Q4_2023'].values[0]",

    "By how much did total assets grow from Q1 2023 to Q4 2024?":
        f"a = balance.loc[{BALANCE} == 'Total assets', 'Q4_2024'].values[0]\n"
        f"b = balance.loc[{BALANCE} == 'Total assets', 'Q1_2023'].values[0]\nresult = a - b",

    "What was the share of total equity in total assets (%) at Q3 2024?":
        f"a = balance.loc[{BALANCE} == 'Total equity', 'Q3_2024'].values[0]\n"
        f"b = balance.loc[{BALANCE} == 'Total assets', 'Q3_2024'].values[0]\nresult = a / b * 100",

    "What were the maximum trade receivables across all quarters?":
        f"cols = {ALL8}\nresult = balance.loc[{BALANCE} == 'Trade receivables', cols].values.max()",

    "What were the average total liabilities for 2024?":
        f"cols = {Y2024}\nresult = balance.loc[{BALANCE} == 'Total liabilities', cols].values.mean()",

    "How much was paid in dividends to shareholders in Q2 2023?":
        f"result = cashflow.loc[{CASHFLOW} == 'Dividends paid to shareholders', 'Q2_2023'].values[0]",

    "What was the net cash from operating activities in Q4 2024?":
        f"result = cashflow.loc[{CASHFLOW} == 'Net cash from operating activities', 'Q4_2024'].values[0]",

    "What was the total of marketing expenses for 2024?":
        f"cols = {Y2024}\nresult = cashflow.loc[{CASHFLOW} == 'Marketing expenses', cols].values.sum()",

    "How much interest was paid on loans in total for 2023?":
        f"cols = {Y2023}\nresult = cashflow.loc[{CASHFLOW} == 'Interest paid on loans', cols].values.sum()",

    "What were the largest proceeds from sales of goods across all quarters?":
        f"cols = {ALL8}\nresult = cashflow.loc[{CASHFLOW} == 'Proceeds from sales of goods', cols].values.max()",

    "How many quarters had net change in cash for the period above 80000?":
        f"cols = {ALL8}\n"
        f"row = cashflow.loc[{CASHFLOW} == 'Net change in cash for the period', cols].values[0]\n"
        "result = (row > 80000).sum()",

    "What was the average net cash from investing activities for 2023?":
        f"cols = {Y2023}\n"
        f"result = cashflow.loc[{CASHFLOW} == 'Net cash from investing activities', cols].values.mean()",

    "By what percent did proceeds from sales of goods grow from Q1 2023 to Q4 2024?":
        f"a = cashflow.loc[{CASHFLOW} == 'Proceeds from sales of goods', 'Q4_2024'].values[0]\n"
        f"b = cashflow.loc[{CASHFLOW} == 'Proceeds from sales of goods', 'Q1_2023'].values[0]\n"
        "result = (a - b) / b * 100",
}

# Deliberate first-attempt failures, one per repair level being demonstrated.
FIRST_ATTEMPT_OVERRIDE = {
    # Numerator and denominator swapped: runs fine, returns 608%, and only the
    # range signal catches it. Exercises the L2 path.
    "What was the net profit margin (%) in Q4 2024?":
        f"a = income.loc[{INCOME} == 'Revenue', 'Q4_2024'].values[0]\n"
        f"b = income.loc[{INCOME} == 'Net profit', 'Q4_2024'].values[0]\nresult = a / b * 100",

    # Label shortened: the filter matches nothing, .values[0] raises
    # IndexError, and the deterministic L0 repair fixes it with no LLM call.
    "What was the total of finance costs for 2023?":
        f"cols = {Y2023}\nresult = income.loc[{INCOME} == 'Finance cost', cols].sum(axis=1).values[0]",

    # Misspelt label, same L0 path.
    "What was the total of marketing expenses for 2024?":
        f"cols = {Y2024}\n"
        f"result = cashflow.loc[{CASHFLOW} == 'Marketting expenses', cols].sum(axis=1).values[0]",
}

# What the model returns once it has been told what went wrong.
CORRECT_RETRY = {
    "What was the net profit margin (%) in Q4 2024?":
        f"a = income.loc[{INCOME} == 'Net profit', 'Q4_2024'].values[0]\n"
        f"b = income.loc[{INCOME} == 'Revenue', 'Q4_2024'].values[0]\nresult = a / b * 100",
}
