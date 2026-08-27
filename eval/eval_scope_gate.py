"""Evaluates the scope gate.

A binary task, measured separately from the eight-class type classification.

In-scope queries come from the labelled sets. Out-of-scope ones are the
adversarial query plus probes declared here, since one example is not enough
to evaluate a gate. Probes vary in how close their vocabulary sits to the
document's own.
"""
import json
import os

import pandas as pd

from core.scope_gate import IN_SCOPE, OUT_OF_SCOPE, check
from load_data import load_tables

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = "sample_report_multipage.pdf"

# Out-of-scope probes; difficulty reflects lexical closeness to the document.
OOS_PROBES = [
    ("Яка вартість акцій Tesla на NASDAQ?", "близька"),
    ("Скільки співробітників працювало у компанії в 2024 році?", "середня"),
    ("Який рейтинг компанії за версією Moody's?", "далека"),
    ("Яка погода була в Києві у Q3 2024?", "далека"),
    ("Скільки патентів зареєстровано у 2023 році?", "далека"),
    ("Яка частка ринку компанії в сегменті роздрібної торгівлі?", "близька"),
    ("Скільки нараховано амортизації основних засобів за 2024 рік?", "дуже близька"),
    ("Який середній чек покупця у Q1 2024?", "середня"),
]


def load_in_scope():
    out, seen = [], set()
    for name in ("test_queries_full.json", "test_queries_phase1.json"):
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            for it in json.load(f):
                if it["type"] == "ADVERSARIAL" or it["query"] in seen:
                    continue
                seen.add(it["query"])
                out.append(it["query"])
    return out


def load_adversarial():
    out, seen = [], set()
    for name in ("test_queries_full.json", "test_queries_phase1.json"):
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            for it in json.load(f):
                if it["type"] == "ADVERSARIAL" and it["query"] not in seen:
                    seen.add(it["query"])
                    out.append(it["query"])
    return out


def main():
    tables = load_tables(PDF)
    rows = []

    for q in load_in_scope():
        r = check(q, tables)
        rows.append({"gold": IN_SCOPE, "pred": r["status"], "difficulty": "-",
                     "coverage": r["coverage"], "unmatched": ",".join(r["unmatched"]),
                     "query": q})
    for q in load_adversarial():
        r = check(q, tables)
        rows.append({"gold": OUT_OF_SCOPE, "pred": r["status"], "difficulty": "з набору",
                     "coverage": r["coverage"], "unmatched": ",".join(r["unmatched"]),
                     "query": q})
    for q, diff in OOS_PROBES:
        r = check(q, tables)
        rows.append({"gold": OUT_OF_SCOPE, "pred": r["status"], "difficulty": diff,
                     "coverage": r["coverage"], "unmatched": ",".join(r["unmatched"]),
                     "query": q})

    df = pd.DataFrame(rows)
    df["correct"] = df["gold"] == df["pred"]

    ins = df[df["gold"] == IN_SCOPE]
    oos = df[df["gold"] == OUT_OF_SCOPE]

    print("=" * 78)
    print("SCHEMA-AWARE ГЕЙТ ОБЛАСТІ ВИЗНАЧЕННЯ (перед М3a)")
    print("=" * 78)
    print(f"Документ: {PDF}, таблиць: {len(tables)}")
    print(f"In-scope запитів: {len(ins)}   Out-of-scope: {len(oos)}"
          f" ({len(load_adversarial())} з набору + {len(OOS_PROBES)} зондів)\n")

    tp = int((oos["pred"] == OUT_OF_SCOPE).sum())   # правильно відхилено
    fn = int((oos["pred"] == IN_SCOPE).sum())       # пропущено
    fp = int((ins["pred"] == OUT_OF_SCOPE).sum())   # хибно відхилено
    tn = int((ins["pred"] == IN_SCOPE).sum())

    print("--- Матриця (позитив = OUT_OF_SCOPE) ---")
    print(f"  правильно відхилено (TP): {tp}/{len(oos)}")
    print(f"  пропущено далі      (FN): {fn}/{len(oos)}")
    print(f"  ХИБНО відхилено     (FP): {fp}/{len(ins)}   <-- найдорожча помилка")
    print(f"  правильно пропущено (TN): {tn}/{len(ins)}")

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"\n  precision (OOS): {prec:.0%}")
    print(f"  recall    (OOS): {rec:.0%}")
    print(f"  хибних відхилень in-scope: {fp / len(ins):.0%}"
          f"   <-- критерій: 0%")

    if fn:
        print("\n--- Пропущені out-of-scope (гейт не спрацював) ---")
        for _, r in oos[oos["pred"] == IN_SCOPE].iterrows():
            print(f"  [{r['difficulty']}] {r['query']}")
            print(f"     coverage={r['coverage']}, нерозпізнано: {r['unmatched'] or '-'}")

    if fp:
        print("\n--- ХИБНО ВІДХИЛЕНІ in-scope (регресія!) ---")
        for _, r in ins[ins["pred"] == OUT_OF_SCOPE].iterrows():
            print(f"  {r['query']}  (нерозпізнано: {r['unmatched']})")

    out = os.path.join(HERE, "scope_gate_results.csv")
    df.to_csv(out, index=False)
    print(f"\nЗбережено: {out}")
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 170)
    main()
