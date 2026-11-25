"""Is there anything for a cross-table check to find?

The applicability estimate for corpus + B2, before any budget is spent on it.
B1 taught the lesson: a mechanism that is structurally silent stays silent
however many questions it is shown, and measuring that costs money that buys
nothing (5.16).

B2 in a corpus refutes when numbers drawn from DIFFERENT tables fail an
arithmetic identity. Three things have to hold for that to be possible, and
each is checked here, in order of how cheaply it can kill the idea:

  1. the accepted wrong answers must come from filings that HAVE more than
     one table -- otherwise the corpus mode does not even apply to them;
  2. values must RECUR across tables of one filing -- if every table holds
     its own numbers and no others, nothing ties them;
  3. arithmetic identities must SPAN tables -- a value in one table equal to
     a sum of values in another.

Everything is computed from cached runs and the dataset. Zero calls.

Run:  python -m eval.finqa_crosstable
"""
import argparse
import collections
import hashlib
import itertools
import os
import re

import numpy as np
import pandas as pd

from eval.finqa_replay import load_cache
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))

REL_TOL = 1e-4
ABS_TOL = 0.5
MAX_MEMBERS = 3


def filing_of(rec: dict):
    parts = (rec.get("filename") or rec.get("id", "")).split("/")
    return (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], "")


def content_hash(table) -> str:
    body = "|".join(";".join(re.sub(r"\s+", " ", str(c)).strip().lower()
                             for c in row) for row in table)
    return hashlib.md5(body.encode()).hexdigest()


def build_pool(records: list) -> dict:
    """{filing: {hash: table}} -- deduplicated BY CONTENT, not by instance.

    Instance-level collection would put copies of one page into the pool, and
    a cross-table check would then confirm a value against a copy of the very
    table it came from: the signal would look alive while measuring nothing.
    66.6% of table instances in FinQA are duplicates (5.14.1).
    """
    pool = collections.defaultdict(dict)
    for r in records:
        t = r.get("table")
        if t:
            pool[filing_of(r)][content_hash(t)] = t
    return pool


def numeric_values(table) -> set:
    out = set()
    df, _ = fq.to_frame(table)
    if df.empty:
        return out
    for c in df.columns[1:]:
        for v in pd.to_numeric(df[c], errors="coerce").dropna().tolist():
            if abs(float(v)) > 1e-9:
                out.add(round(float(v), 6))
    return out


def _close(a, b):
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(b), 1.0))


def cross_identities(values_a: set, values_b: set, limit: int = 400) -> int:
    """How many values of A equal a sum of 2..3 values of B."""
    b = sorted(values_b)[:40]
    found = 0
    for k in range(2, MAX_MEMBERS + 1):
        for combo in itertools.combinations(b, k):
            s = sum(combo)
            for v in values_a:
                if _close(v, s):
                    found += 1
                    if found >= limit:
                        return found
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="ab_no_v2,ab_yes_v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-filings", type=int, default=60)
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    recs_test = fq.load_split(args.split)
    by_id = {r["id"]: r for r in recs_test}
    allrecs = []
    for sp in ("train", "dev", "test"):
        try:
            allrecs += fq.load_split(sp)
        except Exception:
            pass

    pool = build_pool(allrecs)

    print("=" * 78)
    print("ОЦІНКА ЗАСТОСОВНОСТІ corpus + B2 — нуль викликів")
    print("=" * 78)
    print("Три умови, кожна може закрити напрям дешевше за попередню.")
    print()

    # --- 1. do the wrong answers live in multi-table filings? -------------
    wrong = []
    for tag in tags:
        for r in load_cache(tag):
            if r["status"] == "ACCEPT" and r["answer_correct"] is False:
                wrong.append(r)

    multi = 0
    sizes = []
    for r in wrong:
        rec = by_id.get(r["id"])
        if not rec:
            continue
        n = len(pool.get(filing_of(rec), {}))
        sizes.append(n)
        multi += n >= 2

    print("--- 1. Чи взагалі є інші таблиці у філінгу цих помилок ---")
    print(f"  прийнятих хибних: {len(wrong)}")
    print(f"  з них у філінгу з >=2 різними таблицями: {multi}/{len(sizes)} "
          f"= {multi/len(sizes):.0%}" if sizes else "  немає даних")
    if sizes:
        print(f"  таблиць у філінгу: медіана {int(np.median(sizes))}, "
              f"max {max(sizes)}")
    print("  <- ВЕРХНЯ межа застосовності corpus-режиму до цих помилок")

    # --- 2. do values recur across tables of one filing? ------------------
    print()
    print("--- 2. Чи повторюються числа між таблицями одного філінга ---")
    multi_filings = [(f, t) for f, t in pool.items() if len(t) >= 2]
    sample = multi_filings[:args.max_filings]
    shared_any, pair_count, shared_pairs = 0, 0, 0
    overlaps = []
    for f, tabs in sample:
        vals = [numeric_values(t) for t in tabs.values()]
        vals = [v for v in vals if v]
        if len(vals) < 2:
            continue
        any_share = False
        for a, b in itertools.combinations(range(len(vals)), 2):
            pair_count += 1
            inter = vals[a] & vals[b]
            if inter:
                shared_pairs += 1
                any_share = True
                overlaps.append(len(inter) / min(len(vals[a]), len(vals[b])))
        shared_any += any_share
    print(f"  філінгів перевірено: {len(sample)} (з {len(multi_filings)} багатотабличних)")
    print(f"  пар таблиць: {pair_count}")
    print(f"  пар зі спільним числом: {shared_pairs}/{pair_count} = "
          f"{shared_pairs/pair_count:.0%}" if pair_count else "  —")
    print(f"  філінгів, де хоч одна пара ділить число: {shared_any}/{len(sample)} "
          f"= {shared_any/len(sample):.0%}" if sample else "  —")
    if overlaps:
        print(f"  частка спільних значень у парі: медіана {np.median(overlaps):.0%}")

    # --- 3. do identities span tables? ------------------------------------
    print()
    print("--- 3. Чи існують арифметичні тотожності МІЖ таблицями ---")
    with_ident, checked = 0, 0
    for f, tabs in sample[:30]:
        vals = [numeric_values(t) for t in tabs.values()]
        vals = [v for v in vals if len(v) >= 2]
        if len(vals) < 2:
            continue
        checked += 1
        hit = False
        for a, b in itertools.permutations(range(len(vals)), 2):
            if cross_identities(vals[a], vals[b], limit=1):
                hit = True
                break
        with_ident += hit
    print(f"  філінгів перевірено: {checked}")
    print(f"  де знайдено тотожність між таблицями: {with_ident}/{checked} = "
          f"{with_ident/checked:.0%}" if checked else "  —")
    print("  (значення в таблиці A дорівнює сумі 2-3 значень таблиці B)")
    print()
    print("  ЗАСТЕРЕЖЕННЯ: такі збіги на десятках чисел здебільшого")
    print("  ВИПАДКОВІ. Це верхня межа, а не кількість справжніх тотожностей;")
    print("  фільтр «тримається в усіх стовпцях» тут не застосовний, бо")
    print("  стовпці різних таблиць не вирівняні.")

    print()
    print("=" * 78)
    print("ПІДСУМОК ДЛЯ РІШЕННЯ")
    print("=" * 78)
    if sizes and multi / len(sizes) < 0.2:
        print("  Умова 1 не виконується: більшість помилок у однотабличних")
        print("  філінгах. Corpus-режим до них не застосовний -> прогін не робимо.")
    elif pair_count and shared_pairs / pair_count < 0.2:
        print("  Умова 2 не виконується: таблиці одного філінга майже не")
        print("  ділять чисел -> зв'язувати нема чого -> прогін не робимо.")
    else:
        print("  Умови 1-2 виконуються. Рішення за умовою 3 і за бюджетом.")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
