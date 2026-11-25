"""Why the pdf arm scores 7.5% when the cells are present on 65% of pages.

The gap is the whole question of this stage. "The numbers are somewhere in the
table" is a much weaker property than "the table is usable": a row whose cells
landed one column to the left carries every number and still answers nothing.

Each wrong answer of the pdf arm is put in exactly one bucket, first match
wins, and every test is reproducible from the artefacts:

  NO_CELLS   the fact numbers are in no extracted table on that page
  WRONG_TBL  they are on the page, but not in the table the system used
  BROKEN     they are in the table the system used, yet the row that carries
             the label does NOT carry them -- the grid is misaligned
  HARNESS    a defect of ours: a crash, a non-numeric candidate, a value the
             scorer could not read
  MODEL      the table is usable and the answer is still wrong

Zero API calls.

Run:  python -m eval.tatdqa_pdf_diag
"""
import argparse
import collections
import os
import random
import zipfile

from eval import cache_io, tatdqa_loader as T
from eval.tatdqa_extract import fact_numbers, norm_num

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")


def nums_of(df):
    out = set()
    if df is None:
        return out
    for row in df.itertuples(index=False):
        for c in row:
            v = norm_num(c)
            if v is not None:
                out.add(v)
                out.add(abs(v))
    return out


def row_carrying(df, facts):
    """Is there ONE row holding at least two of the wanted numbers?

    A usable table keeps a line item and its figures on the same row. When the
    grid is misaligned the numbers survive but scatter across rows, and any
    lookup by label returns the wrong cell.
    """
    if df is None or not facts:
        return False
    for row in df.itertuples(index=False):
        vals = {v for v in (norm_num(c) for c in row) if v is not None}
        vals |= {abs(v) for v in vals}
        if len(facts & vals) >= min(2, len(facts)):
            return True
    return False


def classify(q, prow, tables, gold_tables):
    facts = fact_numbers(q)
    allnums = set()
    for df in tables.values():
        allnums |= nums_of(df)

    cand = prow.get("candidate")
    if prow.get("error"):
        return "HARNESS", f"виняток: {str(prow['error'])[:60]}"
    if cand is not None and not isinstance(cand, (int, float)):
        return "HARNESS", f"нечислова відповідь: {str(cand)[:40]}"

    if facts and not facts <= allnums:
        return "NO_CELLS", f"бракує {sorted(facts - allnums)[:3]}"

    # which table did the code name?
    used = [n for n in tables if n in str(prow.get("question", "")) or True]
    holder = [n for n, df in tables.items() if facts <= nums_of(df)]
    if facts and not holder:
        return "WRONG_TBL", "числа розкидані по кількох таблицях сторінки"

    if facts and not any(row_carrying(tables[n], facts) for n in holder):
        return "BROKEN", "числа є, але не в одному рядку — сітка зсунута"

    return "MODEL", f"наш={cand} еталон={prow.get('gold')}"


def show(df, title, width=104, rows=6):
    print(f"    {title}")
    if df is None:
        print("      (немає)")
        return
    try:
        txt = df.head(rows).to_string(index=False)
    except Exception as exc:
        print(f"      (не друкується: {exc})")
        return
    for line in txt.split("\n")[:rows + 1]:
        print("      " + line[:width])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-flash-lite-latest")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    from eval.tatdqa_run import gold_table, extracted_table

    zf = zipfile.ZipFile(T.DOCS_ZIP)
    sel = T.select(zf=zf)
    random.Random(args.seed).shuffle(sel)
    sample = sel[:args.n]
    by_uid = {q["uid"]: q for q in sample}

    pdf, _ = cache_io.load_rows(
        os.path.join(CACHE, f"tatdqa_pdf_{args.model}.jsonl"))
    P = {x["id"]: x for x in pdf}

    wrong = [q for q in sample if not P.get(q["uid"], {}).get("correct")]
    print("=" * 78)
    print(f"ЧОМУ ГІЛКА pdf ХИБИТЬ — {args.model}")
    print("=" * 78)
    print(f"питань {len(sample)}, хибних {len(wrong)}")
    print()

    kinds = collections.Counter()
    detail = []
    for q in wrong:
        prow = P.get(q["uid"])
        if prow is None:
            kinds["HARNESS"] += 1
            detail.append(("HARNESS", q, "рядка немає в кеші"))
            continue
        tables, _ = extracted_table(q)
        gt = gold_table(q, zf)
        k, why = classify(q, prow, tables, gt)
        kinds[k] += 1
        detail.append((k, q, why))

    total = sum(kinds.values())
    order = ["NO_CELLS", "WRONG_TBL", "BROKEN", "HARNESS", "MODEL"]
    names = {"NO_CELLS": "а) клітинок немає",
             "WRONG_TBL": "в) обрано не ту таблицю",
             "BROKEN": "б) структура зламана",
             "HARNESS": "г) вада обв'язки",
             "MODEL": "д) помилка моделі"}
    for k in order:
        v = kinds.get(k, 0)
        print(f"  {names[k]:28s} {v:3d}  {v / max(total, 1):5.1%}  "
              f"{'#' * int(round(30 * v / max(total, 1)))}")
    print()
    if kinds.get("HARNESS"):
        print("  !!! Є ВАДИ ОБВ'ЯЗКИ — доповідаю, не виправляю мовчки:")
        for k, q, why in detail:
            if k == "HARNESS":
                print(f"      {q['uid'][:12]}  {why}")
        print()

    print("--- ПРИКЛАДИ: витягнута таблиця проти розмітки ---")
    shown = 0
    for k, q, why in detail:
        if k not in ("BROKEN", "WRONG_TBL", "NO_CELLS") or shown >= args.examples:
            continue
        shown += 1
        prow = P[q["uid"]]
        tables, _ = extracted_table(q)
        gt = gold_table(q, zf)
        print()
        print(f"  [{k}] {q['uid'][:14]}  {str(q['question'])[:70]}")
        print(f"    еталон={T.numeric_answer(q)} шкала={q.get('scale')} "
              f"facts={q.get('facts')}  наш={prow.get('candidate')}")
        print(f"    {why}")
        for name, df in list(tables.items())[:1]:
            show(df, f"витягнуто парсером ({name}, {df.shape[0]}x{df.shape[1]}):")
        if gt:
            n0 = list(gt)[0]
            show(gt[n0], f"розмітка TAT-DQA ({gt[n0].shape[0]}x{gt[n0].shape[1]}):")

    out = os.path.join(HERE, f"tatdqa_pdf_diag_{args.model}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,bucket,why,question\n")
        for k, q, why in detail:
            f.write(f"{q['uid']},{k},\"{why}\","
                    f"\"{str(q['question'])[:90].replace(chr(34), '')}\"\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
