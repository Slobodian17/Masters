"""End-to-end run on TAT-DQA: from a real PDF page to a number.

This is the part FinQA cannot test. FinQA ships tables already extracted, so
nothing measured on it says what happens when the table has to be found in a
page first. Here the pipeline starts at the PDF.

Two arms on the SAME questions, so the difference is attributable:

  pdf    our parser reads the page, the method answers from what it extracted
  gold   the method answers from the table rows TAT-DQA itself annotated

The gap between them is the cost of extraction. It is not a model result and
must not be reported as one.

Measured before spending a call (eval.tatdqa_extract): with the text strategy
the parser returns a table on 100% of pages but the needed cells are present
on 65%. That 65% is the ceiling of the `pdf` arm, and a question outside it
cannot be answered by any model.

Run:  python -m eval.tatdqa_run --n 40
"""
import argparse
import json
import os
import random
import time
import zipfile

import pandas as pd

from core import decision
from core.pipeline import run as pipeline_run
from eval import cache_io, tatdqa_loader as T
from eval.tatdqa_extract import ensure_pdf, fact_numbers, table_numbers
from generation import llm_client
from ingestion.ingest_pdf import _unique_columns

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")
THRESHOLD = 0.7
MAX_ITER = 3

INSTRUCTIONS = (
    "Conventions of this dataset:\n"
    "- Answer with a plain number. If the table is stated in thousands or "
    "millions, answer in the same unit the table uses.\n"
    "- Express a percentage as a number out of 100: write 14.1, not 0.141.\n"
    "- `result` must be a single number, never a DataFrame or a Series."
)


_CELL = __import__("re").compile(
    r"\(?-?[\d,]+\.?\d*\)?%?$|^[-–—]$|^\$$")


def _split_row(text):
    """A layout row into (label, cell, cell, ...).

    Splitting on whitespace was wrong and it is what made this arm look like a
    failure of the method: "Cost of revenue 43 % 50 % 55 %" became nine
    columns with the label shredded across three of them, so every lookup by
    row label found nothing and the code returned 0. The label is the leading
    run of non-numeric tokens; the cells are what follows.
    """
    toks = str(text or "").split()
    i = 0
    while i < len(toks) and not _CELL.match(toks[i]):
        i += 1
    label = " ".join(toks[:i]).strip()
    cells = [t for t in toks[i:] if t not in ("%", "$")]
    if not label and not cells:
        return []
    return [label] + cells


def gold_table(q, zf):
    """The table as TAT-DQA annotated it: layout rows, split into cells.

    A row arrives as one string, "Cost of revenue 43 % 50 % 55 %", so the split
    is on runs of whitespace. This is weaker than a gold grid and is labelled
    as such wherever its numbers are used.
    """
    blocks = sorted(T.page_blocks(q["doc_uid"], zf),
                    key=lambda b: b.get("order", 0))
    # A page carries prose and often several tables. Consecutive table-like
    # rows form one table; the right one is the group that holds the cells the
    # gold answer was derived from, which TAT-DQA states in `facts`. Picking
    # by facts is legitimate here and only here: this arm exists to say what
    # the method does when the table is given correctly, so handing it the
    # wrong table would measure table search instead.
    # A single prose line inside a table -- a sub-heading such as "Operating
    # expenses:" -- must not split it in two. Measured: breaking on the first
    # non-table row left the needed cells outside the chosen group on 7 of 20
    # wrong answers (35%).
    groups, cur, gap = [], [], 0
    for b in blocks:
        if T.is_table_row(b.get("text")):
            r = _split_row(b["text"])
            if r:
                cur.append(r)
                gap = 0
        elif cur:
            gap += 1
            if gap > 1:
                groups.append(cur)
                cur, gap = [], 0
    if cur:
        groups.append(cur)
    if not groups:
        return None

    from eval.tatdqa_extract import fact_numbers, norm_num
    facts = fact_numbers(q)

    def covers(g):
        nums = set()
        for r in g:
            for c in r[1:]:
                v = norm_num(c)
                if v is not None:
                    nums.add(v)
                    nums.add(abs(v))
        return facts <= nums if facts else False

    # THE HINT, and it is why this arm is an upper bound rather than a
    # measurement of the method: when a group holds the cells the gold answer
    # used, that group is handed over directly. Where no group does, no hint
    # exists to give, so every group goes through and the system searches --
    # the same thing the pdf arm always does.
    hit = [g for g in groups if covers(g)]
    chosen = hit if hit else groups
    out = {}
    for i, rows in enumerate(chosen):
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        if len(rows) <= 1:
            continue
        out[f"table_{i}"] = pd.DataFrame(rows[1:],
                                         columns=_unique_columns(rows[0]))
    return out or None
    # Same trap as defect 19, and it caught me twice: a header row split on
    # whitespace repeats tokens ("2019 2018 2019 2018") and empties, so the
    # columns must be made unique here too or `df[col]` stops being a Series.
    return pd.DataFrame(rows[1:], columns=_unique_columns(rows[0]))


def extracted_table(q):
    from ingestion.ingest_pdf import extract_tables
    zf = zipfile.ZipFile(T.DOCS_ZIP)
    path = ensure_pdf(q["doc_uid"], zf)
    try:
        tabs = extract_tables(path)
    except Exception:
        return {}, False
    facts = fact_numbers(q)
    covers = any(facts <= table_numbers(df) for df in tabs) if facts else False
    # Name them as the prompt will: the search for the right one is part of
    # the task when a page carries several (28% of pages do).
    return {f"table_{i}": df for i, df in enumerate(tabs)}, covers


def one(q, tables, tag):
    llm_client.reset_stats()
    t0 = time.perf_counter()
    err = ecls = None
    try:
        res = pipeline_run(q["question"], tables, threshold=THRESHOLD,
                           max_iter=MAX_ITER, verbose=False,
                           source_file=f"tatdqa:{q['doc_uid']}",
                           instructions=INSTRUCTIONS)
    except Exception as exc:
        err, ecls = f"{type(exc).__name__}: {exc}", type(exc).__name__
        res = {"status": "ERROR", "trace": []}
    o = decision.outcome(res, THRESHOLD)
    st = llm_client.get_stats()
    cand = o["candidate"] if o["candidate"] is not None else o["answer"]

    def plain(v):
        """numpy scalars are not JSON-serialisable and killed a run mid-write."""
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)

    cand, ans = plain(cand), plain(o["answer"])
    return {"id": q["uid"], "doc_uid": q["doc_uid"], "arm": tag,
            "question": q["question"], "gold": T.numeric_answer(q),
            "scale": q.get("scale"), "status": res.get("status"),
            "abstain_reason": o["abstain_reason"],
            "answer": ans, "candidate": cand,
            "correct": T.is_correct(cand, q),
            "n_tables": len(tables),
            "confidence": plain(o["confidence"]),
            "signals": {k: (bool(v) if isinstance(v, bool) else v)
                        for k, v in (o["signals"] or {}).items()},
            "iterations": o["iterations"],
            "calls_total": st.get("calls", 0),
            "tokens": st.get("total_tokens", 0),
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "error": err, "error_class": ecls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pace", type=float, default=60.0)
    ap.add_argument("--arm", default="pdf", choices=["pdf", "gold"])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if llm_client.active_provider() == "mock":
        print("MOCK не містить цих питань — потрібен ключ.")
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ.setdefault(llm_client.LEDGER_ENV,
                          os.path.join(CACHE_DIR, "token_ledger.jsonl"))

    zf = zipfile.ZipFile(T.DOCS_ZIP)
    sel = T.select(zf=zf)
    rng = random.Random(args.seed)
    rng.shuffle(sel)
    sample = sel[:args.n]

    tag = args.tag or f"tatdqa_{args.arm}_{llm_client.active_model().split('/')[-1]}"
    path = os.path.join(CACHE_DIR, f"{tag}.jsonl")
    done = {r["id"] for r in cache_io.load_rows(path)[0]}

    print("=" * 78)
    print(f"TAT-DQA, наскрізний прогін, гілка «{args.arm}»")
    print("=" * 78)
    print(f"модель: {llm_client.active_model()}   питань: {len(sample)}   "
          f"вже готово: {len(done)}")
    print(f"темп {args.pace} с   -> {tag}.jsonl")
    print()

    n_fake = 0
    with open(path, "a", encoding="utf-8") as f:
        for i, q in enumerate(sample, start=1):
            if q["uid"] in done:
                continue
            if args.arm == "pdf":
                tables, covers = extracted_table(q)
            else:
                tabs = gold_table(q, zf)
                # `covers` for this arm says whether the hint existed at all:
                # True means a group held the gold cells and was handed over.
                tables = tabs or {}
                covers = bool(tabs) and len(tabs) == 1
            if not tables:
                print(f"[{i}/{len(sample)}] таблиці немає — пропускаю без запису")
                continue
            row = one(q, tables, args.arm)
            row["covers_facts"] = covers
            try:
                cache_io.write_row(f, row)
            except cache_io.FakeRow as exc:
                n_fake += 1
                print(f"[{i}/{len(sample)}] ВИКЛИК НЕ ВІДБУВСЯ: {exc}")
                if "tpd" in str(exc).lower() or "tokens per day" in str(exc).lower():
                    print("  добовий ліміт — зупиняюсь")
                    break
                continue
            print(f"[{i}/{len(sample)}] {row['status']:7s} "
                  f"таблиць={row['n_tables']} містить={covers} "
                  f"tok={row['tokens']} {row['elapsed_s']}s")
            if i < len(sample):
                time.sleep(args.pace)

    a = cache_io.audit(path)
    print()
    print(f"кеш: рядків {a['rows']}, справжніх {a['real']}, "
          f"фальшивих {a['transport']}, викликів не відбулось {n_fake}")
    print("Збережено:", path)


if __name__ == "__main__":
    main()
