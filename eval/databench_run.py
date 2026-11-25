"""The three arms on DataBench, one model, one post-processing. Resumable.

  A       retrieval plus direct answer: the model sees the schema and the
          top-k rows a lexical retriever picked, and is told how many rows the
          table really has, so it can see that it is not seeing all of them.
  A-lite  the same prompt shape on DataBench Lite, where the 20-row sample IS
          the whole table. Scored against `sample_answer`, which is the answer
          recomputed on that sample -- scoring it against `answer` would mark
          it on a question it was never asked.
  B       generated pandas plus crash repair: the model sees column names,
          dtypes and a few rows, never the retrieved window, and the answer is
          whatever the code computes.

What makes the arms comparable is held fixed in one place: the same model, the
same `normalise` on whatever comes back, and the organisers' own comparison via
`databench_loader.compare`. The gold answer never enters a prompt.

Run:  python -m eval.databench_run --arm A --limit 100
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

from core import sandbox
from eval import cache_io
from generation import llm_client
from ingestion import databench_loader as db

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, "databench_dev.json")
CACHE = os.path.join(HERE, "databench_cache")

TOPK = 30
HEAD_ROWS = 5
# Only arm A's RETRIEVED WINDOW is capped, because only it is unbounded:
# without a cap one question on 037_Ted (TED transcripts, 30 rows x 19
# columns) renders a 459k-token prompt, which exceeds the per-minute token
# limit on its own and can never succeed -- and an unbounded prompt is not a
# system anyway. Arm B's 5 format rows and the 20-row Lite table are bounded
# by construction and are left alone: capping them would alter prompts for no
# reason and make those runs incomparable with what is already measured.
# Cells are clipped first, then rows dropped, and what survived is reported.
CELL_CHARS = 200
WINDOW_CHARS = 48000
MAX_TOKENS = 1200
SLEEP = 4.2
MAX_REPAIR = 2

SYS_DIRECT = ("You answer questions about tabular data. You reply with one "
              "line and nothing else.")
SYS_CODE = ("You write pandas code. You reply with code only, no prose and no "
            "markdown fences.")

P_DIRECT = """Table: {name}
Rows in the full table: {nrows}
Columns and types:
{schema}

{window_note}
{rows}

Question: {q}
Expected answer type: {atype}

Reply with exactly one line:
ANSWER: <value>
For a list, write it as [a, b, c]. For a yes/no question, write true or false.
Give the value only, with no units, no commentary and no thousands separators."""

P_CODE = """You are given a pandas DataFrame `df`.

Table: {name}
Rows in the full table: {nrows}
Columns and types:
{schema}

First {nhead} rows, for format only:
{rows}

Question: {q}
Expected answer type: {atype}

Write pandas code that assigns the answer to a variable named `result`.
`df` is the FULL table, not only the rows shown. For a list answer, make
`result` a Python list. For a yes/no question, make it a bool.
Reply with code only."""

P_REPAIR = """Your code raised an error.

{schema_block}

Question: {q}
Expected answer type: {atype}

Your code:
{code}

Error: {err}

Write corrected pandas code assigning the answer to `result`. Code only."""


# ---- one post-processing for every arm ----

def normalise(v):
    """Whatever an arm produced -> the string the official comparison reads."""
    if isinstance(v, (pd.Series, pd.Index)):
        v = v.tolist()
    elif isinstance(v, np.ndarray):
        v = v.tolist()
    elif isinstance(v, pd.DataFrame):
        v = v.values.flatten().tolist()
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, (list, tuple, set)):
        out = []
        for x in v:
            if isinstance(x, (np.generic,)):
                x = x.item()
            if isinstance(x, pd.Timestamp):
                x = x.date().isoformat()
            out.append(x)
        return "[" + ", ".join(str(x) for x in out) + "]"
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def parse_direct(raw):
    m = re.search(r"ANSWER\s*:?\s*(.+)", str(raw or ""), re.I)
    txt = (m.group(1) if m else str(raw or "")).strip()
    return txt.split("\n")[0].strip().strip("`").strip()


def strip_code(raw):
    s = str(raw or "")
    s = re.sub(r"^\s*```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"```\s*$", "", s.strip())
    return s.strip()


# ---- prompt pieces ----

def schema_of(df):
    return "\n".join(f"  {c!r}: {df[c].dtype}" for c in df.columns)


def clip_cells(df, n=CELL_CHARS):
    """Long free-text cells shortened, so one transcript cannot fill a prompt."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(
                lambda v: (v[:n] + "…") if isinstance(v, str) and len(v) > n
                else v)
    return out


def render_rows(df):
    with pd.option_context("display.max_columns", 200,
                           "display.width", 10000,
                           "display.max_colwidth", 40):
        return df.to_string(index=False)


def fit_window(df, budget=WINDOW_CHARS):
    """Clip cells, then drop rows from the end until the block fits."""
    d = clip_cells(df)
    txt = render_rows(d)
    while len(txt) > budget and len(d) > 1:
        d = d.iloc[:max(1, int(len(d) * 0.6))]
        txt = render_rows(d)
    return d, txt


_TOK = re.compile(r"[a-z0-9]+")


def top_rows(df, question, k=TOPK):
    """Lexical retrieval over rows: how many question words each row contains.

    Deliberately simple and the same for every question. On an aggregate
    question no window of rows can hold the answer, and that is the point the
    arm is there to show rather than to hide.
    """
    q = set(_TOK.findall(str(question).lower()))
    if not q or df.empty:
        return df.head(k)
    sub = df.head(20000)
    joined = sub.astype(str).agg(" ".join, axis=1).str.lower()
    score = joined.map(lambda s: len(q & set(_TOK.findall(s))))
    order = np.lexsort((np.arange(len(score)), -score.to_numpy()))
    return sub.iloc[order[:k]]


# ---- arms ----

def run_direct(row, df, lite):
    nrows = int(df.shape[0])
    if lite:
        # The 20-row sample is bounded by construction, so the cap is not
        # applied here: clipping it would change prompts for no reason and
        # make this arm incomparable with the run already in the cache.
        win, txt = df, render_rows(df)
        note = "The table in full:"
        seen = 1.0
    else:
        win, txt = fit_window(top_rows(df, row["question"]))
        note = (f"The {len(win)} rows a retriever judged most relevant "
                f"(of {nrows}):")
        seen = len(win) / max(nrows, 1)
    prompt = P_DIRECT.format(
        name=row["dataset"], nrows=nrows, schema=schema_of(df),
        window_note=note, rows=txt, q=row["question"], atype=row["type"])
    raw = llm_client.call_llm(prompt, system=SYS_DIRECT, tag="db_direct",
                              max_tokens=MAX_TOKENS)
    return parse_direct(raw), {"rows_seen": len(win), "rows_total": nrows,
                               "frac_seen": seen}, [{"raw": str(raw)[:300]}]


def run_code(row, df, lite):
    nrows = int(df.shape[0])
    head = df.head(HEAD_ROWS)
    schema_block = (f"Table: {row['dataset']}\nRows: {nrows}\n"
                    f"Columns and types:\n{schema_of(df)}")
    prompt = P_CODE.format(name=row["dataset"], nrows=nrows,
                           schema=schema_of(df), nhead=len(head),
                           rows=render_rows(head), q=row["question"],
                           atype=row["type"])
    steps, value = [], None
    code, err = None, None
    for it in range(MAX_REPAIR + 1):
        if it == 0:
            raw = llm_client.call_llm(prompt, system=SYS_CODE, tag="db_code",
                                      max_tokens=MAX_TOKENS)
        else:
            raw = llm_client.call_llm(
                P_REPAIR.format(schema_block=schema_block, q=row["question"],
                                atype=row["type"], code=code, err=err),
                system=SYS_CODE, tag="db_repair", max_tokens=MAX_TOKENS)
        code = strip_code(raw)
        res, exc = sandbox.execute(code, {"df": df})
        steps.append({"iter": it, "code": code,
                      "exception_type": type(exc).__name__ if exc else None,
                      "error": str(exc)[:200] if exc else None})
        if exc is None:
            value = res
            break
        err = f"{type(exc).__name__}: {exc}"
        if it < MAX_REPAIR:
            time.sleep(SLEEP)
    return (normalise(value) if value is not None else None,
            {"rows_seen": len(head), "rows_total": nrows,
             "frac_seen": len(head) / max(nrows, 1),
             "iterations": len(steps)}, steps)


# `B-lite` is the code route on the 20-row sample, scored against
# `sample_answer`. It is the arm that separates "code wins because it can see
# all the rows" from "code wins because writing the operation down is easier
# than doing it in the head": on Lite both routes see the whole table, so any
# gap left there is not about visibility.
ARMS = {"A": (run_direct, False), "A-lite": (run_direct, True),
        "B": (run_code, False), "B-lite": (run_code, True)}


def main():
    global SLEEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dev", default=DEV)
    ap.add_argument("--tag", default="dev100")
    # Arm A's prompts carry up to 30 rows of a 74-column table -- about 7.8k
    # tokens a call, measured -- so the 250k tokens-per-minute cap, not the
    # 15 requests-per-minute one, is what it runs into. Pacing is widened for
    # that arm instead of shortening the prompt, which would make the calls
    # already in the cache incomparable with the rest.
    ap.add_argument("--sleep", type=float, default=SLEEP)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    with open(args.dev, encoding="utf-8") as f:
        spec = json.load(f)
    rows = [r for r in spec["questions"] if db.have_table(r["dataset"])]
    if args.limit:
        rows = rows[:args.limit]

    SLEEP = args.sleep
    fn, lite = ARMS[args.arm]
    rows = [r for r in rows if not lite or db.have_table(r["dataset"], True)]
    if lite:
        rows = [r for r in rows
                if r.get("sample_answer") not in (None, "")]

    path = os.path.join(CACHE, f"{args.arm}_{args.tag}.jsonl")
    done = {r["id"] for r in cache_io.load_rows(path)[0]}
    todo = [r for r in rows if r["qid"] not in done]
    print(f"арм {args.arm}, модель {llm_client.active_model()}, "
          f"питань {len(rows)}, зроблено {len(done)}, лишилось {len(todo)}")

    with open(path, "a", encoding="utf-8") as fh:
        for n, row in enumerate(todo, 1):
            df = db.table(row["dataset"], lite=lite)
            truth = db.truth_of(row, lite)
            llm_client.reset_stats()
            t0 = time.perf_counter()
            value, meta, steps = None, {}, []
            err = err_class = None
            try:
                value, meta, steps = fn(row, df, lite)
            except Exception as exc:
                err, err_class = f"{type(exc).__name__}: {exc}", type(exc).__name__
            ok = False
            if value is not None and not err:
                try:
                    ok = db.compare(value, truth, row["type"])
                except Exception:
                    ok = False
            out = {"id": row["qid"], "arm": args.arm,
                   "dataset": row["dataset"], "question": row["question"],
                   "type": row["type"], "truth": str(truth),
                   "value": None if value is None else str(value),
                   "correct": bool(ok), "steps": steps, **meta,
                   "elapsed_s": round(time.perf_counter() - t0, 2),
                   "tokens": (llm_client.get_stats() or {}).get("total_tokens"),
                   "error": err, "error_class": err_class}
            try:
                cache_io.write_row(fh, out)
            except cache_io.FakeRow as bad:
                print(f"  [{n}] {row['qid']}: транспорт, не записано — {bad}")
                break
            print(f"  [{n}/{len(todo)}] {row['dataset']:22s} {row['type']:14s} "
                  f"{'OK ' if ok else 'ні '} наш={str(value)[:28]:28s} "
                  f"еталон={str(truth)[:24]}")
            time.sleep(SLEEP)

    report(path, args.arm)


def report(path, arm=""):
    rows = cache_io.load_rows(path)[0]
    if not rows:
        print("порожньо")
        return
    ok = sum(1 for r in rows if r["correct"])
    print()
    print(f"=== {arm}: {ok}/{len(rows)} = {ok / len(rows):.1%} ===")
    print(f"  {'тип відповіді':16s} {'n':>4s} {'правильних':>11s} {'точність':>9s}")
    for t in db.TYPES:
        sub = [r for r in rows if r["type"] == t]
        if sub:
            k = sum(1 for r in sub if r["correct"])
            print(f"  {t:16s} {len(sub):4d} {k:11d} {k / len(sub):9.1%}")
    nov = sum(1 for r in rows if r["value"] is None)
    frac = [r.get("frac_seen") for r in rows if r.get("frac_seen") is not None]
    tok = sum(r.get("tokens") or 0 for r in rows)
    print(f"  без відповіді: {nov}")
    if frac:
        frac.sort()
        print(f"  частка таблиці, яку бачила модель: медіана "
              f"{frac[len(frac) // 2]:.1%}, мін {frac[0]:.2%}, макс {frac[-1]:.0%}")
    print(f"  викликів {len(rows)}, токенів {tok:,}")


if __name__ == "__main__":
    main()
