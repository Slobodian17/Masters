"""Arbitration: one call, only where the two routes disagree.

The system rule this implements:

    R1 == R2 within 1%   -> that answer
    R2 has no number     -> R1
    otherwise            -> the arbiter's answer

WHY IT RUNS SEPARATELY, AND WHY THAT MATTERS. The arbiter sees both routes'
outputs, so its input is not independent of R1. Route independence is what the
agreement check rests on (5.18, and the confirmed precision of 93.9% in
5.29.1), so R1 must never enter R2's generation. Running arbitration as a
third step, after both caches exist, keeps that intact: R2 was generated
blind, the agreement signal is still computed on blind outputs, and only the
DELIVERED answer combines them.

The consequence has to be stated rather than discovered: once the delivered
answer is the arbiter's, agreement can no longer verify it -- the answer is
derived from the agreement. Arbitration and the audit mode are two different
products of the same two routes, not one.

A ceiling from 5.29.3 does NOT bind this: the perfect-oracle arbiter was worth
only +1.4 pp over "always R1" on the OLD code route. With the code route given
the table's values, disagreements should contain more cases where the code is
the right one, and the ceiling has to be re-measured, not assumed.

Run:  python -m eval.finqa_arbiter --r2-tag confirm300_values --n 100
"""
import argparse
import json
import os
import re
import time

import pandas as pd

from eval import cache_io
from eval.finqa_baseline import serialise
from eval.finqa_run import auto_pace, is_correct
from generation import llm_client
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")
TOL = 0.01

MAX_TOKENS = int(os.environ.get("VKSK_ARBITER_MAX_TOKENS", "2400"))

SYSTEM_PLAIN = (
    "You are a financial analyst resolving a disagreement between two "
    "independent attempts at the same table question. Decide which value is "
    "correct, or give the correct value if neither is. Answer with a single "
    "number and nothing else."
)

SYSTEM_MARKED = (
    "You are a financial analyst resolving a disagreement between two "
    "independent attempts at the same table question. Decide which value is "
    "correct, or give the correct value if neither is. End your reply with a "
    "final line of exactly this form:\nANSWER: <number>"
)


def system_prompt() -> str:
    from eval.finqa_baseline import use_marker
    return SYSTEM_MARKED if use_marker() else SYSTEM_PLAIN


SYSTEM = SYSTEM_PLAIN

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_answer(text):
    """Marker-aware, for the same reason the direct route is.

    The arbiter reasons in prose before answering, so taking the FIRST number
    picks a year out of a column header. Observed on all 17 disagreements of
    the flash-lite run: every arbitration was scored NEITHER and 0 of 17
    correct, because the parser never read the answer.
    """
    from eval.finqa_baseline import parse_answer as _parse
    return _parse(text)


def build_prompt(question, df, r1, r2, code, instructions):
    """Both candidates, and the code that produced one of them.

    The code and its exact executed value are included because that is what
    makes the disagreement diagnosable: a reader of the code can see which
    cells it took, which is the commonest way it goes wrong (5.30.2).
    """
    return (
        f"Table:\n{serialise(df)}\n\n"
        f"{instructions}\n\n"
        f"Question: {question}\n\n"
        f"Attempt A read the table directly and answered: {r1}\n\n"
        f"Attempt B wrote this Pandas code:\n```python\n{code}\n```\n"
        f"Executing it returned exactly: {r2}\n\n"
        "The two disagree. Decide which formula and which cells are correct, "
        "then give the correct answer.\n"
        "Answer with one number only."
    )


def cell_trace(code, df):
    """Which cells the code actually read, with their values.

    Version 1 handed the arbiter the code and asked it to read. Measured on dev
    (5.30.2), the commonest failure is not arithmetic but reading the wrong
    row, so the useful thing to show is not the code but WHICH CELLS it took --
    the model then compares two concrete readings instead of simulating a
    program in its head.

    Derived from the string literals in the code, matched against the table's
    labels and column names. Reported as what it is: literals found, not a
    dataflow analysis. A literal that matches nothing is shown too, because a
    label the model invented is itself the diagnosis.
    """
    if df is None or df.empty:
        return "(таблиця недоступна)"
    lits = []
    for m in re.finditer(r"""['"]([^'"]{1,90})['"]""", code or ""):
        v = m.group(1)
        if v not in lits:
            lits.append(v)
    labels = [str(x) for x in df.iloc[:, 0].tolist()]
    cols = [str(c) for c in df.columns]
    rows_hit = [l for l in lits if l in labels]
    cols_hit = [l for l in lits if l in cols]
    unknown = [l for l in lits if l not in labels and l not in cols]

    out = []
    if rows_hit and cols_hit:
        for rl in rows_hit:
            i = labels.index(rl)
            vals = ", ".join(
                f"{c}={df.iloc[i][c]}" for c in cols_hit if c in df.columns)
            out.append(f"  row '{rl}': {vals}")
    elif rows_hit:
        for rl in rows_hit:
            i = labels.index(rl)
            vals = ", ".join(f"{c}={df.iloc[i][c]}" for c in cols[1:])
            out.append(f"  row '{rl}': {vals}")
    if unknown:
        out.append(f"  labels not present in the table: {unknown}")
    if not out:
        out.append("  (no table labels appear as literals in the code)")
    return "\n".join(out)


# Formula shapes FinQA's gold programs actually use, stated so the arbiter can
# check which one the question calls for instead of inferring it. Measured
# reason for including this: the largest error class on dev is a ratio computed
# against the wrong denominator (5.30.2 -- BASE, 12 of 53 agreed-wrong), which
# is a choice-of-formula error, not an arithmetic one.
FORMULA_SHEET = """Formula shapes used by this dataset:
- "percentage change / increase / decrease / growth from A to B" = (B - A) / A
- "percent of total" / "what portion" = part / total
- "return on investment", "total return" with an index based at 100 =
  (value - 100) / 100
- "difference between A and B" = A - B  (a plain difference, NOT divided)
- "change in X from Y to Z" = X_Z - X_Y  (a plain difference)
- a percentage is reported as a plain fraction: 0.141, not 14.1
Pick the shape the question asks for. A question asking for a PERCENTAGE must
divide; a question asking for a CHANGE or DIFFERENCE must not."""


def build_prompt_v2(question, df, r1, r2, code, instructions):
    """Both readings side by side, plus the formula shapes."""
    return (
        f"Table:\n{serialise(df)}\n\n"
        f"{instructions}\n\n"
        f"{FORMULA_SHEET}\n\n"
        f"Question: {question}\n\n"
        f"Reading A read the table directly and answered: {r1}\n\n"
        f"Reading B wrote code and executed it, returning exactly: {r2}\n"
        f"The cells reading B took:\n{cell_trace(code, df)}\n\n"
        "The two disagree. State which formula shape the question asks for and "
        "which cells are the right ones, then give the answer.\n"
        "Answer with one number only."
    )


def decide(r1, r2):
    """The SYSTEM rule. Deliberately NOT the pre-registered rule of 5.26.2.

    One branch differs, and the difference is the point: 5.26.2 treats a
    disagreement of exactly 10^k as ACCEPT with a UNIT flag and delivers the
    code's value. That was defensible while the code was the designated
    answerer, but a 10^k gap is precisely the case where one of the two has the
    unit wrong and neither is more trustworthy a priori -- so here it goes to
    arbitration instead of being decided by fiat.

    Verified against eval.finqa_confirm.decide: for (0.141, 14.1) the
    pre-registered rule returns ACCEPT/unit=True and this one returns ARBITER.
    Both are kept, because 5.26 has to stay reproducible as it was measured.
    """
    if r2 is None:
        return "R1", None
    if r1 is None:
        return "R2", None
    if abs(r1 - r2) <= TOL * max(abs(r2), 1.0):
        return "AGREE", r2
    return "ARBITER", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2-tag", required=True, help="кеш нового R2")
    ap.add_argument("--r1-tag", default="baseline_table_confirm300",
                    help="кеш прямого маршруту (наявний, не перезапускається)")
    ap.add_argument("--ids", default=os.path.join(HERE, "finqa_dev100.ids"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pace", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--version", type=int, default=1, choices=[1, 2],
                    help="1: код і його значення; 2: слід клітинок + "
                         "шпаргалка формул")
    args = ap.parse_args()
    pace = args.pace if args.pace is not None else auto_pace() / 2

    if llm_client.active_provider() == "mock":
        print("MOCK не містить питань FinQA — потрібен GROQ_API_KEY.")
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ.setdefault(llm_client.LEDGER_ENV,
                          os.path.join(CACHE_DIR, "token_ledger.jsonl"))

    r2rows, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.r2_tag}.jsonl"))
    r1rows, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.r1_tag}.jsonl"))
    r1 = {r["id"]: r for r in r1rows}
    recs = {r["id"]: r for r in fq.load_split(args.split)}
    from eval.finqa_run import FINQA_INSTRUCTIONS

    with open(args.ids, encoding="utf-8") as f:
        want = {l.strip() for l in f if l.strip()}
    suffix = "" if args.version == 1 else f"_v{args.version}"
    out_path = args.out or os.path.join(
        CACHE_DIR, f"arbiter_{args.r2_tag}{suffix}.jsonl")

    todo = []
    for r in r2rows:
        if r["id"] not in want or r["id"] not in r1:
            continue
        v2 = r.get("candidate")
        if v2 is None:
            v2 = r.get("answer")
        v1 = r1[r["id"]].get("answer")
        route, _ = decide(v1, v2)
        if route == "ARBITER":
            todo.append((r, v1, v2))

    print("=" * 78)
    print(f"РОЗБІР РОЗБІЖНОСТІ на питаннях {os.path.basename(args.ids)}")
    print("=" * 78)
    print(f"R2: {args.r2_tag} ({len(r2rows)} рядків)   "
          f"R1: {args.r1_tag} ({len(r1rows)})")
    print(f"розбіжностей, що потребують виклику: {len(todo)}")
    print(f"темп: {pace} с   оцінка ~{len(todo) * (pace + 3) / 60:.0f} хв")
    print()

    done = {r["id"] for r in cache_io.load_rows(out_path)[0]}
    if done:
        print(f"вже пораховано: {len(done)}, продовжую")
    n_new = n_fake = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, (r, v1, v2) in enumerate(todo, start=1):
            if r["id"] in done:
                continue
            rec = recs.get(r["id"])
            df = fq.to_tables(rec).get("table") if rec else None
            if df is None or df.empty:
                continue
            code = next((s.get("code") for s in reversed(r.get("steps") or [])
                         if s.get("code")), "")
            llm_client.reset_stats()
            t0 = time.perf_counter()
            try:
                raw = llm_client.call_llm(
                    (build_prompt if args.version == 1 else build_prompt_v2)(
                        r["question"], df, v1, v2, code, FINQA_INSTRUCTIONS),
                    system=system_prompt(), tag="arbiter",
                    max_tokens=MAX_TOKENS)
                err = ecls = None
            except Exception as exc:
                raw = ""
                err, ecls = f"{type(exc).__name__}: {exc}", type(exc).__name__
            st = llm_client.get_stats()
            val = parse_answer(raw)
            row = {"id": r["id"], "version": args.version,
                   "question": r["question"],
                   "gold": r.get("gold"), "r1": v1, "r2": v2,
                   "arbiter": val, "raw": str(raw)[:900],
                   "picked": ("R1" if val is not None and v1 is not None
                              and abs(val - v1) <= TOL * max(abs(v1), 1.0)
                              else ("R2" if val is not None and v2 is not None
                                    and abs(val - v2) <= TOL * max(abs(v2), 1.0)
                                    else ("NEITHER" if val is not None else None))),
                   "tokens": st.get("total_tokens", 0),
                   "prompt_tokens": st.get("prompt_tokens", 0),
                   "completion_tokens": st.get("completion_tokens", 0),
                   "reserved_tokens": MAX_TOKENS * st.get("calls", 0),
                   "truncated": llm_client.was_truncated(),
                   "empty_reply": not str(raw).strip(),
                   "elapsed_s": round(time.perf_counter() - t0, 2),
                   "error": err, "error_class": ecls}
            try:
                cache_io.write_row(f, row, on_unknown=lambda m: print(
                    f"  УВАГА: нерозпізнане походження — {m}"))
            except cache_io.FakeRow as exc:
                n_fake += 1
                print(f"[{i}/{len(todo)}] ВИКЛИК НЕ ВІДБУВСЯ, рядок НЕ записано: {exc}")
                if "tpd" in str(exc).lower() or "tokens per day" in str(exc).lower():
                    print("  добовий ліміт вичерпано — зупиняюсь")
                    break
                continue
            n_new += 1
            # No correctness printed: a run may be blind.
            print(f"[{i}/{len(todo)}] обрано={row['picked']}  "
                  f"tok={row['tokens']}  {row['elapsed_s']}s")
            if args.limit and n_new >= args.limit:
                print(f"  пробний ліміт {args.limit} досягнуто — зупиняюсь")
                break
            if i < len(todo):
                time.sleep(pace)

    a = cache_io.audit(out_path)
    print()
    print(f"записано {n_new}, викликів не відбулось {n_fake}")
    print(f"аудит: рядків {a['rows']}, справжніх {a['real']}, "
          f"фальшивих {a['transport']}, дублікатів {a['duplicates']}")
    print("Збережено:", out_path)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
