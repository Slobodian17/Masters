"""Does what the program READ match what the query ASKED?

A different question from the one invariants answer. Invariants test whether
numbers are arithmetically consistent; this tests whether they are the right
numbers at all. The distinction matters because the dominant error on FinQA is
neither: 78% of accepted wrong answers read a genuine row from the right
table, just not the one the question was about. Arithmetic over genuine cells
is consistent with invariants derived from those same cells, so nothing in the
invariant line can refute it. Correspondence can.

Two checks, both over the set of cells the program read:

  period  the query names a year or quarter, the read cells name a different
          one and no overlapping one
  metric  no content term of the query resembles any label of a read row

BOTH DIRECTIONS ARE MEASURED. A catch rate on wrong answers means nothing
without the false-alarm rate on correct ones, so every figure below is
reported as a pair.

THE READ SET IS INFERRED, NOT TRACED (see eval.finqa_errors). Everything here
is therefore an ESTIMATE of what a real read-trace would support.

Run:  python -m eval.finqa_correspondence
"""
import argparse
import collections
import difflib
import os

import pandas as pd

from core import scope_gate
from core.text_utils import tokenize
from eval.finqa_errors import final_code, infer_read_set
from eval.finqa_replay import load_cache
from ingestion import finqa_loader as fq
from retrieval.temporal_index import parse_periods

HERE = os.path.dirname(os.path.abspath(__file__))

NO_VERDICT = None


def read_texts(read: dict, df: pd.DataFrame) -> tuple:
    """(row labels read, column names read) as plain strings."""
    if df is None or df.empty:
        return [], []
    labels = [str(v) for v in df.iloc[:, 0].tolist()]
    rows = [labels[i] for i in read["rows"] if i < len(labels)]
    return rows, [str(c) for c in read["cols"]]


def period_check(query: str, read: dict, df: pd.DataFrame):
    """-> True (mismatch), False (consistent), or None (cannot tell).

    None is returned whenever either side names no period. Silence is the
    right answer there: a query without a period constrains nothing, and a
    table whose columns are '12/11' says nothing this parser can read.
    """
    q = parse_periods(query)
    q_set = set(q.get("years") or []) | set(q.get("quarters") or [])
    if not q_set:
        return NO_VERDICT

    rows, cols = read_texts(read, df)
    r = parse_periods(" ".join(rows + cols))
    r_set = set(r.get("years") or []) | set(r.get("quarters") or [])
    if not r_set:
        return NO_VERDICT

    return not bool(q_set & r_set)


def metric_check(query: str, read: dict, df: pd.DataFrame,
                 cutoff: float = None):
    """-> True (mismatch), False (consistent), or None (cannot tell).

    The same fuzzy comparison the scope gate uses, applied to the rows that
    were actually read instead of to the whole table.
    """
    content = scope_gate._content_tokens(query)
    if not content:
        return NO_VERDICT

    rows, _ = read_texts(read, df)
    if not rows:
        return NO_VERDICT

    read_tokens = set()
    for label in rows:
        read_tokens |= tokenize(label)
    if not read_tokens:
        return NO_VERDICT

    cutoff = cutoff if cutoff is not None else scope_gate.TOKEN_SIMILARITY \
        if hasattr(scope_gate, "TOKEN_SIMILARITY") else 0.8
    pool = sorted(read_tokens)
    for t in content:
        if difflib.get_close_matches(t, pool, n=1, cutoff=cutoff):
            return False
    return True


def evaluate(tags: list, split: str = "test") -> pd.DataFrame:
    records = {r["id"]: r for r in fq.load_split(split)}
    out = []
    for tag in tags:
        for r in load_cache(tag):
            if r["status"] != "ACCEPT" or r["answer_correct"] is None:
                continue
            rec = records.get(r["id"])
            df = fq.to_tables(rec).get("table") if rec else None
            read = infer_read_set(final_code(r) or "", df)
            out.append({
                "tag": tag,
                "id": r["id"],
                "correct": bool(r["answer_correct"]),
                "candidate": r["candidate"],
                "gold": r["gold"],
                "n_read_rows": len(read["rows"]),
                "period_mismatch": period_check(r["question"], read, df),
                "metric_mismatch": metric_check(r["question"], read, df),
                "scale": bool(r.get("candidate_scale_error")),
            })
    return pd.DataFrame(out)


def _rate(df, col, want_correct):
    sub = df[df.correct == want_correct]
    verdicts = sub[sub[col].notna()]
    flagged = int((verdicts[col] == True).sum())  # noqa: E712
    return flagged, len(verdicts), len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="ab_no_v2,ab_yes_v2")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    df = evaluate(tags, args.split)
    n_wrong = int((~df.correct).sum())
    n_right = int(df.correct.sum())

    print("=" * 78)
    print("ВІДПОВІДНІСТЬ ПРОЧИТАНОГО ЗАПИТУ — ОЦІНКА, не вимір")
    print("=" * 78)
    print("Множина прочитаних клітинок ВИВЕДЕНА з коду, не трасована.")
    print("Числа показують, що підтримав би справжній трейс, не що він дасть.")
    print()
    print(f"прийнятих відповідей: {len(df)}   хибних {n_wrong}, правильних {n_right}")
    print()

    print("--- Спрацювання: на хибних (ціль) проти правильних (ціна) ---")
    rows = []
    for col, name in (("period_mismatch", "період"),
                      ("metric_mismatch", "показник"),
                      (None, "будь-яка з двох")):
        if col is None:
            any_col = df[["period_mismatch", "metric_mismatch"]].eq(True).any(axis=1)
            has = df[["period_mismatch", "metric_mismatch"]].notna().any(axis=1)
            fw = int((any_col & ~df.correct).sum())
            vw = int((has & ~df.correct).sum())
            fr = int((any_col & df.correct).sum())
            vr = int((has & df.correct).sum())
        else:
            fw, vw, _ = _rate(df, col, False)
            fr, vr, _ = _rate(df, col, True)
        rows.append({
            "перевірка": name,
            "спіймано хибних": f"{fw}/{n_wrong} = {fw/n_wrong:.0%}" if n_wrong else "—",
            "вердикт був (хибні)": f"{vw}/{n_wrong}",
            "ХИБНИХ спрацювань": f"{fr}/{n_right} = {fr/n_right:.0%}" if n_right else "—",
            "вердикт був (правильні)": f"{vr}/{n_right}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("  «вердикт був» — на скількох випадках перевірка взагалі висловилась.")
    print("  Там, де вона мовчить, вона нічого не коштує і нічого не дає.")

    print()
    print("--- Серед хибних: період чи показник ---")
    w = df[~df.correct]
    c = collections.Counter()
    for _, r in w.iterrows():
        p, m = r["period_mismatch"], r["metric_mismatch"]
        if p is True and m is True:
            c["обидва"] += 1
        elif p is True:
            c["лише період"] += 1
        elif m is True:
            c["лише показник"] += 1
        elif p is None and m is None:
            c["жодна не висловилась"] += 1
        else:
            c["обидві мовчать або згодні"] += 1
    for k, v in c.most_common():
        print(f"  {k:26s} {v:2d}  ({v/len(w):.0%})")

    print()
    print("--- Ті самі числа лише на робочій конфігурації (з конвенціями) ---")
    sub = df[df.tag == "ab_yes_v2"]
    if len(sub):
        nw = int((~sub.correct).sum())
        nr = int(sub.correct.sum())
        anyc = sub[["period_mismatch", "metric_mismatch"]].eq(True).any(axis=1)
        print(f"  хибних {nw}, правильних {nr}")
        print(f"  спіймано хибних: {int((anyc & ~sub.correct).sum())}/{nw}")
        print(f"  ХИБНИХ спрацювань: {int((anyc & sub.correct).sum())}/{nr}")

    out = os.path.join(HERE, "finqa_correspondence.csv")
    df.to_csv(out, index=False)
    print()
    print("Збережено:", out)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    main()
