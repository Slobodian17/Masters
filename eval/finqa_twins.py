"""Near-twin rows: is "the model was in a confusable spot" a usable signal?

A different question from correctness. Instead of asking whether an answer is
right, this asks whether the program was reading a row that has a close
neighbour in the same table -- the situation in which the model picked the
wrong one in 78% of accepted wrong answers (5.16).

    прочитано: 'equity compensation plans approved by security holders'
    еталон:    'equity compensation plans not approved by security holders'

PRE-REGISTERED DEFINITION. Fixed before any result was seen, because a
threshold chosen after looking at the outcome is fitted, not measured.

    Two distinct rows of one table are near-twins when their raw labels,
    lowercased and split on non-alphanumerics, have a symmetric word-set
    difference of at most TWINS_MAX_DIFF = 2 words, and each label has at
    least TWINS_MIN_WORDS = 2 words.

The threshold 2 covers all three motivating cases: "not" (1 word),
"1"/"2" (2 words), "net"/"gross" (2 words). Sensitivity at 1 and 3 is
reported for reference; 2 remains the headline.

A catch rate on wrong answers is meaningless without the same rate on correct
ones, so both are always reported. And if near-twins are simply everywhere in
financial tables, the signal carries no information however well it correlates
-- hence the third figure, over all tables.

THE READ SET IS INFERRED, NOT TRACED. This is an ESTIMATE.

Run:  python -m eval.finqa_twins
"""
import argparse
import collections
import os
import re

import pandas as pd

from eval.finqa_errors import final_code, infer_read_set
from eval.finqa_replay import load_cache
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))

# --- pre-registered, do not tune ---
TWINS_MAX_DIFF = 2
TWINS_MIN_WORDS = 2
SENSITIVITY = (1, 2, 3)

_WORD = re.compile(r"[a-z0-9]+")


def words(label: str) -> set:
    return set(_WORD.findall(str(label).lower()))


def twin_pairs(labels: list, max_diff: int = TWINS_MAX_DIFF,
               min_words: int = TWINS_MIN_WORDS) -> list:
    """[(i, j, differing words)] for every near-twin pair in one table."""
    ws = [words(l) for l in labels]
    out = []
    for i in range(len(labels)):
        if len(ws[i]) < min_words:
            continue
        for j in range(i + 1, len(labels)):
            if len(ws[j]) < min_words:
                continue
            if str(labels[i]).strip().lower() == str(labels[j]).strip().lower():
                continue
            diff = ws[i] ^ ws[j]
            if 0 < len(diff) <= max_diff or (not diff):
                out.append((i, j, sorted(diff)))
    return out


def rows_with_twin(labels: list, max_diff: int = TWINS_MAX_DIFF) -> dict:
    """{row index: [differing words with its closest twin]}"""
    out = {}
    for i, j, diff in twin_pairs(labels, max_diff):
        out.setdefault(i, diff)
        out.setdefault(j, diff)
    return out


def evaluate(tags: list, split: str, max_diff: int) -> pd.DataFrame:
    records = {r["id"]: r for r in fq.load_split(split)}
    out = []
    for tag in tags:
        for r in load_cache(tag):
            if r["status"] != "ACCEPT" or r["answer_correct"] is None:
                continue
            rec = records.get(r["id"])
            df = fq.to_tables(rec).get("table") if rec else None
            if df is None or df.empty:
                continue
            labels = [str(v) for v in df.iloc[:, 0].tolist()]
            twins = rows_with_twin(labels, max_diff)
            read = infer_read_set(final_code(r) or "", df)
            touched = [i for i in read["rows"] if i in twins]
            out.append({
                "tag": tag,
                "id": r["id"],
                "correct": bool(r["answer_correct"]),
                "n_read_rows": len(read["rows"]),
                "read_has_twin": bool(touched),
                "table_has_twin": bool(twins),
                "example": ",".join(twins[touched[0]]) if touched else "",
            })
    return pd.DataFrame(out)


def table_prevalence(split: str, max_diff: int) -> dict:
    recs = [r for r in fq.load_split(split)
            if fq.evidence_source(r) == "table" and r.get("table")]
    seen, with_twin, rows_total, rows_twin = set(), 0, 0, 0
    for rec in recs:
        df, _ = fq.to_frame(rec["table"])
        if df.empty or df.shape[0] < 2:
            continue
        key = id(rec)
        labels = [str(v) for v in df.iloc[:, 0].tolist()]
        t = rows_with_twin(labels, max_diff)
        seen.add(key)
        with_twin += bool(t)
        rows_total += len(labels)
        rows_twin += len(t)
    return {"tables": len(seen), "with_twin": with_twin,
            "rows_total": rows_total, "rows_twin": rows_twin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="ab_no_v2,ab_yes_v2")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    print("=" * 78)
    print("МАЙЖЕ-ДВІЙНИКИ: чи корисний сигнал «модель була у сплутуваному місці»")
    print("=" * 78)
    print(f"ЗАРЕЄСТРОВАНО НАПЕРЕД: різниця <= {TWINS_MAX_DIFF} слів, "
          f"мітка >= {TWINS_MIN_WORDS} слів.")
    print("Поріг зафіксовано до перегляду результатів. ОЦІНКА, не вимір:")
    print("множина прочитаних клітинок виведена з коду, а не трасована.")
    print()

    df = evaluate(tags, args.split, TWINS_MAX_DIFF)
    n_w = int((~df.correct).sum())
    n_r = int(df.correct.sum())
    a = int((df.read_has_twin & ~df.correct).sum())
    b = int((df.read_has_twin & df.correct).sum())

    print(f"прийнятих відповідей: {len(df)} — хибних {n_w}, правильних {n_r}")
    print()
    print("--- ГОЛОВНЕ ПОРІВНЯННЯ (поріг 2) ---")
    print(f"  а) хибні, де прочитаний рядок має двійника:      "
          f"{a}/{n_w} = {a/n_w:.0%}")
    print(f"  б) ПРАВИЛЬНІ, де прочитаний рядок має двійника:  "
          f"{b}/{n_r} = {b/n_r:.0%}")
    lift = (a / n_w) - (b / n_r) if n_w and n_r else 0.0
    print(f"     різниця (а) − (б): {lift:+.0%}")

    print()
    print("--- в) Наскільки двійники поширені взагалі ---")
    prev = table_prevalence(args.split, TWINS_MAX_DIFF)
    print(f"  таблиць у підмножині: {prev['tables']}")
    print(f"  таблиць з хоча б однією парою двійників: "
          f"{prev['with_twin']}/{prev['tables']} = {prev['with_twin']/prev['tables']:.0%}")
    print(f"  рядків, що мають двійника: "
          f"{prev['rows_twin']}/{prev['rows_total']} = {prev['rows_twin']/prev['rows_total']:.0%}")

    print()
    print("--- Чутливість до порога (довідково) ---")
    rows = []
    for d in SENSITIVITY:
        dd = evaluate(tags, args.split, d)
        nw = int((~dd.correct).sum())
        nr = int(dd.correct.sum())
        aa = int((dd.read_has_twin & ~dd.correct).sum())
        bb = int((dd.read_has_twin & dd.correct).sum())
        pv = table_prevalence(args.split, d)
        rows.append({
            "різниця слів <=": d,
            "хибні з двійником": f"{aa}/{nw} = {aa/nw:.0%}" if nw else "—",
            "правильні з двійником": f"{bb}/{nr} = {bb/nr:.0%}" if nr else "—",
            "різниця": f"{(aa/nw - bb/nr):+.0%}" if nw and nr else "—",
            "таблиць з двійниками": f"{pv['with_twin']/pv['tables']:.0%}",
            "": "  <- зареєстровано" if d == TWINS_MAX_DIFF else "",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("--- Які слова розрізняють двійників (прочитані рядки) ---")
    c = collections.Counter()
    for e in df[df.read_has_twin].example:
        for w in str(e).split(","):
            if w:
                c[w] += 1
    for k, v in c.most_common(12):
        print(f"    {k}: {v}")

    out = os.path.join(HERE, "finqa_twins.csv")
    df.to_csv(out, index=False)
    print()
    print("Збережено:", out)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    main()
