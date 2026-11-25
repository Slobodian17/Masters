"""Ceiling of single-operand repair: if ONE bound operand were re-bound, how
many answers that the direct route gets wrong would become right?

POST-HOC ANALYSIS OF THE DEV SET (confirm300). Gold is used ONLY to score;
nothing here is a method result.

Mechanism being bounded. The code route writes a program whose operands are
explicit addresses: a quoted row label and a quoted column name. Re-binding one
operand = replacing ONE of those string literals by another label / column of
the same table and re-executing. No model call.

Three candidate spaces, from widest to narrowest:
  ALL      any other row label or column            (perfect-selector ceiling)
  TWIN     a near-twin row (labels differ by <= 2 words) or another column
  COVER    the single swap that strictly increases how many of the question's
           content tokens the program's addresses cover -- a deterministic
           rule, so for this one the NET effect (fixed minus broken) is exact

Zero API calls.

Run:  python -m eval.finqa_operand_ceiling
"""
import collections
import math
import os
import re

from core import sandbox
from eval import cache_io
from eval.finqa_run import is_correct
from eval.finqa_twins import rows_with_twin, words
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")

_LIT = re.compile(r"""(['"])(.*?)\1""")
_STOP = set("""what was the of in to a an and for from is are by as at on
how much many did does do total change percent percentage value ratio
between than with its their which during year years per that this be
were if compared portion proportion average rate amount""".split())


def final_code(r):
    """Code of the step whose result became the candidate."""
    steps = r.get("steps") or []
    cand = r.get("candidate")
    for s in reversed(steps):
        if s.get("code") and s.get("result") is not None and cand is not None:
            try:
                if math.isclose(float(s["result"]), float(cand), rel_tol=1e-9,
                                abs_tol=1e-12):
                    return s["code"]
            except (TypeError, ValueError):
                continue
    for s in reversed(steps):
        if s.get("code"):
            return s["code"]
    return None


def as_float(v):
    try:
        if hasattr(v, "iloc"):
            v = v.iloc[0] if len(v) == 1 else None
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError, IndexError):
        return None


def q_tokens(question):
    return {w for w in words(question) if w not in _STOP and len(w) > 1}


def coverage(question, addresses):
    qt = q_tokens(question)
    cov = set()
    for a in addresses:
        cov |= words(a) & qt
    return len(cov)


def addresses_of(code, labels, cols):
    rows, columns = [], []
    for _, lit in _LIT.findall(code):
        low = lit.strip().lower()
        if not low:
            continue
        if lit in cols:
            columns.append(lit)
        elif any(low == l.lower() or low in l.lower() for l in labels):
            rows.append(lit)
    return rows, columns


def swap(code, old, new):
    return re.sub(r"(['\"])" + re.escape(old) + r"\1",
                  lambda m: m.group(1) + new.replace("\\", "\\\\") + m.group(1),
                  code)


def candidates(code, df, question):
    """[(space, kind, old, new, value, cover_gain)]"""
    labels = [str(v) for v in df.iloc[:, 0].tolist()]
    cols = [str(c) for c in df.columns[1:]]
    twins = rows_with_twin(labels)
    twin_of = collections.defaultdict(set)
    for i in twins:
        for j in twins:
            if i != j and len(words(labels[i]) ^ words(labels[j])) <= 2:
                twin_of[labels[i].lower()].add(labels[j])
    rows, columns = addresses_of(code, labels, cols)
    base_cov = coverage(question, rows + columns)
    out = []
    for old in dict.fromkeys(rows):
        match = [l for l in labels if old.lower() == l.lower()
                 or old.lower() in l.lower()]
        tw = set()
        for m in match:
            tw |= twin_of.get(m.lower(), set())
        for new in labels:
            if new.lower() == old.lower() or new in match:
                continue
            val = as_float(sandbox.execute(swap(code, old, new),
                                           {"table": df})[0])
            if val is None:
                continue
            new_cov = coverage(question, [new if a == old else a
                                          for a in rows] + columns)
            out.append(("TWIN" if new in tw else "ALL", "row", old, new, val,
                        new_cov - base_cov))
    for old in dict.fromkeys(columns):
        for new in cols:
            if new == old:
                continue
            val = as_float(sandbox.execute(swap(code, old, new),
                                           {"table": df})[0])
            if val is None:
                continue
            new_cov = coverage(question, rows + [new if a == old else a
                                                 for a in columns])
            out.append(("TWIN", "col", old, new, val, new_cov - base_cov))
    return out


def main():
    code_rows = {r["id"]: r for r in cache_io.load_rows(
        os.path.join(CACHE, "confirm300.jsonl"))[0]}
    dir_rows = {r["id"]: r for r in cache_io.load_rows(
        os.path.join(CACHE, "baseline_table_confirm300.jsonl"))[0]}
    records = {}
    for sp in ("test", "dev", "train"):
        for rec in fq.load_split(sp):
            records.setdefault(rec["id"], rec)

    stat = collections.Counter()
    rows_out = []
    for qid, r in code_rows.items():
        gold = r.get("gold")
        if gold is None:
            stat["no_gold"] += 1
            continue
        r1 = dir_rows.get(qid, {}).get("answer")
        r1_ok = bool(is_correct(r1, gold)) if r1 is not None else False
        r2 = r.get("candidate")
        r2_ok = bool(is_correct(r2, gold)) if r2 is not None else False
        code = final_code(r)
        rec = records.get(qid)
        df = fq.to_tables(rec).get("table") if rec else None
        cands = candidates(code, df, r["question"]) if code and df is not None \
            and not df.empty else []
        in_all = any(is_correct(c[4], gold) for c in cands)
        in_twin = any(is_correct(c[4], gold) for c in cands if c[0] == "TWIN")
        gain = [c for c in cands if c[5] > 0]
        best = max((c[5] for c in gain), default=0)
        top = [c for c in gain if c[5] == best]
        # deterministic COVER rule fires only when the best swap is unique in value
        fire = bool(top) and len({round(c[4], 6) for c in top}) == 1
        cover_val = top[0][4] if fire else None
        cover_ok = bool(is_correct(cover_val, gold)) if fire else None
        agree = r2 is not None and r1 is not None and \
            bool(is_correct(r2, r1)) if r1 not in (None, 0) else False
        rows_out.append(dict(id=qid, r1_ok=r1_ok, r2_ok=r2_ok, agree=agree,
                             has_code=bool(code), n_cands=len(cands),
                             gold_in_all=in_all, gold_in_twin=in_twin,
                             n_twin=sum(c[0] == "TWIN" for c in cands),
                             cover_fire=fire, cover_ok=cover_ok))

    import pandas as pd
    d = pd.DataFrame(rows_out)
    d.to_csv(os.path.join(HERE, "finqa_operand_ceiling_confirm300.csv"),
             index=False)
    n = len(d)
    print(f"questions with numeric gold: {n}  (skipped no-gold {stat['no_gold']})")
    print(f"R1 correct {d.r1_ok.sum()}  R2 correct {d.r2_ok.sum()}  "
          f"R2 has code {d.has_code.sum()}")
    w = d[~d.r1_ok]
    sw = d[~d.r1_ok & ~d.r2_ok & d.agree]
    for name, sub in (("R1 wrong", w), ("shared wrong (R1=R2)", sw)):
        print(f"\n--- {name}: n={len(sub)} ---")
        print(f"  gold reachable by ONE swap, any cell   : {sub.gold_in_all.sum()}")
        print(f"  gold reachable by ONE swap, twin/column: {sub.gold_in_twin.sum()}")
        print(f"  median candidates per question         : "
              f"{sub.n_cands.median():.0f}  (twin {sub.n_twin.median():.0f})")
    ok = d[d.r1_ok]
    print(f"\n--- R1 correct: n={len(ok)} (exposure to being broken) ---")
    print(f"  have >=1 twin/column alternative: {(ok.n_twin > 0).sum()}")
    print(f"  median twin candidates          : {ok.n_twin.median():.0f}")
    f = d[d.cover_fire]
    print("\n--- COVER rule (deterministic, replaces the answer when it fires) ---")
    print(f"  fires on {len(f)}  | on R1-wrong: {(~f.r1_ok).sum()} "
          f"| on R1-correct: {f.r1_ok.sum()}")
    fixed = (~f.r1_ok & f.cover_ok.astype(bool)).sum()
    broken = (f.r1_ok & ~f.cover_ok.astype(bool)).sum()
    print(f"  vs R1: fixed {fixed}, broken {broken}, net {fixed - broken:+d} "
          f"of {n}")
    oracle = (d.r1_ok | d.gold_in_twin).sum()
    print(f"\nperfect selector over {{R1}} + twin swaps: {oracle}/{n} = "
          f"{oracle / n:.1%}  (R1 alone {d.r1_ok.sum() / n:.1%})")


if __name__ == "__main__":
    main()
