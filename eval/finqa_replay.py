"""Offline replay of the verification cascade over a cached FinQA run.

The generation pass is the expensive half and is done once (`eval.finqa_run`).
Everything here reads that cache: the candidate value, the signal vector and
the code at every iteration. Two cascade configurations are then applied to
exactly the same candidates, which makes the comparison paired — a
prerequisite for McNemar — and costs nothing.

  A   the four surrogate signals as they stand.
  B1  A plus the arithmetic invariants of the table, with a right of veto:
      a refutation rejects whatever the weighted score says.

B1 is expected to be narrow, and the number that says how narrow is reported
next to every rate rather than below it. `invariants.check_value` can only
speak when the answer IS an invariant's target cell; a value derived from
several cells leaves it UNDECIDABLE. On the synthetic document that was 18% of
queries, and FinQA answers are derived more often, not less. B1 is here to
show the plumbing works end to end, not to carry the result — that is B2's
job, and B2 needs the read-trace the sandbox does not yet produce.

Run:  python -m eval.finqa_replay --tag pilot2
"""
import argparse
import collections
import json
import os

import numpy as np
import pandas as pd

from core import calibration, decision, invariants
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")

THRESHOLD = 0.7

REFUTED = "REFUTED"
CONSISTENT = "CONSISTENT"
UNDECIDABLE = "UNDECIDABLE"


def load_cache(tag: str) -> list:
    path = os.path.join(CACHE_DIR, f"{tag}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"немає кешу {path}; спершу `python -m eval.finqa_run --tag {tag}`")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---- the invariant signal ----

def invariant_verdict(value, df: pd.DataFrame, inv_list: list, tol: float = 0.01) -> dict:
    """Test one candidate answer against the table's invariants.

    An invariant constrains a specific cell. The candidate is only comparable
    to it when it is close to one of that cell's values, so the check is:
    find an invariant whose target row holds a value near the candidate in
    some column, then test whether the members really sum to it.

    That is a weak hook, and deliberately not dressed up: when nothing
    matches, the verdict is UNDECIDABLE, which is neutral. An invariant
    refutes or says nothing; it never confirms.
    """
    if value is None or df is None or df.empty or not inv_list:
        return {"verdict": UNDECIDABLE, "reason": "no invariants or no value"}

    matrix, cols, labels = invariants._numeric_matrix(df)
    if matrix.size == 0:
        return {"verdict": UNDECIDABLE, "reason": "no numeric columns"}

    try:
        v = float(value)
    except (TypeError, ValueError):
        return {"verdict": UNDECIDABLE, "reason": "candidate not numeric"}

    for inv in inv_list:
        if inv["kind"] != "row_sum":
            continue
        t = inv["target_index"]
        if t >= matrix.shape[0]:
            continue
        for c in range(matrix.shape[1]):
            target_val = matrix[t, c]
            if not np.isfinite(target_val):
                continue
            if abs(v - target_val) > tol * max(abs(target_val), 1.0):
                continue
            # The candidate claims to be this cell. Does the identity hold?
            members = [i for i in inv["member_indices"] if i < matrix.shape[0]]
            expected = float(matrix[members, c].sum())
            if not np.isfinite(expected):
                continue
            if abs(v - expected) <= tol * max(abs(expected), 1.0):
                return {"verdict": CONSISTENT, "expected": expected, "given": v,
                        "invariant": invariants.describe(inv), "column": cols[c]}
            return {"verdict": REFUTED, "expected": expected, "given": v,
                    "invariant": invariants.describe(inv), "column": cols[c]}

    return {"verdict": UNDECIDABLE, "reason": "candidate matches no constrained cell"}


# ---- replay ----

def replay(rows: list, records_by_id: dict, threshold: float = THRESHOLD) -> pd.DataFrame:
    out = []
    for r in rows:
        rec = records_by_id.get(r["id"])
        df = fq.to_tables(rec).get("table") if rec else None

        inv_list, n_inv = [], 0
        if df is not None and not df.empty:
            det = invariants.detect({"table": df})
            inv_list = det["invariants"]
            n_inv = len(inv_list)

        signals = r.get("signals")
        candidate = r.get("candidate")

        # Config A: the decision the run actually took.
        conf = calibration.confidence(signals) if signals else None
        accept_a = bool(signals) and conf is not None and conf >= threshold

        # Config B1: A, plus a veto.
        verdict = invariant_verdict(candidate, df, inv_list)
        veto = verdict["verdict"] == REFUTED
        accept_b = accept_a and not veto

        out.append({
            "id": r["id"],
            "n_ops": r["n_ops"],
            "gold": r["gold"],
            "candidate": candidate,
            "candidate_correct": r.get("candidate_correct"),
            "candidate_scale_error": r.get("candidate_scale_error"),
            "status_run": r["status"],
            "abstain_reason": r.get("abstain_reason"),
            "confidence": conf,
            "n_invariants": n_inv,
            "inv_verdict": verdict["verdict"],
            "accept_A": accept_a,
            "accept_B1": accept_b,
            "reject_A": (not accept_a) if signals else True,
            "reject_B1": (not accept_b) if signals else True,
            "veto_fired": veto,
        })
    return pd.DataFrame(out)


def metrics(df: pd.DataFrame, config: str) -> dict:
    recs = [{"rejected": bool(row[f"reject_{config}"]),
             "candidate_correct": (None if pd.isna(row["candidate_correct"])
                                   else bool(row["candidate_correct"])),
             "abstain_reason": row.get("abstain_reason")}
            for _, row in df.iterrows()]
    return decision.rejection_metrics(recs)


def _fmt(x):
    return "—" if x is None else f"{x:.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    rows = load_cache(args.tag)
    records = {r["id"]: r for r in fq.load_split(args.split)}

    print("=" * 78)
    print(f"REPLAY КАСКАДУ НА КЕШІ «{args.tag}» — конфігурації A та B1")
    print("=" * 78)
    print(f"питань у кеші: {len(rows)}   поріг: {args.threshold}")
    print("Жодного звернення до LLM: усе рахується з кешу.")
    print()

    df = replay(rows, records, args.threshold)

    # --- execution accuracy -------------------------------------------------
    answered = df[df.status_run == "ACCEPT"]
    n = len(df)
    ea = int(((df.status_run == "ACCEPT") & (df.candidate_correct == True)).sum())  # noqa: E712
    scale = int(df.candidate_scale_error.fillna(False).astype(bool).sum())
    print("--- Execution Accuracy (допуск 1%, еталон exe_ans) ---")
    print(f"  EA (сувора):                  {ea}/{n} = {ea / n:.1%}")
    print(f"  з них ABSTAIN:                {(df.status_run == 'ABSTAIN').sum()}/{n}")
    print(f"  кандидатів з похибкою масштабу (степінь 10): {scale}/{n}")
    print()
    print("  EA НЕ порівнюється з FinAgent-RAG (76.81%): вони читають текст,")
    print("  ми — лише таблиці. Порівнювана метрика нижче.")

    # --- rejection ----------------------------------------------------------
    print()
    print("--- Rejection precision / recall (порівнюване з FinAgent-RAG) ---")
    print("  орієнтир: recall 58.4%, precision 76.2%")
    table = []
    for cfg in ("A", "B1"):
        m = metrics(df, cfg)
        table.append({
            "конфігурація": cfg,
            "покриття каскадом": _fmt(m["coverage"]),
            "оцінено": m["n_scored"],
            "хибних": m["n_wrong"],
            "відхилено": m["n_rejected"],
            "recall": _fmt(m["recall"]),
            "precision": _fmt(m["precision"]),
            "хибних відхилень": m["false_rejections"],
        })
    print(pd.DataFrame(table).to_string(index=False))

    m = metrics(df, "A")
    print()
    print("  ДО каскаду відсіяно (інші механізми, у recall НЕ входять):")
    print(f"    гейт області визначення: {m['n_gate_refusals']}")
    print(f"    код не виконався:        {m['n_no_code']}")
    for k, v in sorted(m["undecidable_by_reason"].items()):
        if k not in ("OUT_OF_SCOPE", "EXCEPTION"):
            print(f"    {k}: {v}")
    print(f"    каскад судив лише {m['n_scored']} з {m['n']} "
          f"({_fmt(m['coverage'])}) — recall вище стосується тільки їх")

    # --- applicability ------------------------------------------------------
    print()
    print("--- Застосовність вето інваріантів (без цього recall не читається) ---")
    vc = collections.Counter(df.inv_verdict)
    for k in (REFUTED, CONSISTENT, UNDECIDABLE):
        print(f"  {k:12s}: {vc.get(k, 0)}/{n} = {vc.get(k, 0) / n:.1%}")
    with_inv = int((df.n_invariants > 0).sum())
    print(f"  таблиць, де взагалі знайдено інваріант: {with_inv}/{n} = {with_inv / n:.1%}")
    print(f"  середньо інваріантів на таблицю: {df.n_invariants.mean():.1f}")
    if vc.get(REFUTED, 0) == 0:
        print()
        print("  ВЕТО НЕ СПРАЦЮВАЛО ЖОДНОГО РАЗУ. B1 = A, і різниця між ними")
        print("  дорівнює нулю не тому, що інваріанти не працюють, а тому що")
        print("  відповідь FinQA майже ніколи не є цілою клітинкою-підсумком.")
        print("  Це і є та стеля, заради виміру якої B1 і будувався.")

    out = os.path.join(HERE, f"finqa_replay_{args.tag}.csv")
    df.to_csv(out, index=False)
    print()
    print("Збережено:", out)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
