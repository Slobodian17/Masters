"""DataBench (SemEval-2025 Task 8), with the organisers' own comparison.

Answer checking is NOT reimplemented here. `databench_eval.Evaluator` is the
package the task organiser published and used for the official scoring, and
its `default_compare` is called directly: five answer types, numbers truncated
to two decimals, lists compared without regard to order. Writing our own
comparison would make every accuracy number in this project incomparable with
the published leaderboard, and the five types are exactly where a hand-rolled
comparison goes wrong.

One detail that is easy to get wrong and changes the numbers: DataBench Lite
is scored against `sample_answer`, the answer recomputed on the 20-row sample,
not against `answer`. A Lite arm checked against `answer` would be scored on a
question it was never asked.

Tables and questions are cached under `data/databench/` so a run needs the
network once.

Run:  python -m ingestion.databench_loader          # report what is there
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "data", "databench")
TABLES = os.path.join(CACHE, "tables")

HF = "hf://datasets/cardiffnlp/databench/data"
TYPES = ("boolean", "category", "number", "list[category]", "list[number]")


def _ensure():
    os.makedirs(TABLES, exist_ok=True)


# ---- questions ----

def questions(split: str = "train", config: str = "semeval") -> list:
    """QA rows as plain dicts. Cached as json after the first download."""
    _ensure()
    path = os.path.join(CACHE, f"qa_{config}_{split}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/databench", name=config, split=split)
    rows = [dict(r) for r in ds]
    for i, r in enumerate(rows):
        r["qid"] = f"{config}/{split}/{i}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    return rows


# ---- tables ----

def table(ds_id: str, lite: bool = False) -> pd.DataFrame:
    """One dataset's table. `lite` gives the 20-row sample."""
    _ensure()
    name = "sample" if lite else "all"
    local = os.path.join(TABLES, f"{ds_id}__{name}.parquet")
    if not os.path.exists(local):
        df = pd.read_parquet(f"{HF}/{ds_id}/{name}.parquet")
        df.to_parquet(local, index=False)
        return df
    return pd.read_parquet(local)


def have_table(ds_id: str, lite: bool = False) -> bool:
    name = "sample" if lite else "all"
    return os.path.exists(os.path.join(TABLES, f"{ds_id}__{name}.parquet"))


# ---- the organisers' comparison ----

_EVAL = None


def compare(value, truth, semantic: str) -> bool:
    """The official `default_compare`, called as the organisers wrote it."""
    global _EVAL
    if _EVAL is None:
        from databench_eval import Evaluator
        # `qa=...` only matters for Evaluator.eval(); default_compare is
        # stateless, and loading the full QA set here would cost a download
        # for nothing.
        _EVAL = Evaluator.__new__(Evaluator)
    return bool(_EVAL.default_compare(value, truth, str(semantic)))


def truth_of(row: dict, lite: bool = False):
    """`sample_answer` for Lite, `answer` otherwise -- see the module docstring."""
    return row["sample_answer"] if lite else row["answer"]


# ---- selection ----

def stratified(rows: list, n: int, seed: int = 20261031,
               per_type: dict = None) -> list:
    """`n` questions spread evenly over the five answer types.

    Even coverage matters because the types are not equally hard and not
    equally represented: a sample drawn at random would let one type dominate
    and make the arms' accuracies say more about the mix than about the method.
    """
    import random
    rnd = random.Random(seed)
    by_type = {t: [r for r in rows if r.get("type") == t] for t in TYPES}
    for t in TYPES:
        rnd.shuffle(by_type[t])
    want = per_type or {}
    out, i = [], 0
    while len(out) < n:
        added = False
        for t in TYPES:
            cap = want.get(t)
            got = sum(1 for r in out if r["type"] == t)
            if cap is not None and got >= cap:
                continue
            if i < len(by_type[t]) and len(out) < n:
                out.append(by_type[t][i])
                added = True
        if not added:
            break
        i += 1
    return out


def main():
    for config, split in (("semeval", "train"), ("semeval", "dev")):
        try:
            rows = questions(split, config)
        except Exception as exc:
            print(f"{config}/{split}: не завантажилось — {type(exc).__name__}: {exc}")
            continue
        ds = sorted({r["dataset"] for r in rows})
        print(f"=== {config}/{split}: {len(rows)} питань, {len(ds)} таблиць ===")
        cnt = {t: sum(1 for r in rows if r.get("type") == t) for t in TYPES}
        for t in TYPES:
            print(f"    {t:16s} {cnt[t]:5d}")
        miss = [r for r in rows if not r.get("answer")]
        print(f"    без відповіді: {len(miss)}")
        print(f"    із sample_answer: "
              f"{sum(1 for r in rows if r.get('sample_answer') not in (None, ''))}")
        print(f"    таблиці: {', '.join(ds[:6])}{' ...' if len(ds) > 6 else ''}")
        print()


if __name__ == "__main__":
    main()
