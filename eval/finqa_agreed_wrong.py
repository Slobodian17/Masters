"""The 53 questions where both routes agreed on the same wrong number.

These are the residue no arbitration can touch: the two routes returned the
same value, so there is nothing to choose between. Knowing what they are is
what says whether the ceiling is a method problem or a dataset problem.

Hand-checkable categories, each decided by a reproducible test rather than by
reading:

  DATASET     the gold program contradicts its own question (5.29.7 found 3
              such in a sample of 20 OTHER); counted here over all of them
  SIGN        |value| right, sign wrong
  PERCENT     out by exactly 100 -- the fraction/percent convention
  SCALE       out by another power of ten (thousands / millions)
  BASE        a ratio computed against the wrong denominator
  ROW         the value is a raw table cell, or the gold operation applied to
              other cells
  OTHER       none of the above

Zero API calls.

Run:  python -m eval.finqa_agreed_wrong
"""
import argparse
import collections
import os
import re

from eval import cache_io
from eval.finqa_confirm import build, decide
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")
TOL = 0.01
_NUM = re.compile(r"-?\d+\.?\d*")

# A question asking for a percentage or a change, whose gold program performs
# no division, cannot be answering what it asks. That is the reproducible test
# for a dataset defect; it does not rely on reading the question's intent.
_ASKS_RATIO = re.compile(
    r"percent|percentage|portion|proportion|ratio|share of|what part", re.I)
_ASKS_CHANGE = re.compile(r"percent(age)? change|percent(age)? increase|"
                          r"percent(age)? decrease|growth rate|roi|return on",
                          re.I)


def close(a, b, tol=TOL):
    try:
        return abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1.0)
    except (TypeError, ValueError):
        return False


def table_numbers(rec):
    out = set()
    for row in ((rec or {}).get("table") or []):
        for cell in row:
            s = str(cell).replace(",", "").replace("$", "").replace("%", "")
            neg = s.strip().startswith("(") and s.strip().endswith(")")
            for m in _NUM.findall(s):
                try:
                    v = float(m)
                except ValueError:
                    continue
                out.add(v)
                if neg:
                    out.add(-v)
    return out


def gold_ops_list(prog):
    return re.findall(r"(add|subtract|multiply|divide|exp|greater)\(", str(prog or ""))


def dataset_defect(question, prog):
    """Does the gold program fail to do what the question asks?"""
    ops = gold_ops_list(prog)
    q = str(question or "")
    if (_ASKS_RATIO.search(q) or _ASKS_CHANGE.search(q)) and "divide" not in ops:
        return True
    return False


def classify(r, rec):
    v, gold = r["r2"], r["gold"]
    if gold is None:
        return "NO_GOLD"
    if v is None:
        return "NO_NUMBER"
    if dataset_defect(r.get("question"), r.get("gold_program")):
        return "DATASET"
    if close(-v, gold):
        return "SIGN"
    for f in (100.0,):
        if close(v / f, gold) or close(v * f, gold):
            return "PERCENT"
    for f in (10.0, 1e3, 1e6, 1e9):
        if close(v / f, gold) or close(v * f, gold):
            return "SCALE"
    nums = table_numbers(rec)
    if any(close(v, c) for c in nums):
        return "ROW"
    # a ratio against the wrong denominator: v * d == gold * d' for some pair
    ops = gold_ops_list(r.get("gold_program"))
    if "divide" in ops and v not in (None, 0):
        num = [x for x in _NUM.findall(str(r.get("gold_program") or ""))]
        try:
            nums2 = [float(x) for x in num]
        except ValueError:
            nums2 = []
        for a in nums2:
            for c in nums:
                if c and close(a / c, v):
                    return "BASE"
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2-tag", default="confirm300")
    ap.add_argument("--r1-tag", default="baseline_table_confirm300")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    code, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.r2_tag}.jsonl"))
    dirr, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.r1_tag}.jsonl"))
    rows = build(code, dirr)
    by_id = {r["id"]: r for r in code}
    recs = {r["id"]: r for r in fq.load_split("test")}
    for r in rows:
        r["gold_program"] = by_id[r["id"]].get("gold_program")
        r["question"] = by_id[r["id"]].get("question")

    agreed_wrong = [r for r in rows
                    if r["verdict"] == "ACCEPT" and not r["correct"]]

    print("=" * 78)
    print(f"СПІЛЬНІ ХИБНІ ВІДПОВІДІ  (R2={args.r2_tag}, n={len(rows)})")
    print("=" * 78)
    print(f"обидва маршрути погодились на хибному: {len(agreed_wrong)}")
    print("Арбітраж тут дає рівно нуль: значення однакове, вибирати нема з чого.")
    print()

    lab = {r["id"]: classify(r, recs.get(r["id"])) for r in agreed_wrong}
    cnt = collections.Counter(lab.values())
    for k, v in cnt.most_common():
        bar = "#" * int(round(30 * v / len(agreed_wrong)))
        print(f"  {k:10s} {v:3d}  {v / len(agreed_wrong):6.1%}  {bar}")

    ds = [r for r in agreed_wrong if lab[r["id"]] == "DATASET"]
    print()
    print(f"--- ДЕФЕКТ ДАТАСЕТУ: {len(ds)} із {len(agreed_wrong)} ---")
    print("Тест відтворюваний: питання просить відсоток / відношення / зміну,")
    print("а gold-програма не містить divide. Не спирається на прочитання суті.")
    for r in ds[:12]:
        print(f"  {r['id'][:26]:26s} gold={r['gold']}  R1=R2={r['r2']}")
        print(f"     {str(r['question'])[:96]}")
        print(f"     gold_program={r['gold_program']}")

    # the same test over ALL wrong answers, not just the agreed ones
    all_wrong = [r for r in rows if not r["correct"]]
    ds_all = sum(1 for r in all_wrong
                 if dataset_defect(r.get("question"), r.get("gold_program")))
    nogold = sum(1 for r in rows if r["gold"] is None)
    print()
    print("--- СТЕЛЯ ДОСЯЖНОЇ ТОЧНОСТІ НА ЦІЙ ВИБІРЦІ ---")
    print(f"  хибних усього                       : {len(all_wrong)} з {len(rows)}")
    print(f"  з дефектом датасету (весь набір)    : {ds_all}")
    print(f"  без числового еталона (yes/no)      : {nogold}")
    print(f"  разом непереможних                  : {ds_all + nogold} "
          f"({(ds_all + nogold) / len(rows):.1%})")
    print(f"  МАКСИМАЛЬНА досяжна EA              : "
          f"{1 - (ds_all + nogold) / len(rows):.1%}")
    print()
    print("  Це НИЖНЯ межа кількості дефектів: тест ловить лише один їх вид")
    print("  (питання про відношення без divide у програмі), інші не ловить.")

    out = os.path.join(HERE, f"finqa_agreed_wrong_{args.r2_tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,gold,value,category,gold_program,question\n")
        for r in agreed_wrong:
            q = str(r["question"]).replace(",", " ")
            gp = str(r["gold_program"]).replace(",", ";")
            f.write(f"{r['id']},{r['gold']},{r['r2']},{lab[r['id']]},{gp},{q}\n")
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
