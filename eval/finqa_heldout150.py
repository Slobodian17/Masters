"""The 150 held-out questions the confirmatory run will use. Zero API calls.

Written and hashed BEFORE the run, so "the same 150" means the same 150 and
cannot be reshaped by a result.

Why 150 and not the full 295: the Groq daily bucket holds 200k tokens and
refills at 8,333/hour, and both routes together cost a measured 2,253 tokens
per question (5.42). 295 questions need 665k tokens -- 3.3 days. 150 fits what
is available before the deadline with room for failures.

STRATIFIED on `n_ops`, the number of operations in the gold program, to match
confirm300's mix (1: 61%, 2: 33%, 3: 2.3%, 4: 1%, 5: 2.7%). A simple random
draw would shift that mix, and since accuracy falls with n_ops, a shifted mix
would move the headline numbers for a reason that has nothing to do with the
method.

Run:  python -m eval.finqa_heldout150
"""
import argparse
import collections
import hashlib
import json
import os
import random

from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
HELDOUT = os.path.join(HERE, "finqa_heldout295.json")
OUT = os.path.join(HERE, "finqa_heldout150.json")
IDS = os.path.join(HERE, "finqa_heldout150.ids")

SEED = 20261101
N = 150
# confirm300's own mix, counted from its cache, not guessed.
TARGET = {1: 183, 2: 99, 3: 7, 4: 3, 5: 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N)
    args = ap.parse_args()

    with open(HELDOUT, encoding="utf-8") as f:
        held = json.load(f)
    pool = list(held["ids"])
    print(f"held-out: {len(pool)} id, md5 у файлі {held['md5'][:12]}…")

    recs = {r["id"]: r for r in fq.load_split("test")}
    missing = [q for q in pool if q not in recs]
    if missing:
        print(f"  УВАГА: {len(missing)} id немає у split test")

    def nops(q):
        return (len(fq.program_ops(fq.gold_program(recs[q])))
                if q in recs else 0)

    by = collections.defaultdict(list)
    for q in pool:
        by[min(nops(q), 5)].append(q)
    print("  розподіл n_ops у held-out:",
          {k: len(v) for k, v in sorted(by.items())})

    total = sum(TARGET.values())
    want = {k: round(args.n * v / total) for k, v in TARGET.items()}
    # rounding can miss the target by one or two
    while sum(want.values()) > args.n:
        want[max(want, key=lambda k: want[k])] -= 1
    while sum(want.values()) < args.n:
        want[1] += 1
    print("  цільова квота:", want)

    rnd = random.Random(SEED)
    sel, short = [], {}
    for k in sorted(want):
        avail = sorted(by.get(k, []))
        rnd.shuffle(avail)
        take = avail[:want[k]]
        if len(take) < want[k]:
            short[k] = want[k] - len(take)
        sel.extend(take)
    # Any shortfall in a thin stratum is refilled from n_ops=1, the largest,
    # and the refill is reported rather than absorbed silently.
    if len(sel) < args.n:
        rest = [q for q in sorted(by.get(1, [])) + sorted(by.get(2, []))
                if q not in set(sel)]
        rnd.shuffle(rest)
        sel.extend(rest[:args.n - len(sel)])
    if short:
        print(f"  страти не вистачило: {short} — добрано з n_ops 1-2")

    sel = sorted(set(sel))
    got = collections.Counter(min(nops(q), 5) for q in sel)
    print(f"  відібрано {len(sel)}; фактичний n_ops: {dict(sorted(got.items()))}")
    nonnum = [q for q in sel if fq.gold_answer(recs[q]) is None]
    print(f"  нечисловий еталон: {len(nonnum)} (у confirm300 було 7/300)")

    md5 = hashlib.md5("\n".join(sel).encode()).hexdigest()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"n": len(sel), "seed": SEED, "md5": md5, "ids": sel,
                   "n_ops": {str(k): v for k, v in sorted(got.items())},
                   "note": ("Передреєстровано 01.11.2026 ДО прогону. "
                            "Стратифіковано на n_ops за складом confirm300. "
                            "Обидва маршрути відповідають на ці самі id.")},
                  f, ensure_ascii=False, indent=1)
    with open(IDS, "w", encoding="utf-8") as f:
        f.write("\n".join(sel) + "\n")
    print()
    print(f"md5 списку: {md5}")
    print("Записано:", OUT, "та", IDS)


if __name__ == "__main__":
    main()
