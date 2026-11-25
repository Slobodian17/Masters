"""The fixed 100-question dev subset every candidate is measured on.

Written once, before the first candidate runs, so that "the same 100" means
the same 100 and not whatever a later seed produces.

STRATIFIED, not a simple random draw, and for a reason worth stating. The
caption's largest known effect is on the 23 questions the scope gate refused
(5.29.8), which are 7.7% of the dev set -- a random 100 would contain about
eight of them and spend the other 92 calls where nothing changes. So:

  stratum G  all 23 gate-refused questions: a CENSUS, no sampling error
  stratum N  a seeded random 77 of the remaining 277

An effect on the whole dev set is then recovered as a weighted sum,
(23/300)*delta_G + (277/300)*delta_N, and the only sampling variance is in
stratum N. This is strictly more powerful than a simple random 100 at the same
cost, and it is decided BEFORE any candidate is run so it cannot be tuned to a
result.

Power, stated in advance so it is not discovered afterwards: n = 100 paired is
too small to establish a +4-5 pp change. McNemar with roughly 8 discordant
pairs against 4 gives p ~ 0.4. The dev measurement is for CHOOSING a
candidate; establishing it is the job of the held-out 291.

Run:  python -m eval.finqa_subset
"""
import argparse
import collections
import hashlib
import json
import os
import random

from eval import cache_io

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "finqa_dev100.json")
IDS = os.path.join(HERE, "finqa_dev100.ids")
N_OTHER = 77
SEED = 42


def build():
    code, _ = cache_io.load_rows(
        os.path.join(HERE, "finqa_cache", "confirm300.jsonl"))
    gate = sorted(r["id"] for r in code
                  if r.get("abstain_reason") == "OUT_OF_SCOPE")
    rest = [r for r in code if r["id"] not in set(gate)]

    # stratify the remainder by gold program length, as the parent sample was
    by_len = collections.defaultdict(list)
    for r in rest:
        by_len[min(r.get("n_ops") or 0, 3)].append(r["id"])
    rng = random.Random(SEED)
    for v in by_len.values():
        v.sort()
        rng.shuffle(v)
    total = sum(len(v) for v in by_len.values())
    pick = []
    for k in sorted(by_len):
        share = round(N_OTHER * len(by_len[k]) / total)
        pick.extend(by_len[k][:share])
    pool = [i for k in sorted(by_len) for i in by_len[k][len(pick):]]
    while len(pick) < N_OTHER and pool:
        pick.append(pool.pop())
    pick = sorted(pick[:N_OTHER])

    return {"gate_census": gate, "random_stratum": pick,
            "ids": sorted(set(gate) | set(pick)),
            "n_dev": len(code), "n_gate": len(gate), "n_rest": len(rest)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    p = build()
    digest = hashlib.md5("\n".join(p["ids"]).encode()).hexdigest()

    print("=" * 70)
    print("ФІКСОВАНА ПІДВИБІРКА DEV ДЛЯ ЗАМІРІВ КАНДИДАТІВ")
    print("=" * 70)
    print(f"  dev усього                    : {p['n_dev']}")
    print(f"  страта G (відмови гейта, ценз) : {p['n_gate']}")
    print(f"  страта N (випадкові з решти)   : {len(p['random_stratum'])} "
          f"з {p['n_rest']}")
    print(f"  РАЗОМ                          : {len(p['ids'])}")
    print(f"  вага G у переносі на 300        : {p['n_gate'] / p['n_dev']:.3f}")
    print(f"  вага N                         : {p['n_rest'] / p['n_dev']:.3f}")
    print(f"  md5 списку                     : {digest}")

    if args.check:
        if not os.path.exists(OUT):
            print("  ФАЙЛ ВІДСУТНІЙ")
            return 1
        prev = json.load(open(OUT, encoding="utf-8"))
        same = prev["md5"] == digest
        print(f"  збіг із записаним              : {'ТАК' if same else 'НІ'}")
        return 0 if same else 1

    if os.path.exists(OUT):
        prev = json.load(open(OUT, encoding="utf-8"))
        if prev["md5"] != digest:
            print("  УВАГА: список відрізняється від записаного — НЕ перезаписую")
            return 1
        print("  вже записано, збігається")
        return 0

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"note": ("Зафіксовано 2026-09-27, до першого заміру "
                            "кандидата. Стратифікація: ценз відмов гейта плюс "
                            "випадкові 77 з решти. Усі кандидати міряються на "
                            "цих самих id, парно з поточними R1 і R2."),
                   "md5": digest, "seed": SEED, **p}, f,
                  ensure_ascii=False, indent=1)
    with open(IDS, "w", encoding="utf-8") as f:
        f.write("\n".join(p["ids"]) + "\n")
    print(f"  записано: {OUT}")
    print(f"            {IDS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
