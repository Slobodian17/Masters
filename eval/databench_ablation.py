"""Which of our blocks earns its place on DataBench, and against which gold.

Zero API calls: every arm is reconstructed by re-executing cached code against
the local tables, so the ablation costs nothing and can be repeated.

Two blocks are ablated:
  crash repair     the final iteration vs the FIRST attempt only
  normalisation    `normalise()` vs plain `str()` on whatever the code returned

Two golds, because the official one is demonstrably wrong on 8 of 100
questions (5.41):
  official   the organisers' answers, as published and as the leaderboard uses
  audited    the 4 defects whose correct value follows from the question alone
             are corrected; the other 4 are EXCLUDED, not corrected, because
             there the correct value would be our own output and we would be
             grading ourselves by our own reading of the question.

Run:  python -m eval.databench_ablation --tag dev100
"""
import argparse
import os

from core import sandbox
from eval import cache_io
from eval.databench_run import normalise
from ingestion import databench_loader as db

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "databench_cache")

# Corrected because the right answer follows from the question and the table,
# with no choice of column or reading left to us.
CORRECTED = {
    "019_Aircraft|How many incidents resulted in non-zero fatalities?":
        ("9231", "(Fatalities > 0).sum() = 9231 із 23519; стовпець uint16"),
    "028_Predict|Does anyone have a Diabetes Pedigree Function score above 2.5?":
        ("False", "max(DiabetesPedigreeFunction) = 2.42, отже понад 2.5 немає"),
    "021_Telco|Do all customers have phone service?":
        ("False", "PhoneService: 6361 'Yes', 682 'No'"),
    "040_Speed|Are all participants who expected more than 5 matches male?":
        ("False", "454 учасники з expected>5 мають gender != 'male'"),
}

# Defective, but the correct value would coincide with our own answer and
# depends on which column we chose, so these are dropped from the denominator.
EXCLUDED = {
    "047_Bank|Who are the six customer ids with the highest total revolving balance?":
        "еталон [2517, 2517, 2517, 2517, 2517, 2517] — шість однакових id",
    "043_Predict|What are the 3 highest disbursement gross for companies with a franchise code of 1?":
        "еталон [50000, 100000, 10000] — не найбільші й не впорядковані",
    "005_NYC|Name the 2 least frequent descriptors for complaints by inverse alphabetical order.":
        "питання просить 2 елементи, еталон містить 3",
    "025_Data|What are the lowest 4 rankings associated with the keyword 'no code data science'?":
        "еталон [10, 9, 8, 7]; у цього keyword рейтинги лише 1..8",
}


def key(r):
    return f"{r['dataset']}|{r['question']}"


def code_at(r, first):
    steps = r.get("steps") or []
    if not steps:
        return None
    if first:
        return steps[0].get("code")
    return next((s.get("code") for s in reversed(steps) if s.get("code")), None)


def value_of(r, df, first, norm):
    code = code_at(r, first)
    if not code:
        return None
    res, exc = sandbox.execute(code, {"df": df})
    if exc is not None or res is None:
        return None
    return normalise(res) if norm else str(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev100")
    args = ap.parse_args()

    rows = cache_io.load_rows(os.path.join(CACHE, f"B_{args.tag}.jsonl"))[0]
    print("=" * 76)
    print(f"АБЛЯЦІЯ АРМА B — DataBench {args.tag}, нуль викликів API")
    print("=" * 76)

    # re-execute once per (first, norm) combination
    variants = {}
    repro_ok = repro_bad = 0
    for r in rows:
        df = db.table(r["dataset"])
        for first in (True, False):
            for norm in (True, False):
                variants.setdefault((first, norm), {})[r["id"]] = \
                    value_of(r, df, first, norm)
        # honesty gate: re-running the final code must give what was cached
        v = variants[(False, True)][r["id"]]
        if (v is None) == (r["value"] is None) and (
                v is None or str(v) == str(r["value"])):
            repro_ok += 1
        else:
            repro_bad += 1

    print()
    print(f"--- ВОРОТА ЧЕСНОСТІ: перезапуск кешованого коду ---")
    print(f"  відтворилось {repro_ok}/{len(rows)}, розійшлось {repro_bad}")
    if repro_bad > len(rows) * 0.05:
        print("  !! відтворюваність низька — абляція ненадійна")

    corr = {key(r): r for r in rows if key(r) in CORRECTED}
    excl = {key(r): r for r in rows if key(r) in EXCLUDED}
    print(f"  виправлених еталонів знайдено {len(corr)}/4, "
          f"виключених {len(excl)}/4")

    def score(vals, gold):
        ok = n = 0
        for r in rows:
            k = key(r)
            if gold == "audited" and k in EXCLUDED:
                continue
            truth = (CORRECTED[k][0] if (gold == "audited" and k in CORRECTED)
                     else r["truth"])
            n += 1
            v = vals.get(r["id"])
            if v is None:
                continue
            try:
                ok += bool(db.compare(v, truth, r["type"]))
            except Exception:
                pass
        return ok, n

    print()
    print("--- АБЛЯЦІЯ ---")
    print(f"  {'конфігурація':46s} {'офіційний еталон':>18s} "
          f"{'перевірений':>14s}")
    names = {(True, False): "перша спроба, без нормалізації",
             (True, True): "перша спроба, з нормалізацією",
             (False, False): "з ремонтом падінь, без нормалізації",
             (False, True): "З РЕМОНТОМ І НОРМАЛІЗАЦІЄЮ (повна)"}
    res = {}
    for k in ((True, False), (True, True), (False, False), (False, True)):
        o, no = score(variants[k], "official")
        a, na = score(variants[k], "audited")
        res[k] = (o, no, a, na)
        print(f"  {names[k]:46s} {f'{o}/{no} = {o / no:.0%}':>18s} "
              f"{f'{a}/{na} = {a / na:.0%}':>14s}")

    print()
    print("--- ВНЕСОК КОЖНОГО БЛОКУ (офіційний еталон) ---")
    full = res[(False, True)][0]
    print(f"  ремонт падінь   : {full - res[(True, True)][0]:+d} питань")
    print(f"  нормалізація    : {full - res[(False, False)][0]:+d} питань")
    nrep = sum(1 for r in rows if len(r.get("steps") or []) > 1)
    print(f"  (ремонт узагалі запускався на {nrep} питаннях зі {len(rows)})")

    print()
    print("=" * 76)
    print("ХИБНІ ЕТАЛОНИ: 8 випадків із доказом")
    print("=" * 76)
    print()
    print("ВИПРАВЛЕНО (правильне значення випливає з питання й таблиці):")
    for k, (val, why) in CORRECTED.items():
        r = corr.get(k)
        if not r:
            continue
        ds, q = k.split("|", 1)
        print()
        print(f"  {ds} / {r['type']}")
        print(f"    питання      : {q[:88]}")
        print(f"    еталон       : {r['truth'][:60]}")
        print(f"    правильно    : {val}")
        print(f"    наша відповідь: {str(r['value'])[:60]}")
        print(f"    доказ        : {why}")
    print()
    print("ВИКЛЮЧЕНО (еталон дефектний, але виправлення збіглося б із нашою")
    print("відповіддю й залежить від обраного нами стовпця):")
    for k, why in EXCLUDED.items():
        r = excl.get(k)
        if not r:
            continue
        ds, q = k.split("|", 1)
        print()
        print(f"  {ds} / {r['type']}")
        print(f"    питання : {q[:88]}")
        print(f"    еталон  : {r['truth'][:60]}")
        print(f"    наша    : {str(r['value'])[:60]}")
        print(f"    дефект  : {why}")

    out = os.path.join(HERE, f"databench_ablation_{args.tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("config,gold,correct,n,accuracy\n")
        for k in res:
            o, no, a, na = res[k]
            cfg = names[k].replace(",", ";")
            f.write(f"\"{cfg}\",official,{o},{no},{o / no:.4f}\n")
            f.write(f"\"{cfg}\",audited,{a},{na},{a / na:.4f}\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
