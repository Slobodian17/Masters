"""Reserve the held-out set, and refuse to be the thing that touches it.

After the confirmatory run (5.26) the 300 questions it used become a dev set:
every ceiling, taxonomy and candidate improvement from here on is measured on
them, and anything measured on the set a hypothesis was formed on is an
estimate of that set, not of the method.

What is left untouched is 595 - 300 = 295 questions. Reserving them now, as
an explicit list written before any candidate exists, is what makes the next
confirmatory run confirmatory. Reserving them later -- after seeing which
questions a candidate helps on -- would not.

This module writes the list and checks it. It deliberately provides no way to
load the questions' answers.

Run:  python -m eval.finqa_heldout          # write and verify
      python -m eval.finqa_heldout --check  # verify only
"""
import argparse
import hashlib
import json
import os

import eval.finqa_run as R
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "finqa_heldout295.json")

EXPLORATORY = ["ab_yes_v2", "ab_no_v2", "m_qwen27", "m_gptoss20", "m_allam7",
               "corpus50_corpus", "diag", "diag_fixed", "pilot", "pilot2"]


def is_numeric(rec):
    """Does this question have a NUMERIC gold answer?

    Defect №18: seven of the 300 dev questions answer "yes"/"no". A numeric
    pipeline cannot answer them by construction, yet they were scored as
    errors -- 2.3% of the sample counted as unavoidable failures.

    Only the TYPE of the gold answer is read here, never its value, and it is
    read before any prediction on the held-out set exists. This is a property
    of the corpus, like table size, not a result.
    """
    return fq.gold_answer(rec) is not None


def partition():
    recs = fq.load_split("test")
    subset = R.table_only(recs)
    seen = set()
    for t in EXPLORATORY:
        try:
            seen |= {r["id"] for r in R.load_cache_ids(t)}
        except FileNotFoundError:
            pass
    untouched = [r for r in subset if r["id"] not in seen]
    dev = {r["id"] for r in R.stratified_sample(untouched, 300, 42)}
    held_all = [r for r in untouched if r["id"] not in dev]
    held_num = [r for r in held_all if is_numeric(r)]
    return {"table_only": len(subset), "exploratory": len(seen),
            "untouched": len(untouched), "dev": sorted(dev),
            "held_out": sorted(r["id"] for r in held_all),
            "held_out_numeric": sorted(r["id"] for r in held_num),
            "dropped_nonnumeric": sorted(r["id"] for r in held_all
                                         if not is_numeric(r))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    p = partition()
    digest = hashlib.md5("\n".join(p["held_out"]).encode()).hexdigest()
    digest_num = hashlib.md5("\n".join(p["held_out_numeric"]).encode()).hexdigest()

    print("=" * 78)
    print("РОЗДІЛЕННЯ ТАБЛИЧНОЇ ПІДМНОЖИНИ FinQA test")
    print("=" * 78)
    print(f"  еталон у таблиці           : {p['table_only']}")
    print(f"  торкалась розвідка         : {p['exploratory']}")
    print(f"  недоторканих               : {p['untouched']}")
    print(f"  DEV (прогін 5.26, використано): {len(p['dev'])}")
    print(f"  HELD-OUT (зарезервовано)   : {len(p['held_out'])}")
    print(f"  перетин dev/held-out       : {len(set(p['dev']) & set(p['held_out']))}")
    print(f"  md5 списку held-out        : {digest}")
    print()
    print("  ЗМІНА СУКУПНОСТІ (дефект №18), зафіксована ДО прогону:")
    print(f"    без числового еталона (yes/no) : {len(p['dropped_nonnumeric'])}")
    print(f"    HELD-OUT ЧИСЛОВИЙ              : {len(p['held_out_numeric'])}")
    print(f"    md5 числового списку           : {digest_num}")
    print("    прочитано лише ТИП еталона, не значення; предиктів ще не існує")

    if args.check:
        if not os.path.exists(OUT):
            print("  ФАЙЛ ВІДСУТНІЙ")
            return 1
        prev = json.load(open(OUT, encoding="utf-8"))
        same = prev["md5"] == digest
        print(f"  збіг із записаним          : {'ТАК' if same else 'НІ — ЗМІНИВСЯ'}")
        return 0 if same else 1

    if os.path.exists(OUT):
        prev = json.load(open(OUT, encoding="utf-8"))
        if prev["md5"] != digest:
            print("  УВАГА: список відрізняється від записаного раніше.")
            print("  Це означає зміну вибірки — НЕ перезаписую.")
            return 1
        print("  вже записано, збігається")
        return 0

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Зарезервовано 2026-09-26, ДО появи будь-якого кандидата "
                     "на покращення. Не дивитись і не використовувати, доки "
                     "кандидат не обраний і не передреєстрований на dev."),
            "md5": digest,
            "n": len(p["held_out"]),
            "ids": p["held_out"],
            "md5_numeric": digest_num,
            "n_numeric": len(p["held_out_numeric"]),
            "ids_numeric": p["held_out_numeric"],
            "dropped_nonnumeric": p["dropped_nonnumeric"],
            "note_numeric": ("Дефект №18: питання з еталоном yes/no числовий "
                             "конвеєр не може відповісти за побудовою. Фільтр "
                             "зафіксовано 2026-09-26, ДО прогону на held-out; "
                             "прочитано лише тип еталона, не значення."),
        }, f, ensure_ascii=False, indent=1)
    print(f"  записано: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
