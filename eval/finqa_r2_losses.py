"""Where the code route loses an answer the model already knows.

43 questions on dev have the direct answer right and the code wrong; only 4 go
the other way. So the model is not ignorant of those numbers -- something
between the question and the returned value drops them. This module locates
that something, from cache, with zero API calls.

POST-HOC DIAGNOSIS OF THE DEV SET. Not a result; input to choosing a candidate.

Deliberately uses nothing from R1 except the labels "R1 right, R2 wrong" to
pick the 43. No R1 value is fed to any code-route mechanism: the routes have
to stay independent or the agreement check stops working (5.18, 5.29.1).

Run:  python -m eval.finqa_r2_losses
"""
import argparse
import collections
import os
import re

from eval import cache_io
from eval.finqa_ceilings import classify
from eval.finqa_confirm import build
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")

_LIT = re.compile(r"""['"]([^'"]{2,80})['"]""")


def literals(code):
    return set(_LIT.findall(code or ""))


def l0_substitutions(steps):
    """Label swaps the deterministic repair made, as (before, after) pairs."""
    out = []
    for a, b in zip(steps, steps[1:]):
        if a.get("level") != "L0":
            continue
        before, after = literals(a.get("code")), literals(b.get("code"))
        gone, new = before - after, after - before
        if gone or new:
            out.append((sorted(gone), sorted(new)))
    return out


def stage(r, crow):
    """The coarsest useful answer: at which stage did the answer get lost?"""
    steps = crow.get("steps") or []
    if crow.get("abstain_reason") == "OUT_OF_SCOPE":
        return "ГЕЙТ відмовив (код не запускався)"
    if any(s.get("level") == "L0" for s in steps):
        return "L0 детерміністичний фікс"
    if any(s.get("exception_type") for s in steps):
        return "виняток у sandbox"
    if r["r2"] is None:
        return "немає числа з інших причин"
    return "код виконався, значення хибне"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    code, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.tag}.jsonl"))
    dirr, _ = cache_io.load_rows(
        os.path.join(CACHE_DIR, f"baseline_table_{args.tag}.jsonl"))
    rows = build(code, dirr)
    by_id = {r["id"]: r for r in code}
    recs = {r["id"]: r for r in fq.load_split("test")}
    for r in rows:
        r["gold_program"] = by_id[r["id"]].get("gold_program")
        r["question"] = by_id[r["id"]].get("question")

    lost = [r for r in rows if r["r1_correct"] and not r["correct"]]
    gained = [r for r in rows if r["correct"] and not r["r1_correct"]]

    print("=" * 78)
    print("ДЕ КОДОВИЙ МАРШРУТ ГУБИТЬ ВІДПОВІДЬ (пост-хок діагностика dev)")
    print("=" * 78)
    print(f"пряма правильна, код хибний : {len(lost)}")
    print(f"навпаки                     : {len(gained)}")
    print()

    print("--- СТАДІЯ, НА ЯКІЙ ВІДПОВІДЬ ЗНИКЛА (43 питання) ---")
    st = collections.Counter(stage(r, by_id[r["id"]]) for r in lost)
    for k, v in st.most_common():
        print(f"  {k:42s} {v:3d}  {v / len(lost):6.1%}")

    print()
    print("--- ТИП ПОМИЛКИ СЕРЕД ТИХ, ДЕ КОД ВИКОНАВСЯ ---")
    ran = [r for r in lost if r["r2"] is not None]
    lab = collections.Counter(classify(r, recs.get(r["id"])) for r in ran)
    for k, v in lab.most_common():
        print(f"  {k:16s} {v:3d}  {v / max(len(ran), 1):6.1%}")

    # ---- (б) L0 across ALL wrong answers ----
    print()
    print("--- (б) L0-ФІКС НА ВСІХ 300 ---")
    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]
    def l0n(rs):
        return sum(1 for r in rs
                   if any(s.get("level") == "L0"
                          for s in (by_id[r["id"]].get("steps") or [])))
    print(f"  хибних відповідей з L0-фіксом      : {l0n(wrong)} з {len(wrong)}")
    print(f"  правильних відповідей з L0-фіксом  : {l0n(right)} з {len(right)}")
    subs = []
    for r in rows:
        for gone, new in l0_substitutions(by_id[r["id"]].get("steps") or []):
            subs.append((r["id"], r["correct"], gone, new))
    print(f"  усього підстановок мітки            : {len(subs)}")
    for qid, ok, gone, new in subs[:10]:
        print(f"    {qid[:26]:26s} {'OK ' if ok else 'ХИБ'} "
              f"{str(gone)[:40]} -> {str(new)[:40]}")
    if not subs:
        print("    підстановок немає: L0 у цьому прогоні майже не працював")

    # ---- (а) what the generator sees ----
    print()
    print("--- (а) ЩО БАЧИТЬ ГЕНЕРАТОР КОДУ ---")
    from core import pipeline
    from eval.finqa_baseline import build_prompt as r1_prompt
    from eval.finqa_run import FINQA_INSTRUCTIONS
    lens2, lens1, has_values = [], [], 0
    for r in rows[:120]:
        rec = recs.get(r["id"])
        if not rec:
            continue
        tabs = fq.to_tables(rec)
        p2 = pipeline._build_prompt(r["question"], tabs,
                                    instructions=FINQA_INSTRUCTIONS)
        df = tabs.get("table")
        if df is None or df.empty:
            continue
        p1 = r1_prompt(r["question"], df, "table")
        lens2.append(len(p2))
        lens1.append(len(p1))
        # does the code prompt contain any interior cell value?
        vals = [str(v) for v in df.iloc[:, 1:].values.ravel()[:20]
                if str(v) not in ("", "nan")]
        if any(v in p2 for v in vals[:8]):
            has_values += 1
    print(f"  промпт R2: лише назви стовпців і мітки рядків "
          f"(`_schema_description`), БЕЗ значень")
    print(f"  промпт R1: уся таблиця зі значеннями")
    print(f"  промптів R2, де знайдено бодай одне значення таблиці: "
          f"{has_values} з {len(lens2)}")
    print(f"  середня довжина: R2 {sum(lens2) / len(lens2):.0f} символів, "
          f"R1 {sum(lens1) / len(lens1):.0f}  "
          f"(додати значення -> {(sum(lens1) - sum(lens2)) / sum(lens2) * 100:+.1f}% "
          f"до вхідних токенів коду)")
    print("  ОТЖЕ: маршрути бачили РІЗНУ інформацію, і код — меншу.")

    print()
    print(f"--- ПРИКЛАДИ (перші {args.show} з 43) ---")
    for r in lost[:args.show]:
        crow = by_id[r["id"]]
        steps = crow.get("steps") or []
        c = next((s.get("code") for s in reversed(steps) if s.get("code")), "")
        print()
        print(f"  {r['id']}  {stage(r, crow)}")
        print(f"    питання: {str(r['question'])[:100]}")
        print(f"    еталон {r['gold']}   R1={r['r1']} ✓   R2={r['r2']} ✗   "
              f"gold_program={r['gold_program']}")
        if c:
            print(f"    код: {str(c)[:190].replace(chr(10), ' | ')}")

    out = os.path.join(HERE, f"finqa_r2_losses_{args.tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,gold,r1,r2,stage,error_type,gold_program,question\n")
        for r in lost:
            q = str(r["question"]).replace(",", " ")
            gp = str(r["gold_program"]).replace(",", ";")
            f.write(f"{r['id']},{r['gold']},{r['r1']},{r['r2']},"
                    f"{stage(r, by_id[r['id']])},"
                    f"{classify(r, recs.get(r['id']))},{gp},{q}\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
