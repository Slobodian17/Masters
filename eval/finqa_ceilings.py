"""Step 0: what each way of raising accuracy could possibly buy, from cache.

POST-HOC ANALYSIS OF THE DEV SET. The 300 questions here were used by the
confirmatory run (5.26); every number below is an estimate of these 300, not
of the method. Nothing here is a confirmed result, and the candidate chosen on
the strength of it has to be pre-registered and re-measured on the held-out
295 (eval.finqa_heldout) before it means anything.

Zero API calls: everything comes from the two caches.

The question is where the errors at FULL coverage live, because that is the
new target -- not accuracy among accepted, which selective abstention already
raised, but accuracy over all 300.

Run:  python -m eval.finqa_ceilings
"""
import argparse
import collections
import os
import re

from eval import cache_io
from eval.finqa_confirm import build, cluster_of
from eval.finqa_run import is_correct, scale_error
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")
TOL = 0.01

_NUM = re.compile(r"-?\d+\.?\d*")


def close(a, b, tol=TOL):
    try:
        return abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1.0)
    except (TypeError, ValueError):
        return False


def table_numbers(rec):
    """Every number that appears in the question's own table."""
    out = set()
    if not rec:
        return out
    for row in (rec.get("table") or []):
        for cell in row:
            s = str(cell).replace(",", "").replace("$", "").replace("%", "")
            neg = s.strip().startswith("(") and s.strip().endswith(")")
            for m in _NUM.findall(s):
                try:
                    v = float(m)
                except ValueError:
                    continue
                out.add(-v if neg else v)
                out.add(v)
    return out


def gold_operands(prog):
    """Literal numbers in the gold program, e.g. subtract(68, 74) -> [68, 74]."""
    if not prog:
        return []
    out = []
    for m in _NUM.findall(str(prog)):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


OPS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b if b else None,
}


def classify(r, rec):
    """One label per wrong answer. First match wins; order is the argument."""
    v, gold = r["r2"], r["gold"]
    if v is None:
        return "NO_NUMBER"
    if gold is None:
        return "NO_GOLD"
    if scale_error(v, gold):
        return "SCALE"
    if close(v, -gold):
        return "SIGN"
    try:
        if gold and close(v, 1.0 / gold):
            return "RECIPROCAL"
    except ZeroDivisionError:
        pass

    ops = gold_operands(r.get("gold_program"))
    if len(ops) >= 2:
        a, b = ops[0], ops[1]
        for name, f in OPS.items():
            for x, y in ((a, b), (b, a)):
                try:
                    q = f(x, y)
                except Exception:
                    continue
                if q is not None and close(v, q):
                    return "WRONG_OP"

    nums = table_numbers(rec)
    if any(close(v, c) for c in nums):
        return "RAW_CELL"

    # the gold operation applied to some other pair of table numbers
    goldop = None
    for name in OPS:
        if name in str(r.get("gold_program") or ""):
            goldop = name
            break
    if goldop and len(nums) <= 400:
        f = OPS[goldop]
        ns = list(nums)
        for i, x in enumerate(ns):
            for y in ns:
                if x is y:
                    continue
                try:
                    q = f(x, y)
                except Exception:
                    continue
                if q is not None and close(v, q):
                    return "WRONG_CELLS"
    return "OTHER"


def bar(n, total, width=34):
    k = int(round(width * n / total)) if total else 0
    return "#" * k + "." * (width - k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    args = ap.parse_args()

    code, _ = cache_io.load_rows(os.path.join(CACHE_DIR, f"{args.tag}.jsonl"))
    dirr, _ = cache_io.load_rows(
        os.path.join(CACHE_DIR, f"baseline_table_{args.tag}.jsonl"))
    rows = build(code, dirr)
    by_id = {r["id"]: r for r in code}
    recs = {r["id"]: r for r in fq.load_split("test")}
    for r in rows:
        r["gold_program"] = by_id[r["id"]].get("gold_program")
        r["exception"] = any(s.get("exception_type")
                             for s in (by_id[r["id"]].get("steps") or []))

    n = len(rows)
    print("=" * 78)
    print("КРОК 0: СТЕЛІ СПОСОБІВ ПІДВИЩЕННЯ ТОЧНОСТІ")
    print("=" * 78)
    print("ПОСТ-ХОК АНАЛІЗ DEV-НАБОРУ. Ці 300 питань уже використані прогоном")
    print("5.26, тож кожне число нижче — оцінка ЦИХ 300, не методу. Обраний")
    print("кандидат передреєструється і міряється на held-out 295.")
    print()

    agree = [r for r in rows if r["verdict"] == "ACCEPT"]
    disagree = [r for r in rows if r["verdict"] == "ABSTAIN" and r["r2"] is not None]
    nonum = [r for r in rows if r["r2"] is None]
    base = sum(1 for r in rows if r["correct"])

    print(f"точність при повному покритті (віддаємо R2 завжди): "
          f"{base}/{n} = {base / n:.1%}")
    print(f"  згода {len(agree)}   розбіжність {len(disagree)}   "
          f"R2 без числа {len(nonum)}")
    print()

    # ---- (а) disagreements ----
    print("--- (а) РОЗБІЖНІСТЬ МАРШРУТІВ: хто правий ---")
    d_r1 = sum(1 for r in disagree if r["r1_correct"] and not r["correct"])
    d_r2 = sum(1 for r in disagree if r["correct"] and not r["r1_correct"])
    d_both = sum(1 for r in disagree if r["correct"] and r["r1_correct"])
    d_none = sum(1 for r in disagree if not r["correct"] and not r["r1_correct"])
    m = len(disagree)
    for label, k in (("правий лише R1 (пряма відповідь)", d_r1),
                     ("правий лише R2 (код)", d_r2),
                     ("обидва праві (різні, обидва в межах 1%)", d_both),
                     ("обидва хибні", d_none)):
        print(f"  {label:42s} {k:3d}  {k / m:6.1%}  {bar(k, m)}")
    fixable = d_r1
    print()
    print(f"  СТЕЛЯ розбору розбіжності: ідеальний арбітр вибирає правильну")
    print(f"  відповідь там, де хоч одна правильна -> +{fixable} питань")
    print(f"  точність стала б {(base + fixable)}/{n} = "
          f"{(base + fixable) / n:.1%}  (+{fixable / n * 100:.1f} п.п.)")
    print(f"  недосяжне навіть ідеальним арбітром: {d_none} питань, обидва хибні")

    # ---- (б) no number from R2 ----
    print()
    print("--- (б) R2 НЕ ДАВ ЧИСЛА ---")
    k = len(nonum)
    r1ok = sum(1 for r in nonum if r["r1_correct"])
    print(f"  таких питань: {k}   з них R1 правильний: {r1ok}  "
          f"({r1ok / k:.1%} якщо k>0)" if k else "  таких питань немає")
    if k:
        print(f"  СТЕЛЯ підстановки R1 замість порожнього R2: +{r1ok} питань")
        print(f"  точність стала б {(base + r1ok)}/{n} = {(base + r1ok) / n:.1%}"
              f"  (+{r1ok / n * 100:.1f} п.п.)")
        причини = collections.Counter(
            by_id[r["id"]].get("abstain_reason") for r in nonum)
        print(f"  причини: {dict(причини)}")

    # ---- (в) agreed but wrong ----
    print()
    print("--- (в) ОБИДВА ПОГОДИЛИСЬ НА ХИБНОМУ ---")
    agreed_wrong = [r for r in agree if not r["correct"]]
    print(f"  таких питань: {len(agreed_wrong)}")
    print(f"  за побудовою значення в обох маршрутів ОДНАКОВЕ (тому ACCEPT),")
    print(f"  тож питання не 'чи той самий рядок', а ЯКИЙ САМЕ тип помилки:")
    cnt = collections.Counter(classify(r, recs.get(r["id"])) for r in agreed_wrong)
    for lab, c in cnt.most_common():
        print(f"    {lab:14s} {c:3d}  {c / len(agreed_wrong):6.1%}  "
              f"{bar(c, len(agreed_wrong))}")
    raw = cnt.get("RAW_CELL", 0) + cnt.get("WRONG_CELLS", 0)
    print()
    print(f"  з них помилки ЧИТАННЯ (сира клітинка або інші клітинки): {raw}")
    print(f"  СТЕЛЯ будь-якого арбітра між двома маршрутами тут: 0 —")
    print(f"  обидва дали те саме хибне число, вибирати нема з чого.")

    # ---- (г) full taxonomy ----
    print()
    print("--- (г) ПОВНА ТАКСОНОМІЯ ПОМИЛОК ПРИ ПОВНОМУ ПОКРИТТІ ---")
    wrong = [r for r in rows if not r["correct"]]
    print(f"  хибних усього: {len(wrong)} з {n}")
    full = collections.Counter(classify(r, recs.get(r["id"])) for r in wrong)
    for lab, c in full.most_common():
        print(f"    {lab:14s} {c:3d}  {c / len(wrong):6.1%}  {bar(c, len(wrong))}")
    print()
    exc = sum(1 for r in wrong if r["exception"])
    print(f"  наскрізна ознака: у {exc} з {len(wrong)} хибних код кидав виняток")
    print(f"  розподіл по згоді: згода {len(agreed_wrong)}, "
          f"розбіжність {sum(1 for r in disagree if not r['correct'])}, "
          f"без числа {len(nonum)}")

    # ---- combined ceiling ----
    print()
    print("--- ЗВЕДЕНА СТЕЛЯ ---")
    print(f"  зараз (R2 завжди)                        {base / n:6.1%}")
    print(f"  + ідеальний арбітр на розбіжностях       "
          f"{(base + fixable) / n:6.1%}   (+{fixable / n * 100:.1f} п.п.)")
    print(f"  + R1 там, де R2 мовчить                  "
          f"{(base + fixable + r1ok) / n:6.1%}   (+{r1ok / n * 100:.1f} п.п.)")
    rest = n - (base + fixable + r1ok)
    print(f"  недосяжне вибором між двома маршрутами   {rest} питань "
          f"({rest / n:.1%}): {d_none} обидва хибні при розбіжності, "
          f"{len(agreed_wrong)} згода на хибному, "
          f"{len(nonum) - r1ok} мовчання без правильного R1")

    out = os.path.join(HERE, f"finqa_ceilings_{args.tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,gold,r1,r2,verdict,correct,r1_correct,error_type,exception\n")
        for r in rows:
            lab = "" if r["correct"] else classify(r, recs.get(r["id"]))
            f.write(f"{r['id']},{r['gold']},{r['r1']},{r['r2']},{r['verdict']},"
                    f"{int(r['correct'])},{int(r['r1_correct'])},{lab},"
                    f"{int(r['exception'])}\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
