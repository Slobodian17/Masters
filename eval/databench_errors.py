"""What is left wrong in arm B, and whether it is the data or the formatting.

Zero API calls. The split that matters is between an answer that is wrong
because the code read the wrong data and an answer that is wrong only because
of how it was written down: the first needs a better method, the second needs
a rule and costs nothing. On FinQA the residual was mostly the data (56% wrong
cells, 5.39); whether DataBench is the same is a question, not an assumption,
and the answer decides what the one improvement in the next step should be.

The separation is made by comparing twice: once with the organisers' own
comparison, and once with a deliberately lenient one that ignores case,
whitespace, duplicates, list length and precision beyond two decimals. An
answer the lenient comparison accepts and the official one rejects is a
formatting failure by definition -- the value is there, written wrongly.

Run:  python -m eval.databench_errors --arm B --tag dev100
"""
import argparse
import collections
import os
import re

from eval import cache_io
from ingestion import databench_loader as db

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "databench_cache")

STRIP = "[]'\" "
_NUM = re.compile(r"-?\d+\.?\d*")


def items(s):
    return [x.strip(STRIP) for x in str(s).strip("[]").split(",") if x.strip(STRIP)]


def nums(s):
    out = []
    for m in _NUM.findall(str(s)):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def lenient(value, truth, semantic):
    """Right value, possibly written wrongly. Intentionally generous."""
    v, t = str(value), str(truth)
    if semantic == "boolean":
        tv = {"true", "yes", "y", "1", "1.0"}
        fv = {"false", "no", "n", "0", "0.0"}
        a, b = v.strip(STRIP).lower(), t.strip(STRIP).lower()
        return (a in tv and b in tv) or (a in fv and b in fv)
    if semantic == "number":
        a, b = nums(v), nums(t)
        if not a or not b:
            return False
        return abs(a[0] - b[0]) <= 1e-4 * max(abs(b[0]), 1.0)
    if semantic == "category":
        return (re.sub(r"[^a-z0-9]", "", v.lower())
                == re.sub(r"[^a-z0-9]", "", t.lower()))
    if semantic == "list[category]":
        return ({re.sub(r"[^a-z0-9]", "", x.lower()) for x in items(v)}
                == {re.sub(r"[^a-z0-9]", "", x.lower()) for x in items(t)})
    if semantic == "list[number]":
        a = {round(x, 2) for x in nums(v)}
        b = {round(x, 2) for x in nums(t)}
        return a == b and bool(a)
    return False


def is_listy(s):
    return str(s).strip().startswith("[")


def classify(r):
    v, t, sem = r["value"], r["truth"], r["type"]
    if v is None:
        return "EXCEPTION", "код не повернув значення жодного разу"
    if lenient(v, t, sem):
        return "FORMAT", "значення правильне, запис не проходить порівняння"
    want_list = sem.startswith("list")
    if want_list != is_listy(v):
        return "SHAPE", ("повернув скаляр замість списку" if want_list
                         else "повернув список замість скаляра")
    if sem in ("number", "list[number]"):
        a, b = nums(v), nums(t)
        if a and b:
            if len(a) == 1 and len(b) == 1:
                for f in (10.0, 100.0, 1e3, 1e6):
                    if abs(a[0] / f - b[0]) <= 1e-4 * max(abs(b[0]), 1.0) or \
                       abs(a[0] * f - b[0]) <= 1e-4 * max(abs(b[0]), 1.0):
                        return "SCALE", f"масштаб x{f:g}"
                if abs(abs(a[0]) - abs(b[0])) <= 1e-4 * max(abs(b[0]), 1.0):
                    return "SIGN", "знак"
            if want_list and len(a) != len(b):
                over = set(round(x, 2) for x in a) & set(round(x, 2) for x in b)
                if over:
                    return "LIST_LEN", (f"довжина {len(a)} проти {len(b)}, "
                                        f"спільних {len(over)}")
    if want_list:
        sa = {x.lower() for x in items(v)}
        sb = {x.lower() for x in items(t)}
        if sa & sb:
            return "LIST_PART", f"спільних {len(sa & sb)} з {len(sb)}"
    return "DATA", "виконалось, тип і форма правильні, значення інше"


NAMES = {
    "EXCEPTION": "код упав на всіх спробах",
    "FORMAT": "формат запису (значення правильне)",
    "SHAPE": "не та форма: скаляр проти списку",
    "SCALE": "масштаб числа",
    "SIGN": "знак числа",
    "LIST_LEN": "список: ті самі елементи, інша довжина",
    "LIST_PART": "список: частковий перетин",
    "DATA": "НЕ ТІ ДАНІ (виконалось, значення інше)",
}
ORDER = ["EXCEPTION", "FORMAT", "SHAPE", "SCALE", "SIGN", "LIST_LEN",
         "LIST_PART", "DATA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B")
    ap.add_argument("--tag", default="dev100")
    ap.add_argument("--examples", type=int, default=10)
    args = ap.parse_args()

    path = os.path.join(CACHE, f"{args.arm}_{args.tag}.jsonl")
    rows = cache_io.load_rows(path)[0]
    wrong = [r for r in rows if not r["correct"]]
    print("=" * 76)
    print(f"РОЗБІР ПОМИЛОК АРМА {args.arm} — {args.tag}, нуль викликів API")
    print("=" * 76)
    ok = len(rows) - len(wrong)
    print(f"  питань {len(rows)}, правильних {ok} = {ok / len(rows):.1%}, "
          f"хибних {len(wrong)}")
    print()

    tagged = {r["id"]: classify(r) for r in wrong}
    cnt = collections.Counter(k for k, _ in tagged.values())
    print(f"  {'клас помилки':42s} {'к-сть':>6s} {'частка хибних':>14s}")
    for k in ORDER:
        n = cnt.get(k, 0)
        if n:
            print(f"  {NAMES[k]:42s} {n:6d} {n / len(wrong):13.0%}")
    fixable = sum(cnt.get(k, 0) for k in
                  ("FORMAT", "SHAPE", "SCALE", "SIGN", "LIST_LEN"))
    print()
    print(f"  {'лагодиться ПРАВИЛАМИ (формат/форма/масштаб)':42s} "
          f"{fixable:6d} {fixable / len(wrong):13.0%}")
    print(f"  {'потребує іншого МЕТОДУ (не ті дані)':42s} "
          f"{cnt.get('DATA', 0) + cnt.get('LIST_PART', 0):6d} "
          f"{(cnt.get('DATA', 0) + cnt.get('LIST_PART', 0)) / len(wrong):13.0%}")
    print(f"  стеля, якби всі правила спрацювали: "
          f"{ok + fixable}/{len(rows)} = {(ok + fixable) / len(rows):.1%}")

    print()
    print(f"  {'тип відповіді':16s} {'хибних':>7s}  розклад")
    for t in db.TYPES:
        sub = [r for r in wrong if r["type"] == t]
        if not sub:
            continue
        c = collections.Counter(tagged[r["id"]][0] for r in sub)
        txt = ", ".join(f"{k} {v}" for k, v in c.most_common())
        print(f"  {t:16s} {len(sub):7d}  {txt}")

    print()
    print("--- ПРИКЛАДИ ---")
    shown = collections.Counter()
    for r in wrong:
        k, why = tagged[r["id"]]
        if shown[k] >= 2 or sum(shown.values()) >= args.examples:
            continue
        shown[k] += 1
        print()
        print(f"  [{NAMES[k]}] {r['dataset']} / {r['type']}")
        print(f"    {str(r['question'])[:92]}")
        print(f"    наш   : {str(r['value'])[:80]}")
        print(f"    еталон: {str(r['truth'])[:80]}")
        print(f"    -> {why}")

    out = os.path.join(HERE, f"databench_errors_{args.arm}_{args.tag}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,dataset,type,bucket,why,ours,truth\n")
        for r in wrong:
            k, why = tagged[r["id"]]
            q = lambda s: '"' + str(s).replace('"', "'") + '"'
            f.write(f"{r['id']},{r['dataset']},{r['type']},{k},{q(why)},"
                    f"{q(r['value'])},{q(r['truth'])}\n")
    print()
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
