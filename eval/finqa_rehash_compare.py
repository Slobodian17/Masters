"""Compare a direct-route cache against a rerun of the same questions.

The first 12 rows of the confirmatory direct route were produced before the
prompt template was hashed, so there is no way to prove they were asked under
the same prompt as the remaining 288. Asking them again under the recorded
hash settles it by measurement instead of by argument.

Nothing here touches gold. The comparison is answer against answer, because
the question is whether the apparatus changed, not whether the model is
right -- and the run is blind, so correctness must not appear in the output.

Run:  python -m eval.finqa_rehash_compare --tag confirm300 --suffix _rehash
"""
import argparse
import os

from eval import cache_io

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "finqa_cache")

TOL = 0.01


def same(a, b, tol=TOL):
    if a is None and b is None:
        return "обидва без числа"
    if a is None or b is None:
        return "одне без числа"
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return "не число"
    if x == y:
        return "ідентично"
    if abs(x - y) <= tol * max(abs(y), 1.0):
        return "у межах 1%"
    for f in (1e2, 1e3, 1e6, 1e9):
        if abs(x * f - y) <= tol * max(abs(y), 1.0) or \
           abs(x / f - y) <= tol * max(abs(y), 1.0):
            return "різниця рівно 10^k"
    return "різне"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="confirm300")
    ap.add_argument("--mode", default="table")
    ap.add_argument("--suffix", default="_rehash")
    args = ap.parse_args()

    base = os.path.join(CACHE_DIR, f"baseline_{args.mode}_{args.tag}.jsonl")
    arch = base.replace(".jsonl", "") + ".archive_prehash.jsonl"
    new = os.path.join(CACHE_DIR,
                       f"baseline_{args.mode}_{args.tag}{args.suffix}.jsonl")

    old_rows, _ = cache_io.load_rows(arch if os.path.exists(arch) else base)
    new_rows, _ = cache_io.load_rows(new)
    old = {r["id"]: r for r in old_rows}
    nw = {r["id"]: r for r in new_rows}
    both = [i for i in old if i in nw]

    print("=" * 78)
    print("ПОШТУЧНЕ ПОРІВНЯННЯ: старі рядки проти перегенерованих")
    print("=" * 78)
    print(f"старих: {len(old)}   нових: {len(nw)}   спільних: {len(both)}")
    print("Еталон НЕ використовується: порівнюється відповідь з відповіддю.")
    print()

    counts = {}
    print(f"{'id':38s} {'стара':>14s} {'нова':>14s}  вердикт")
    print("-" * 86)
    for i in sorted(both):
        a, b = old[i].get("answer"), nw[i].get("answer")
        v = same(a, b)
        counts[v] = counts.get(v, 0) + 1
        print(f"{i[:38]:38s} {str(a):>14s} {str(b):>14s}  {v}")

    print()
    print("--- підсумок ---")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    ident = counts.get("ідентично", 0) + counts.get("у межах 1%", 0)
    print()
    if ident == len(both) and both:
        print("Шаблон промпту НЕ змінювався між прогонами: усі відповіді")
        print("збігаються в межах 1%. Старі рядки однорідні з рештою.")
    else:
        print("Є розбіжності. Це НЕ доводить зміну шаблону саме по собі —")
        print("модель не детермінована за межами temperature=0 лише в тій")
        print("мірі, в якій провайдер це гарантує. Але однорідність старих")
        print("рядків з рештою вибірки більше не можна стверджувати, і в")
        print("прогін беруться перегенеровані.")

    print()
    for name, path in (("старий", arch if os.path.exists(arch) else base),
                       ("новий", new)):
        a = cache_io.audit(path)
        print(f"  {name}: рядків {a['rows']}, справжніх {a['real']}, "
              f"фальшивих {a['transport']}, дублікатів {a['duplicates']}")


if __name__ == "__main__":
    main()
