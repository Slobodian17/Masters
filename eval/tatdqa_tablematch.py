"""A metric for "the table was found" that cannot be satisfied by returning junk.

The previous one -- "extract_tables returned a DataFrame" -- scored 100% while
the parser was slicing the whole page's prose into a 61x11 grid with 13%
numeric cells. It measured the fact of a return, not a table.

DEFINITION, fixed before any filter is tuned:

    a fragment MATCHES the annotated table when
      (1) at least 50% of the annotated table's NUMERIC cells appear in it, and
      (2) it is at most 2x the annotated table's row count

    a page COUNTS AS FOUND when at least one extracted fragment matches.

(1) rules out a fragment that missed the table; (2) rules out the page-sized
grid that contains everything and locates nothing. Both are needed: either
alone is trivially satisfiable.

Zero API calls.

Run:  python -m eval.tatdqa_tablematch --split train
"""
import argparse
import os
import random
import zipfile

from eval import tatdqa_loader as T
from eval.tatdqa_extract import norm_num

HERE = os.path.dirname(os.path.abspath(__file__))

CELL_RECALL = 0.50
SIZE_FACTOR = 2.0


def numeric_cells(df):
    out = []
    if df is None:
        return out
    for row in df.itertuples(index=False):
        for c in row:
            v = norm_num(c)
            if v is not None:
                out.append(v)
    return out


def matches(frag, gold, cell_recall=CELL_RECALL, size_factor=SIZE_FACTOR):
    g = numeric_cells(gold)
    if not g:
        return False
    want = set(g) | {abs(v) for v in g}
    have = set(numeric_cells(frag))
    have |= {abs(v) for v in have}
    hit = len(want & have) / len(want)
    size_ok = frag.shape[0] <= size_factor * max(gold.shape[0], 1)
    return hit >= cell_recall and size_ok


def page_found(frags, gold, **kw):
    return any(matches(f, gold, **kw) for f in frags)


def split_pages(seed=42, n_test=40):
    """Test pages are the ones the 40 evaluated questions live on."""
    zf = zipfile.ZipFile(T.DOCS_ZIP)
    sel = T.select(zf=zf)
    random.Random(seed).shuffle(sel)
    test_uids = {q["doc_uid"] for q in sel[:n_test]}
    train_uids, seen = [], set()
    for q in sel[n_test:]:
        u = q["doc_uid"]
        if u in test_uids or u in seen:
            continue
        seen.add(u)
        train_uids.append(q)
    return sel[:n_test], train_uids, zf


def evaluate(questions, zf, **kw):
    from eval.tatdqa_run import gold_table, extracted_table
    found = total = 0
    sizes = []
    for q in questions:
        gt = gold_table(q, zf)
        if not gt:
            continue
        gold = list(gt.values())[0]
        tabs, _ = extracted_table(q)
        frags = list(tabs.values())
        total += 1
        ok = page_found(frags, gold, **kw)
        found += ok
        if frags:
            sizes.append(max(f.shape[0] for f in frags))
    return found, total, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    test_q, train_q, zf = split_pages()
    qs = test_q if args.split == "test" else train_q[:args.limit]

    print("=" * 74)
    print(f"МЕТРИКА «ТАБЛИЦЮ ЗНАЙДЕНО» — вибірка {args.split}")
    print("=" * 74)
    print(f"визначення: >= {CELL_RECALL:.0%} числових клітинок розмітки "
          f"присутні І рядків <= {SIZE_FACTOR:g}x від розмітки")
    print(f"сторінок: {len(qs)}")
    print()
    found, total, sizes = evaluate(qs, zf)
    print(f"  ЗНАЙДЕНО: {found} з {total} = {found / max(total, 1):.0%}")
    if sizes:
        sizes.sort()
        print(f"  рядків у найбільшому фрагменті: медіана {sizes[len(sizes) // 2]}, "
              f"максимум {sizes[-1]}")


if __name__ == "__main__":
    main()
