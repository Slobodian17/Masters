"""Does our PDF table extraction find the table the question needs?

Zero API calls. This is the part of the end-to-end test that can fail before
any model is involved, so it is measured on its own first: a method evaluated
on pages where the table was never found would be measuring something else.

Two things are reported per page:
  found      our parser returned at least one table
  covers     at least one returned table contains the numbers the gold
             answer was derived from (`facts`), after normalisation

`covers` is the honest criterion. A parser that returns a table is not useful
if it is the wrong table or lost the cells in question, and TAT-DQA gives the
exact facts an answer used, so this is checkable rather than eyeballed.

Run:  python -m eval.tatdqa_extract --n 40
"""
import argparse
import os
import random
import re
import zipfile

from eval import tatdqa_loader as T

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(T.DATA, "dev")

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm_num(s):
    t = str(s).replace(",", "").replace("$", "").replace("%", "").strip()
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def table_numbers(df):
    out = set()
    for row in df.itertuples(index=False):
        for cell in row:
            v = norm_num(cell)
            if v is not None:
                out.add(v)
                out.add(abs(v))
    return out


def fact_numbers(q):
    out = set()
    for f in (q.get("facts") or []):
        v = norm_num(f)
        if v is not None:
            out.add(v)
            out.add(abs(v))
    return out


def ensure_pdf(uid, zf):
    """Extract one page from the archive, once."""
    path = os.path.join(PDF_DIR, f"{uid}.pdf")
    if not os.path.exists(path):
        os.makedirs(PDF_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(zf.read(f"dev/{uid}.pdf"))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from ingestion.ingest_pdf import extract_tables

    zf = zipfile.ZipFile(T.DOCS_ZIP)
    sel = T.select(zf=zf)
    rng = random.Random(args.seed)
    rng.shuffle(sel)
    sample = sel[:args.n]

    print("=" * 74)
    print("ВИТЯГУВАННЯ ТАБЛИЦЬ З PDF (нуль викликів API)")
    print("=" * 74)
    print(f"придатних питань у dev: {len(sel)}   вибірка: {len(sample)} "
          f"(seed {args.seed})")
    print()

    found = covers = 0
    ntab = []
    rows = []
    for i, q in enumerate(sample, start=1):
        path = ensure_pdf(q["doc_uid"], zf)
        try:
            tabs = extract_tables(path)
        except Exception as exc:
            tabs = []
            print(f"[{i}/{len(sample)}] ЗБІЙ ПАРСЕРА {type(exc).__name__}: "
                  f"{str(exc)[:70]}")
        ntab.append(len(tabs))
        ok_found = len(tabs) > 0
        facts = fact_numbers(q)
        ok_cov = False
        if ok_found and facts:
            for df in tabs:
                if facts <= table_numbers(df):
                    ok_cov = True
                    break
        found += ok_found
        covers += ok_cov
        rows.append((q["doc_uid"], len(tabs), ok_found, ok_cov, sorted(facts)[:4]))
        print(f"[{i}/{len(sample)}] {q['doc_uid'][:12]}  таблиць={len(tabs)}  "
              f"знайдено={'так' if ok_found else 'НІ '}  "
              f"містить факти={'так' if ok_cov else 'НІ '}")

    n = len(sample)
    print()
    print("--- ПІДСУМОК ---")
    print(f"  сторінок                          : {n}")
    print(f"  парсер повернув хоч одну таблицю  : {found} ({found / n:.0%})")
    print(f"  таблиця МІСТИТЬ потрібні клітинки : {covers} ({covers / n:.0%})")
    print(f"  таблиць на сторінку: середнє {sum(ntab) / n:.1f}, "
          f"максимум {max(ntab) if ntab else 0}")
    multi = sum(1 for k in ntab if k > 1)
    print(f"  сторінок з кількома таблицями     : {multi} ({multi / n:.0%}) "
          f"-- саме тут потрібен пошук потрібної таблиці (М2)")
    print()
    print("  «містить факти» — це стеля для наскрізного прогону: там, де")
    print("  клітинок немає в жодній витягнутій таблиці, жодна модель")
    print("  правильної відповіді дати не може.")

    out = os.path.join(HERE, "tatdqa_extract.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("doc_uid,n_tables,found,covers,facts\n")
        for u, k, a, b, fx in rows:
            f.write(f"{u},{k},{int(a)},{int(b)},\"{fx}\"\n")
    print("Порядкові дані:", out)


if __name__ == "__main__":
    main()
