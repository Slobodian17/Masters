"""One corpus of every DataBench table, and BM25 retrieval over its catalog.

Zero LLM calls. Retrieval is lexical and deterministic, so it can be tuned and
re-measured for free, and so the end-to-end number has one less model in it.

What a catalog entry holds, and why each part is there:

  name          `001_Forbes` -> "forbes": the question often names the subject
  column names  the strongest signal -- questions quote column names directly
  dtypes        separates "how many" questions over numeric columns from
                category ones
  sample values kept in the entry but NOT in the search document, because
                measurement contradicted the reason they were put there

Sample values were added on the assumption that a question naming "Germany" or
"Pikachu" matches no column name and would otherwise be unretrievable. Measured
on dev-100 over the 60-table corpus, that assumption is wrong: values only
recall@1 18%, schema only 48%, both together 45%. The values of 3.2 million
rows collide across tables and act as noise, so the document is schema only.
They stay in the catalog because a later per-column retrieval step may want
them.

The variants are within noise of each other (44-48% recall@1 across six
weightings, i.e. one to four questions out of 100), so "schema only" is chosen
for being the simplest, not for winning. It is then applied to the frozen test
UNCHANGED.

Run:  python -m ingestion.databench_catalog
"""
import json
import math
import os
import re

import pandas as pd

from ingestion import databench_loader as db

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "data", "databench")
CAT = os.path.join(CACHE, "catalog.json")

VALUES_PER_COL = 10
VALUE_CHARS = 40

STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "it", "its", "this", "that", "these",
    "those", "with", "from", "by", "as", "how", "what", "which", "who", "whom",
    "whose", "many", "much", "there", "do", "does", "did", "have", "has", "had",
    "not", "no", "all", "any", "most", "least", "more", "less", "than", "s",
}

_TOK = re.compile(r"[a-z0-9]+")


def tokens(text) -> list:
    return [t for t in _TOK.findall(str(text).lower())
            if t not in STOP and len(t) > 1]


def split_name(ds_id: str) -> str:
    """`001_Forbes` -> `Forbes`; `046_120` keeps its digits."""
    parts = ds_id.split("_", 1)
    return parts[1] if len(parts) > 1 else ds_id


def entry_for(ds_id: str) -> dict:
    df = db.table(ds_id)
    cols, vals = [], []
    for c in df.columns:
        cols.append(str(c))
        s = df[c]
        try:
            uniq = s.dropna().astype(str).unique()[:VALUES_PER_COL]
        except Exception:
            uniq = []
        for v in uniq:
            v = str(v)[:VALUE_CHARS]
            if v:
                vals.append(v)
    return {"id": ds_id, "name": split_name(ds_id),
            "rows": int(df.shape[0]), "cols_n": int(df.shape[1]),
            "columns": cols,
            "dtypes": [str(df[c].dtype) for c in df.columns],
            "values": vals}


def doc_text(e: dict) -> str:
    """The catalog entry as one searchable string: schema, no values."""
    return " ".join([e["name"], e["id"],
                     " ".join(e["columns"]),
                     " ".join(e["dtypes"])])


def build(ids=None, path=CAT, force=False) -> list:
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            cat = json.load(f)
        if ids is None or set(ids) <= {e["id"] for e in cat}:
            return cat
    if ids is None:
        ids = sorted({f.split("__")[0]
                      for f in os.listdir(os.path.join(CACHE, "tables"))
                      if f.endswith("__all.parquet")})
    cat = []
    for i, ds in enumerate(ids, 1):
        try:
            cat.append(entry_for(ds))
        except Exception as exc:
            print(f"  [{i}/{len(ids)}] {ds}: {type(exc).__name__}: {exc}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cat, f, ensure_ascii=False)
    return cat


# ---- BM25, written out rather than imported: no new dependency, and the
# parameters have to be visible because they are part of the method. ----

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokens(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        self.tf = []
        self.df = {}
        for d in self.docs:
            counts = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in self.df.items()}

    def scores(self, query):
        q = tokens(query)
        out = []
        for i, counts in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            s = 0.0
            for t in q:
                f = counts.get(t)
                if not f:
                    continue
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


class Retriever:
    """Catalog + BM25. `search` returns table ids best first."""

    def __init__(self, catalog=None):
        self.catalog = catalog if catalog is not None else build()
        self.ids = [e["id"] for e in self.catalog]
        self.bm25 = BM25([doc_text(e) for e in self.catalog])

    def search(self, question, k=3):
        sc = self.bm25.scores(question)
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], self.ids[i]))
        return [(self.ids[i], sc[i]) for i in order[:k]]


def main():
    cat = build(force=True)
    print(f"каталог: {len(cat)} таблиць")
    tot_cols = sum(e["cols_n"] for e in cat)
    tot_rows = sum(e["rows"] for e in cat)
    print(f"  разом {tot_rows:,} рядків, {tot_cols} стовпців")
    r = Retriever(cat)
    lens = [len(tokens(doc_text(e))) for e in cat]
    lens.sort()
    print(f"  токенів у записі каталогу: медіана {lens[len(lens) // 2]}, "
          f"мін {lens[0]}, макс {lens[-1]}")
    print()
    print("  приклад пошуку:")
    for q in ("How many players are in the Titanic dataset?",
              "What is the average price of listings in London?",
              "Which pokemon has the highest attack?"):
        hits = r.search(q, 3)
        print(f"    {q[:52]:52s} -> "
              + ", ".join(f"{i} ({s:.1f})" for i, s in hits))
    print()
    print("Записано:", CAT)


if __name__ == "__main__":
    main()
