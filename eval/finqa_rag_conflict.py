"""Real-RAG ceiling: does retrieval over a company's reports put CONFLICTING
copies of the asked-for cell into the context?

POST-HOC ANALYSIS OF THE DEV SET (confirm300). Gold is used ONLY to locate the
cells the answer needs, for measurement; retrieval itself never sees it.

Corpus. Every distinct FinQA table (train+dev+test, content-hash dedup) of the
question's company, across ALL years -- the setting where a user asks about one
company and the system must find the right report page. The caption (sentence
before the table) is indexed with it, as a page title would be.

Retrieval. TF-IDF cosine between the question and caption+table text, top-k.

What is measured, per question:
  recall@k        the question's own table is in the top-k
  twin cell       another retrieved table has the SAME row label and the SAME
                  column header as a gold operand cell
  conflict        ... and holds a DIFFERENT value there (restatement, other
                  scope) -- the case a direct answer cannot see, while a program
                  that binds (table, row, column) explicitly can enumerate it
  corroborate     ... and holds the SAME value

Zero API calls.

Run:  python -m eval.finqa_rag_conflict
"""
import collections
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from eval import cache_io
from eval.finqa_run import is_correct
from ingestion import finqa_loader as fq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")
KS = (1, 3, 5)
_NUM = re.compile(r"-?\d+\.?\d*")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def gold_operand_values(prog):
    out = []
    for m in _NUM.findall(re.sub(r"const_\w+|#\d+", "", prog or "")):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def cells(df):
    """{(row label, column header): value} for one parsed table."""
    out = {}
    lab = df.columns[0]
    for _, row in df.iterrows():
        for c in df.columns[1:]:
            v = row[c]
            if pd.notna(v):
                out[(norm(row[lab]), norm(c))] = float(v)
    return out


def text_of(raw, cap):
    return cap + " " + " ".join(" ".join(map(str, r)) for r in raw)


def company_corpus(allrec):
    """ticker -> {content hash: (raw table, caption, year)}"""
    corpus = collections.defaultdict(dict)
    for r in allrec:
        t = r.get("table")
        if t:
            co, yr = fq.filing_of(r)
            corpus[co].setdefault(fq.content_hash(t), (t, fq.caption(r), yr))
    return corpus


def retrieve(rec, corpus, k):
    """Top-k tables of the question's company: [(hash, raw, caption, year)].

    Sees only the question text and the company; never the gold table.
    """
    co, _ = fq.filing_of(rec)
    items = list(corpus[co].items())
    docs = [text_of(raw, cap) for _, (raw, cap, _y) in items]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(docs + [fq.question(rec)])
    sims = (X[:-1] @ X[-1].T).toarray().ravel()
    return [(items[j][0],) + items[j][1] for j in np.argsort(-sims)[:k]]


def main():
    code_rows = {r["id"]: r for r in cache_io.load_rows(
        os.path.join(CACHE, "confirm300.jsonl"))[0]}
    dir_rows = {r["id"]: r for r in cache_io.load_rows(
        os.path.join(CACHE, "baseline_table_confirm300.jsonl"))[0]}
    allrec = []
    for sp in ("train", "dev", "test"):
        allrec += fq.load_split(sp)
    byid = {}
    for r in allrec:
        byid.setdefault(r["id"], r)

    # company corpus: ticker -> {hash: (raw, caption, year)}
    corpus = collections.defaultdict(dict)
    for r in allrec:
        t = r.get("table")
        if t:
            co, yr = fq.filing_of(r)
            corpus[co].setdefault(fq.content_hash(t), (t, fq.caption(r), yr))

    out = []
    for qid, cr in code_rows.items():
        rec = byid.get(qid)
        gold = cr.get("gold")
        if rec is None or gold is None:
            continue
        co, yr = fq.filing_of(rec)
        items = list(corpus[co].items())
        own = fq.content_hash(rec["table"])
        docs = [text_of(raw, cap) for _, (raw, cap, _y) in items]
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform(docs + [fq.question(rec)])
        sims = (X[:-1] @ X[-1].T).toarray().ravel()
        order = list(np.argsort(-sims))
        rank = next(i for i, j in enumerate(order) if items[j][0] == own) + 1

        own_df, _ = fq.to_frame(rec["table"])
        own_cells = cells(own_df)
        ops = gold_operand_values(rec["qa"].get("program", ""))
        addr = [a for a, v in own_cells.items()
                if any(abs(v - o) <= 1e-6 * max(1, abs(o)) for o in ops)]
        row = dict(id=qid, n_tables=len(items), rank=rank,
                   r1_ok=bool(is_correct(dir_rows.get(qid, {}).get("answer"),
                                         gold)),
                   n_addr=len(addr))
        for k in KS:
            top = [items[j] for j in order[:k]]
            conflict = corro = twin = 0
            for h, (raw, _c, _y) in top:
                if h == own:
                    continue
                df, _ = fq.to_frame(raw)
                if df.empty:
                    continue
                oc = cells(df)
                for a in addr:
                    if a in oc:
                        twin += 1
                        if abs(oc[a] - own_cells[a]) <= 1e-6 * max(1, abs(own_cells[a])):
                            corro += 1
                        else:
                            conflict += 1
            row[f"recall@{k}"] = rank <= k
            row[f"twin@{k}"] = twin > 0
            row[f"conflict@{k}"] = conflict > 0
            row[f"corro@{k}"] = corro > 0
        out.append(row)

    d = pd.DataFrame(out)
    d.to_csv(os.path.join(HERE, "finqa_rag_conflict_confirm300.csv"), index=False)
    print(f"questions {len(d)}; company corpus size median "
          f"{d.n_tables.median():.0f}, mean {d.n_tables.mean():.1f}, "
          f"max {d.n_tables.max()}; gold operands located in table: "
          f"{(d.n_addr > 0).sum()}")
    for k in KS:
        print(f"k={k}: recall {d[f'recall@{k}'].mean():.1%} | "
              f"twin cell {d[f'twin@{k}'].mean():.1%} | "
              f"CONFLICT {d[f'conflict@{k}'].mean():.1%} | "
              f"corroborate {d[f'corro@{k}'].mean():.1%}")
    print(f"median gold-table rank {d['rank'].median():.0f}")


if __name__ == "__main__":
    main()
