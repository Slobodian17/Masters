"""Table extraction from PDF, with multi-page stitching.

camelot in lattice mode is the primary parser and pdfplumber the fallback.
Both work page by page, so fragments are passed through `table_stitching`
before being returned.
"""
import re

import pandas as pd

from ingestion.table_stitching import stitch_fragments


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Parse thousands-separated numbers; leave non-numeric cells alone."""
    def conv(x):
        s = str(x).strip().replace(",", "").replace(" ", "")
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return float(s)
        return x
    return series.apply(conv)


def extract_fragments(pdf_path: str) -> list:
    """Raw per-page fragments, before stitching.

    Exposed separately so the stitching ablation can be measured.
    """
    frags = _extract_with_camelot(pdf_path)
    if not frags:
        frags = _extract_with_pdfplumber(pdf_path)
    return frags


def extract_tables(pdf_path: str, stitch: bool = True) -> list[pd.DataFrame]:
    """One DataFrame per logical table; the first row becomes the header.

    `stitch=False` returns raw page fragments, for the ablation.
    """
    frags = extract_fragments(pdf_path)
    if not stitch:
        return [f["df"] for f in frags]
    return [f["df"] for f in stitch_fragments(frags)]


def extract_tables_with_meta(pdf_path: str) -> list:
    """Like `extract_tables`, but keeping which pages each table came from."""
    return stitch_fragments(extract_fragments(pdf_path))


def _extract_with_camelot(pdf_path: str) -> list:
    try:
        import camelot
    except ImportError:
        return []
    try:
        result = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
    except Exception:
        return []
    out = []
    for t in result:
        df = t.df.copy()
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        for col in df.columns[1:]:
            df[col] = _clean_numeric(df[col])
        out.append({"df": df, "page": int(getattr(t, "page", 1))})
    return out


def _extract_with_pdfplumber(pdf_path: str) -> list:
    import pdfplumber
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for raw in page.extract_tables():
                if not raw or len(raw) < 2:
                    continue
                df = pd.DataFrame(raw[1:], columns=raw[0])
                for col in df.columns[1:]:
                    df[col] = _clean_numeric(df[col])
                out.append({"df": df, "page": page_no})
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_report.pdf"
    raw = extract_fragments(path)
    merged = extract_tables_with_meta(path)
    print(f"Фрагментів з парсера: {len(raw)}  ->  логічних таблиць після зшивання: {len(merged)}")
    for i, m in enumerate(merged):
        df = m["df"]
        note = f", зшито з {m['stitched_from']} фрагментів (сторінки {m['pages']})" if m["stitched_from"] > 1 else ""
        print(f"\n--- Таблиця {i} ({df.shape[0]} рядків x {df.shape[1]} стовпців){note} ---")
        print(df.head(3).to_string(index=False))
