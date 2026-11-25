import os
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


# pdfplumber's default strategy looks for ruling lines. Financial report
# tables are usually borderless, so on real filing pages it either finds
# nothing or shreds the table into one-row fragments that the `len < 2` filter
# then drops. Measured on 20 TAT-DQA pages: the line strategy found a table on
# 11, the text strategy on 20 of 20.
#
# Off by default all the same: the Phase 0 baseline was measured on
# sample_report.pdf with the line strategy, and that number has to stay
# reproducible as measured.
TEXT_TABLES_ENV = "VKSK_PDF_TEXT_TABLES"

_TEXT_SETTINGS = {"vertical_strategy": "text", "horizontal_strategy": "text"}


def text_tables() -> bool:
    return os.environ.get(TEXT_TABLES_ENV, "0").strip() not in ("", "0", "false", "no")


# The text strategy finds a grid on every page, and on a filing page most of
# that grid is prose: measured on 40 pages, the median fragment was 61 rows x
# 11 columns with 13% numeric cells, against 6 x 5 and 68% for the annotated
# table. An honest "table found" metric scored it 2%.
#
# So the grid has to be segmented the way a reader segments it: a table is a
# run of rows that carry figures. One interruption is allowed, because a
# sub-heading ("Operating expenses:") sits inside a table without ending it.
TABLE_FILTER_ENV = "VKSK_PDF_TABLE_FILTER"
MIN_NUM_CELLS = int(os.environ.get("VKSK_PDF_MIN_NUM_CELLS", "2"))
MIN_TABLE_ROWS = int(os.environ.get("VKSK_PDF_MIN_TABLE_ROWS", "2"))
MAX_GAP_ROWS = int(os.environ.get("VKSK_PDF_MAX_GAP_ROWS", "1"))

_NUMCELL = re.compile(r"^\(?-?[\d,]+\.?\d*\)?%?$")


def table_filter() -> bool:
    return os.environ.get(TABLE_FILTER_ENV, "0").strip() not in ("", "0", "false", "no")


def _numeric_count(row) -> int:
    k = 0
    for c in row:
        s = str(c).replace("$", "").replace(" ", "").strip()
        if s and _NUMCELL.match(s):
            k += 1
    return k


def segment_table(raw) -> list:
    """Split a page-sized grid into the runs of rows that are actually tables."""
    groups, cur, gap = [], [], 0
    for row in raw:
        if _numeric_count(row) >= MIN_NUM_CELLS:
            cur.append(row)
            gap = 0
        elif cur:
            gap += 1
            if gap > MAX_GAP_ROWS:
                groups.append(cur)
                cur, gap = [], 0
            else:
                cur.append(row)          # keep a sub-heading inside the table
        # a row before any figures is a caption; it is dropped
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        if len(g) < MIN_TABLE_ROWS:
            continue
        # drop columns that are empty throughout: the grid invents them
        keep = [j for j in range(max(len(r) for r in g))
                if any(str((r[j] if j < len(r) else "") or "").strip() for r in g)]
        out.append([[(r[j] if j < len(r) else "") for j in keep] for r in g])
    return out


def _unique_columns(header) -> list:
    """Column names that are unique, so `df[col]` is a Series.

    Text-strategy extraction frequently yields repeated or empty headers on a
    borderless table. With duplicates pandas returns a DataFrame from `df[col]`
    and every downstream caller that expects a Series fails -- observed as
    `'DataFrame' object has no attribute 'unique'` from the L0 repair.
    """
    seen, out = {}, []
    for i, h in enumerate(header):
        name = str(h).strip() if h is not None and str(h).strip() else f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def _extract_with_pdfplumber(pdf_path: str) -> list:
    import pdfplumber
    settings = _TEXT_SETTINGS if text_tables() else None
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            found = (page.extract_tables(settings) if settings
                     else page.extract_tables())
            pieces = []
            for raw in found:
                if not raw:
                    continue
                pieces.extend(segment_table(raw) if table_filter() else [raw])
            for raw in pieces:
                if len(raw) < 2:
                    continue
                df = pd.DataFrame(raw[1:], columns=_unique_columns(raw[0]))
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
