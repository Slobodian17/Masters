"""Single entry point for loading document tables.

Returns every logical table in the document; choosing the relevant one is the
retrieval layer's job, called separately before the pipeline.
"""
import os

from ingestion.ingest_pdf import extract_tables
from ingestion.make_sample_pdf import make_pdf, make_multipage_pdf
from ingestion.synthetic_data import build_tables, build_cashflow

# Naming by a characteristic row label. Names only matter so generated code
# can address a table; retrieval ranks by content, not by name.
_NAME_RULES = [
    ("income", ("виручка", "чистий прибуток")),
    ("balance", ("всього активів", "власний капітал")),
    ("cashflow", ("рух коштів", "надходження від реалізації")),
]


def _labels_blob(df) -> str:
    if df.shape[1] == 0:
        return ""
    return " ".join(str(v).lower() for v in df.iloc[:, 0].tolist())


def name_tables(dfs: list) -> dict:
    """[DataFrame] -> {name: DataFrame}; unrecognised tables become table_N."""
    out, used = {}, set()
    for i, df in enumerate(dfs):
        blob = _labels_blob(df)
        name = None
        for candidate, markers in _NAME_RULES:
            if candidate in used:
                continue
            if any(m in blob for m in markers):
                name = candidate
                break
        if name is None:
            name = f"table_{i}"
        used.add(name)
        out[name] = df
    return out


def load_tables(pdf_path: str = "sample_report.pdf", use_pdf: bool = True) -> dict:
    """All logical tables of the document, generating the PDF if missing."""
    if use_pdf:
        if not os.path.exists(pdf_path):
            if "multipage" in os.path.basename(pdf_path):
                make_multipage_pdf(pdf_path)
            else:
                make_pdf(pdf_path)
        dfs = extract_tables(pdf_path)
        if len(dfs) >= 2:
            return name_tables(dfs)

    # fallback: data straight from synthetic_data, skipping the PDF round-trip
    income, balance = build_tables()
    if "multipage" in os.path.basename(pdf_path):
        return {"income": income, "balance": balance, "cashflow": build_cashflow()}
    return {"income": income, "balance": balance}
