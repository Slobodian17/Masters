"""Merges fragments of one logical table split across pages.

Both camelot and pdfplumber work page by page, so a table continuing onto the
next page comes back as several independent tables. Half its rows then vanish
from the generator's view, and the failure is silent: the code runs, the
number looks plausible, and no verification signal catches it.

Two break variants occur in practice: the header is repeated on the new page,
or it is missing and the parser treats the first data row as the header. The
second case loses a row outright unless it is demoted back into the data.

Matching is deliberately conservative — failing to merge a split table only
loses rows that were already unreachable, while merging two distinct tables
would invent rows and corrupt every later computation. All conditions are
therefore joined by AND.
"""
import re

import pandas as pd

# A fragment counts as a continuation only on the immediately next page.
MAX_PAGE_GAP = 1

# Label overlap above which fragments are treated as different tables.
MAX_LABEL_OVERLAP = 0.34


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _looks_numeric(s) -> bool:
    t = str(s).strip().replace(",", "").replace(" ", "").replace(" ", "")
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", t))


def _header_is_actually_data(df: pd.DataFrame) -> bool:
    """True when the column names are really the first data row.

    Detected by most non-first columns having numeric "names"; a real header
    holds periods there.
    """
    if df.shape[1] < 2:
        return False
    tail = list(df.columns)[1:]
    numeric = sum(1 for c in tail if _looks_numeric(c))
    return numeric > len(tail) / 2


def _demote_header(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Demote a pseudo-header back into the data and apply real names."""
    first_row = pd.DataFrame([list(df.columns)], columns=list(columns))
    body = df.copy()
    body.columns = list(columns)
    return pd.concat([first_row, body], ignore_index=True)


def _labels(df: pd.DataFrame) -> set:
    if df.shape[1] == 0:
        return set()
    return {_norm(v) for v in df.iloc[:, 0].tolist() if _norm(v)}


def _same_columns(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return [_norm(c) for c in a.columns] == [_norm(c) for c in b.columns]


def _is_continuation(prev: dict, cur: dict) -> bool:
    a, b = prev["df"], cur["df"]

    if cur["page"] - prev["last_page"] > MAX_PAGE_GAP or cur["page"] <= prev["last_page"]:
        return False
    if a.shape[1] != b.shape[1] or a.shape[1] == 0:
        return False

    header_repeated = _same_columns(a, b)
    header_missing = _header_is_actually_data(b)
    if not (header_repeated or header_missing):
        return False

    # A continuation brings new labels. Heavy overlap means a separate table
    # with the same structure, and merging it would duplicate rows.
    la, lb = _labels(a), _labels(b if header_repeated else _demote_header(b, a.columns))
    if la and lb and len(la & lb) / min(len(la), len(lb)) > MAX_LABEL_OVERLAP:
        return False

    return True


def _clean_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns back to float after a demoted header added strings."""
    out = df.copy()
    for col in list(out.columns)[1:]:
        converted = out[col].map(lambda x: float(str(x).replace(",", "").replace(" ", ""))
                                 if _looks_numeric(x) else x)
        if converted.map(lambda x: isinstance(x, float)).all():
            out[col] = converted.astype(float)
        else:
            out[col] = converted
    return out


def stitch_fragments(fragments: list) -> list:
    """Merge page fragments in reading order into logical tables."""
    out = []
    for frag in fragments:
        cur = {
            "df": frag["df"],
            "page": frag["page"],
            "last_page": frag["page"],
            "stitched_from": 1,
            "pages": [frag["page"]],
        }
        if out and _is_continuation(out[-1], cur):
            prev = out[-1]
            piece = cur["df"]
            if not _same_columns(prev["df"], piece):
                piece = _demote_header(piece, prev["df"].columns)
            else:
                piece = piece.copy()
                piece.columns = list(prev["df"].columns)
            prev["df"] = _clean_numeric_cols(
                pd.concat([prev["df"], piece], ignore_index=True)
            )
            prev["last_page"] = cur["last_page"]
            prev["stitched_from"] += 1
            prev["pages"].extend(cur["pages"])
        else:
            out.append(cur)
    return out


def stitch(fragments: list) -> list:
    """Same as `stitch_fragments` but returning only the DataFrames."""
    return [f["df"] for f in stitch_fragments(fragments)]
