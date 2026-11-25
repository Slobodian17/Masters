"""FinQA loader.

Reads the original release rather than the HuggingFace mirror. The mirror
exposes only id/question/answer/table/pre_text/post_text/gold_evidence and
drops `qa.program`, `qa.ann_table_rows` and `qa.ann_text_rows` — which are
exactly the fields the audit and the program-accuracy surrogate need. It also
needs the `datasets` package; the original is plain JSON over HTTP and needs
nothing that is not already installed.

Files are cached under data/finqa/ and fetched once.

A FinQA table is list[list[str]] with row 0 as the header and the first column
as the row label. Cells carry presentation, not numbers: "$ 1,234", "( 567 )"
for a negative, "14.1%", "—" for missing. `to_frame` separates the label
column from the parsed numeric columns and records what it could not parse,
because an unparsed cell is not a blank — it is a value the pipeline will not
see, and its rate belongs in the audit rather than in a silent default.
"""
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset/"
SPLITS = ("train", "dev", "test")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "finqa")

TIMEOUT = 120


def local_path(split: str) -> str:
    return os.path.join(DATA_DIR, f"{split}.json")


def fetch(split: str, force: bool = False) -> str:
    """Download one split if it is not cached yet. -> local path."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = local_path(split)
    if os.path.exists(path) and not force:
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    with urllib.request.urlopen(BASE_URL + f"{split}.json", timeout=TIMEOUT) as r:
        raw = r.read()
    # Parse before writing, so a truncated download cannot be cached as valid.
    json.loads(raw)
    with open(path, "wb") as f:
        f.write(raw)
    return path


def load_split(split: str = "test") -> list:
    with open(fetch(split), encoding="utf-8") as f:
        return json.load(f)


# ---- numeric parsing ----

_MISSING = {"", "-", "–", "—", "n/a", "na", "nm", "nil", "none"}
_NUM_RE = re.compile(r"^[-+]?\d*\.?\d+$")

# FinQA writes a negative twice: a signed number followed by its magnitude in
# brackets, as in "-32 ( 32 )" or "$ -4614 ( 4614 )". Measured on the test
# split, missing this form loses 19% of all numeric cells, concentrated in the
# tables that contain the most negatives.
_SIGNED_WITH_MAGNITUDE = re.compile(
    r"^\$?\s*(?P<v>[-+]?[\d,]*\.?\d+)\s*\(\s*[\d,]*\.?\d+\s*\)$")


def parse_number(cell: str):
    """One FinQA cell -> float, or None when it holds no number.

    Returns (value, was_percent) so a caller can tell 14.1% from 14.1; FinQA
    itself is inconsistent about whether a percentage answer is scaled, and
    that inconsistency has to be visible rather than absorbed here.
    """
    s = str(cell).strip().lower()
    if s in _MISSING:
        return None, False

    m = _SIGNED_WITH_MAGNITUDE.match(s)
    if m:
        return float(m.group("v").replace(",", "")), False

    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1].strip()

    percent = s.endswith("%")
    if percent:
        s = s[:-1].strip()

    s = s.replace("$", "").replace(",", "").replace(" ", "")
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]

    if not _NUM_RE.match(s):
        return None, percent
    v = float(s)
    return (-v if negative else v), percent


# A value carries a marker a column name does not: a currency sign in front of
# a digit, thousands grouping, a bracketed negative, or a percent. "$ change"
# and "( square feet in millions )" are captions and must not match.
_VALUE_MARKER = re.compile(r"\$\s*[-(]?\d|\d,\d{3}|\(\s*\d[\d,.]*\s*\)|\d\s*%")


def header_is_data(table: list) -> bool:
    """True when row 0 already holds values and the table has no header.

    Measured on the FinQA test split: 166 of 1147 tables, 14.5%. Treating that
    row as column names silently deletes it -- 2.5% of all data rows -- and the
    scope gate then refuses 26.9% of those questions against a 7.3% baseline,
    because the terms it looks for went out with the row.
    """
    if not table or len(table) < 2 or len(table[0]) < 2:
        return False
    return any(_VALUE_MARKER.search(str(c)) for c in table[0][1:])


def to_frame(table: list) -> tuple:
    """FinQA table -> (DataFrame, stats).

    The frame keeps the project's convention: column 0 holds row labels as
    strings, every other column is numeric. `stats` reports how much of the
    table survived parsing.
    """
    if not table or len(table) < 2:
        return pd.DataFrame(), {"rows": 0, "cols": 0, "cells": 0,
                                "parsed": 0, "unparsed_examples": []}

    headerless = header_is_data(table)
    if headerless:
        # Positional names, because the source has none. Losing the row would
        # be worse than not knowing which period each column is: the gold
        # programs reference values, not column names.
        header = ["line item"] + [f"col_{i}" for i in range(1, len(table[0]))]
        body = table
    else:
        header = [str(h).strip() for h in table[0]]
        body = table[1:]

    # A blank first header is normal: it sits above the label column.
    label_col = header[0] if header[0] else "line item"
    cols = []
    for i, h in enumerate(header[1:], start=1):
        cols.append(h if h else f"col_{i}")

    labels, rows = [], []
    cells = parsed = 0
    unparsed = []
    for r in body:
        r = list(r) + [""] * (len(header) - len(r))
        labels.append(str(r[0]).strip())
        vals = []
        for c in r[1:len(header)]:
            cells += 1
            v, _ = parse_number(c)
            if v is None:
                if str(c).strip().lower() not in _MISSING and len(unparsed) < 5:
                    unparsed.append(str(c))
                vals.append(np.nan)
            else:
                parsed += 1
                vals.append(v)
        rows.append(vals)

    df = pd.DataFrame({label_col: labels})
    for j, name in enumerate(cols):
        # Duplicate headers occur; keep them addressable.
        key = name if name not in df.columns else f"{name}_{j}"
        df[key] = [row[j] if j < len(row) else np.nan for row in rows]

    return df, {"rows": len(body), "cols": len(cols), "cells": cells,
                "parsed": parsed, "unparsed_examples": unparsed,
                "headerless": headerless}


def to_tables(record: dict) -> dict:
    """One record -> {name: DataFrame}, the shape `pipeline.run` expects.

    Gold context is one table per question, so the mapping is a single entry.
    The corpus-retrieval mode builds its pool from many records and does its
    own naming; it does not go through here.
    """
    df, _ = to_frame(record.get("table") or [])
    return {"table": df} if not df.empty else {}


# ---- gold fields ----

def gold_answer(record: dict):
    """Executed gold answer as a float, or None when it is not numeric."""
    qa = record.get("qa", {})
    v = qa.get("exe_ans")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        parsed, _ = parse_number(v)
        return parsed
    return None


def gold_program(record: dict) -> str:
    return str(record.get("qa", {}).get("program", "") or "")


_OP_RE = re.compile(r"([a-z_]+)\(", re.IGNORECASE)


def program_ops(program: str) -> list:
    """Operation names in a FinQA program, in order.

    "subtract(206588, 181001), divide(#0, 181001)" -> ["subtract", "divide"]
    Reference numbers (#0) and constants are dropped: only the operation
    sequence is used, and only as a surrogate for program accuracy.
    """
    return [m.group(1).lower() for m in _OP_RE.finditer(program or "")]


def evidence_source(record: dict) -> str:
    """Where the gold evidence lives: 'table', 'text' or 'both'.

    This is the ceiling on what a table-only pipeline can answer, so it is
    read from the annotation rather than guessed from the question.
    """
    qa = record.get("qa", {})
    has_table = bool(qa.get("ann_table_rows"))
    has_text = bool(qa.get("ann_text_rows"))
    if has_table and has_text:
        return "both"
    if has_table:
        return "table"
    if has_text:
        return "text"
    return "none"


# ---- corpus mode ----
#
# THE POOL IS NOT CAPPED, and that is a measured decision rather than a
# default. Capping at the five largest tables drops the question's own table
# in 22% of cases (measured on the 50-question sample), and every multi-table
# figure would then describe the cap instead of the system. Taking the whole
# filing keeps the gold table in 100% of cases and costs little: a median of
# 24 rows across all its tables, 137 at the worst.
#
# Deduplication is BY CONTENT. 66.6% of FinQA table instances are duplicates
# of another instance, so collecting by record id would put copies of one page
# into the pool -- and a cross-table check would then confirm a value against
# a copy of the very table it came from.
import hashlib as _hashlib
import random as _random


def content_hash(table) -> str:
    body = "|".join(";".join(re.sub(r"\s+", " ", str(c)).strip().lower()
                             for c in row) for row in table)
    return _hashlib.md5(body.encode()).hexdigest()


def filing_of(record: dict):
    parts = (record.get("filename") or record.get("id", "")).split("/")
    return (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], "")


def build_filing_pool(records: list) -> dict:
    """{filing: {content_hash: raw table}} over whatever splits are given."""
    pool = {}
    for r in records:
        t = r.get("table")
        if t:
            pool.setdefault(filing_of(r), {})[content_hash(t)] = t
    return pool


def corpus_tables(record: dict, pool: dict, seed: int = 42) -> dict:
    """Every distinct table of the record's filing, in shuffled order.

    The order is shuffled so that position cannot serve as a cue for which
    table is the right one, and seeded so the arrangement is reproducible.
    Names are positional (`table_0`...) for the same reason: a name derived
    from content would leak the answer.
    """
    tabs = pool.get(filing_of(record), {})
    if not tabs:
        return to_tables(record)

    items = sorted(tabs.items())
    _random.Random(seed).shuffle(items)

    out = {}
    for i, (_, raw) in enumerate(items):
        df, _st = to_frame(raw)
        if not df.empty:
            out[f"table_{i}"] = df
    return out or to_tables(record)


def gold_table_name(record: dict, tables: dict) -> str:
    """Which entry of a corpus pool is the question's own table."""
    own = to_tables(record).get("table")
    if own is None:
        return ""
    for name, df in tables.items():
        if df.shape == own.shape and df.iloc[:, 0].tolist() == own.iloc[:, 0].tolist():
            return name
    return ""


CAPTION_ENV = "VKSK_USE_CAPTION"


def use_caption() -> bool:
    """Off by default. The caption is an option, not a change of method."""
    return os.environ.get(CAPTION_ENV, "0").strip() not in ("", "0", "false", "no")


def caption(record: dict) -> str:
    """The sentence immediately before the table.

    Metadata, not data. It names what the table holds and often states the
    unit; measured on the test split, a unit appears there and nowhere in the
    table in 20% of questions, and its terms would lift 72.5% of the scope
    gate's false refusals. No NUMBER from it is allowed into a computation --
    `pipeline.foreign_literals` records any that appear in generated code.
    """
    pre = record.get("pre_text") or []
    return str(pre[-1]).strip() if pre else ""


def question(record: dict) -> str:
    return str(record.get("qa", {}).get("question", ""))
