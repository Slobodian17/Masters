"""Generates the demo PDFs used as fixtures.

Fonts are bundled in assets/fonts rather than looked up on system paths, so
generation works the same on any OS; reportlab's built-in fonts have no
Cyrillic glyphs.
"""
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from core import lang
from ingestion.synthetic_data import build_tables, build_cashflow

# Bundled Unicode TTFs; without them Cyrillic renders as boxes.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FONT_DIR = os.path.join(_PROJECT_ROOT, "assets", "fonts")

pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(_FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")))


def _styles():
    styles = getSampleStyleSheet()
    styles["Title"].fontName = "DejaVuSans-Bold"
    styles["Heading2"].fontName = "DejaVuSans-Bold"
    return styles


def _df_to_table(df, with_header: bool = True):
    """`with_header=False` draws data rows only, as on a continuation
    page where the header is not repeated."""
    body = df.astype(str).values.tolist()
    data = ([list(df.columns)] if with_header else []) + body
    first_data_row = 1 if with_header else 0
    for r in range(first_data_row, len(data)):
        for c in range(1, len(data[r])):
            try:
                data[r][c] = f"{float(data[r][c]):,.0f}"
            except ValueError:
                pass
    t = Table(data, repeatRows=1 if with_header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]
    if with_header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE6F0")),
        ]
    t.setStyle(TableStyle(style))
    return t


def make_pdf(path="sample_report.pdf"):
    """Two tables on one page. Frozen: baseline metrics are tied to it."""
    prof = lang.get()
    income, balance = build_tables()
    doc = SimpleDocTemplate(path, pagesize=landscape(A4), topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = _styles()
    elements = [Paragraph(prof.PDF_TITLE, styles["Title"]), Spacer(1, 12)]

    elements.append(_df_to_table(income.reset_index(drop=True)))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(prof.PDF_BALANCE_HEADING, styles["Heading2"]))
    elements.append(Spacer(1, 8))
    elements.append(_df_to_table(balance.reset_index(drop=True)))

    doc.build(elements)
    return path


def make_multipage_pdf(path="sample_report_multipage.pdf", repeat_header: bool = True,
                       split_after: int = 16):
    """Three tables, with the cash flow statement split across pages.

    The break is explicit rather than left to reportlab's pagination, so the
    fixture cannot silently lose its page split on another machine and make
    the stitching test pass without testing anything.
    """
    prof = lang.get()
    income, balance = build_tables()
    cashflow = build_cashflow()

    doc = SimpleDocTemplate(path, pagesize=landscape(A4), topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = _styles()

    head, tail = cashflow.iloc[:split_after], cashflow.iloc[split_after:]

    elements = [
        Paragraph(prof.PDF_MULTIPAGE_TITLE, styles["Title"]), Spacer(1, 12),
        Paragraph(prof.PDF_INCOME_HEADING, styles["Heading2"]), Spacer(1, 8),
        _df_to_table(income.reset_index(drop=True)),
        Spacer(1, 24),
        Paragraph(prof.PDF_BALANCE_HEADING, styles["Heading2"]), Spacer(1, 8),
        _df_to_table(balance.reset_index(drop=True)),
        PageBreak(),
        Paragraph(prof.PDF_CASHFLOW_HEADING, styles["Heading2"]), Spacer(1, 8),
        _df_to_table(head.reset_index(drop=True)),
        PageBreak(),
        Paragraph(prof.PDF_CASHFLOW_CONT_HEADING, styles["Heading2"]), Spacer(1, 8),
        _df_to_table(tail.reset_index(drop=True), with_header=repeat_header),
    ]

    doc.build(elements)
    return path


if __name__ == "__main__":
    from load_data import resolve_pdf
    print("written:", make_pdf(resolve_pdf("sample_report.pdf")))
    print("written:", make_multipage_pdf(resolve_pdf("sample_report_multipage.pdf")))
    print("written:", make_multipage_pdf(
        resolve_pdf("sample_report_multipage_noheader.pdf"), repeat_header=False))
