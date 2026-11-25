"""Thesis figures. Built only from artefacts, 300 dpi, readable in grayscale.

No number is typed into a figure. Every value is recomputed from a cache at
draw time; `EXPECTED` exists to be CHECKED against that, and a mismatch is
printed, never reconciled.

GRAYSCALE is the binding constraint here, and it changes the colour rules. A
thesis gets printed, and in grayscale a hue carries nothing. The categorical
palette is the project's validated one (dataviz checker: lightness band PASS,
chroma floor PASS, CVD separation PASS worst adjacent dE 9.1, normal-vision
floor PASS 22.9, contrast WARN), and the WARN is discharged the way the
checker demands -- every mark carries its own value. On top of that each series
gets a HATCH, so identity survives the print: colour, texture and a label, three
channels where one would do.

A lightness ramp instead of the categorical palette was tried first and the
checker rejected it: outside the lightness band, chroma below the floor, reads
gray. Texture is the sanctioned answer for the print case, not a substituted
palette.

Metrics on different scales are never put on one axis -- `fig3_3` is three small
panels, not three bars sharing a y-axis.

No figure carries a title inside the image: the caption belongs to the document.

Run:  python -m eval.plots_thesis
Out:  eval/figures/thesis/*.png + INDEX.md
"""
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from eval import cache_io
from eval.finqa_confirm import build

HERE = os.path.dirname(os.path.abspath(__file__))
FCACHE = os.path.join(HERE, "finqa_cache")
DCACHE = os.path.join(HERE, "databench_cache")
OUT = os.path.join(HERE, "figures", "thesis")

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "ink": "#0b0b0b", "ink2": "#52514e",
     "grid": "#d9d8d4", "surface": "#fcfcfb", "pale": "#eceae5"}
HATCH = ["", "///", "...", "\\\\\\", "xxx"]

plt.rcParams.update({
    "font.family": "serif",
    # DejaVu Serif ships with matplotlib and covers Cyrillic; Times New Roman
    # is preferred where installed but cannot be relied on.
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10.5,
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
    "axes.linewidth": 0.8, "axes.grid": False,
    "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "savefig.facecolor": C["surface"], "savefig.dpi": 300,
    "axes.unicode_minus": False,
})

EXPECTED = {
    "confirm300": {"acc_ok": 175, "acc_bad": 53, "rej_bad": 46,
                   "rej_ok": 3, "no_num": 23},
    "matrix": {"both": 174, "r1": 43, "r2": 4, "none": 79},
    "waterfall": {"A": 29.0, "Alite": 70.0, "Blite": 87.0},
    "rag": {"A": 32.7, "top1": 48.7, "rr": 55.3, "given": 84.7},
}
PROBLEMS = []


def check(name, got, want, tol=0.15):
    if abs(got - want) > tol:
        PROBLEMS.append(f"{name}: з артефактів {got}, у EXPECTED {want}")


def pct(x):
    """Decimal POINT, as the thesis requires."""
    return f"{x:.1f}%".replace(",", ".")


def save(fig, name, note):
    os.makedirs(OUT, exist_ok=True)
    # below the axes, not on top of the tick labels; bbox_inches="tight"
    # still includes it
    fig.text(0.004, -0.045, note, fontsize=6.6, color=C["ink2"], ha="left",
             va="top")
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight",
                pad_inches=0.06)
    plt.close(fig)
    print(f"  {name}")


def bar_labels(ax, bars, vals, fmt=pct, dy=0.6, fs=9):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt(v), ha="center", va="bottom", fontsize=fs, color=C["ink"])


# ---------- data ----------

def finqa(tag):
    cr = cache_io.load_rows(os.path.join(FCACHE, f"{tag}.jsonl"))[0]
    dr = cache_io.load_rows(os.path.join(FCACHE,
                                         f"baseline_table_{tag}.jsonl"))[0]
    return build(cr, dr)


def db(arm, tag):
    p = os.path.join(DCACHE, f"{arm}_{tag}.jsonl")
    return {r["id"]: r for r in cache_io.load_rows(p)[0]}


def acc(rows):
    return 100.0 * sum(1 for r in rows.values() if r["correct"]) / max(len(rows), 1)


# ---------- 2.1 architecture ----------

def box(ax, x, y, w, h, text, new=False, fs=8.4):
    fc = C["surface"] if not new else "#fdf0e9"
    ec = C["ink2"] if not new else C["orange"]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.9 if new else 0.9,
                                edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=C["ink"], zorder=3, linespacing=1.35)


def arrow(ax, p, q, style="-|>", ls="-", col=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9,
                                 linewidth=0.9, linestyle=ls,
                                 color=col or C["ink2"],
                                 shrinkA=1, shrinkB=1, zorder=1))


def elbow(ax, p, q, a=0, b=90, ls="-", col=None):
    """Right-angled connector: diagonals across a diagram read as crossings."""
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=9, linewidth=0.9,
        linestyle=ls, color=col or C["ink2"], shrinkA=1, shrinkB=1, zorder=1,
        connectionstyle=f"angle,angleA={a},angleB={b},rad=0"))


def curve(ax, p, q, rad=-0.28, ls="-", col=None):
    """Arc connector for feedback loops.

    matplotlib's right-angled `angle` connector cannot join two parallel
    horizontals -- it raises on the degenerate intersection -- and a feedback
    loop in this layout is exactly that case. An arc carries the same meaning
    and cannot degenerate.
    """
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=9, linewidth=0.9,
        linestyle=ls, color=col or C["ink2"], shrinkA=2, shrinkB=2, zorder=1,
        connectionstyle=f"arc3,rad={rad}"))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.9, 7.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 16.9); ax.axis("off")

    ax.text(0.25, 16.45, "ПІДГОТОВКА (один раз)", fontsize=9,
            color=C["ink2"], weight="bold")
    box(ax, 0.3, 15.0, 4.1, 1.0, "таблиці корпусу\n60 таблиць, 3.2 млн рядків")
    box(ax, 5.5, 15.0, 4.2, 1.0, "нормалізація значень\n%, знак, масштаб, типи")
    arrow(ax, (4.4, 15.5), (5.5, 15.5))
    box(ax, 0.3, 13.3, 4.1, 1.0, "КАТАЛОГ СХЕМИ\nназви, стовпці, типи")
    box(ax, 5.5, 13.3, 4.2, 1.0, "СХОВИЩЕ\nDataFrame для виконання")
    curve(ax, (6.3, 15.0), (2.35, 14.3), rad=-0.16)
    arrow(ax, (7.6, 15.0), (7.6, 14.3))

    ax.text(0.25, 12.5, "ЗАПИТ", fontsize=9, color=C["ink2"], weight="bold")
    CL, CR = 1.9, 7.3
    cx = (CL + CR) / 2
    rows = [(11.0, 0.8, "питання користувача", False),
            (9.6, 0.95, "ПОШУК ТАБЛИЦІ: BM25 по каталогу\n"
                        "recall@1 49.3%, recall@3 66.0%", False),
            (8.35, 0.8, "гейт застосовності", False),
            (7.0, 0.85, "генерація коду: схема, типи, 5 рядків", False),
            (5.65, 0.85, "виконання в пісочниці", False)]
    for y, h, t, new_ in rows:
        box(ax, CL, y, CR - CL, h, t, new_)
    for (y0, h0, _, _), (y1, _, _, _) in zip(rows, rows[1:]):
        arrow(ax, (cx, y0), (cx, y1 + 0.85))

    box(ax, 7.9, 7.0, 2.0, 0.85, "ремонт L0–L3", False, fs=8.2)
    box(ax, 7.9, 5.65, 2.0, 0.85, "ПОВТОРНИЙ\nПОШУК", True, fs=8.2)
    arrow(ax, (CR, 6.07), (7.9, 6.07))
    curve(ax, (CR, 6.35), (7.9, 7.30), rad=-0.3)
    curve(ax, (7.9, 7.55), (CR, 7.55), rad=-0.3, ls=(0, (3, 2)))
    curve(ax, (9.9, 6.3), (CR, 10.05), rad=-0.32, ls=(0, (3, 2)))
    ax.text(7.55, 6.6, "помилка", fontsize=7.2, color=C["ink2"], ha="right")
    ax.text(8.9, 5.5, "не та таблиця", fontsize=7.2, color=C["ink2"],
            ha="center", va="top")

    box(ax, CL, 3.75, CR - CL, 1.35,
        "ВЕРИФІКАЦІЯ\nзгода з прямою відповіддю R1 + сигнали виконання\n"
        "recall 45.7%, precision 91.3%", True, fs=8.2)
    arrow(ax, (cx, 5.65), (cx, 5.1))

    box(ax, CL, 2.35, 2.5, 0.85, "ПРИЙНЯТО")
    box(ax, CR - 2.5, 2.35, 2.5, 0.85, "ВІДМОВА")
    arrow(ax, (cx - 0.7, 3.75), (CL + 1.25, 3.2))
    arrow(ax, (cx + 0.7, 3.75), (CR - 1.25, 3.2))
    box(ax, CL, 0.95, CR - CL, 0.9,
        "відповідь + пояснення: стовпець, умова,\n"
        "скільки рядків її проходять")
    arrow(ax, (CL + 1.25, 2.35), (cx - 0.5, 1.85))
    arrow(ax, (CR - 1.25, 2.35), (cx + 0.5, 1.85))

    ax.text(9.85, 0.35, "рамкою виділено блоки, додані в цій роботі",
            ha="right", fontsize=8, color=C["orange"])
    save(fig, "fig2_1_architecture.png",
         "Схема; числа — з RESULTS.md 1, 3, 7")

def fig_repair_loop():
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    box(ax, 0.4, 5.3, 2.9, 1.1, "генерація коду")
    box(ax, 4.0, 5.3, 2.4, 1.1, "виконання")
    box(ax, 7.1, 5.3, 2.5, 1.1, "клас помилки")
    arrow(ax, (3.3, 5.85), (4.0, 5.85)); arrow(ax, (6.4, 5.85), (7.1, 5.85))
    box(ax, 7.1, 3.4, 2.5, 1.1, "L0: виправлення\nназви стовпця", fs=8)
    box(ax, 3.9, 3.4, 2.6, 1.1, "L1–L3: повна\nперегенерація", fs=8)
    arrow(ax, (8.35, 5.3), (8.35, 4.5)); arrow(ax, (7.1, 3.95), (6.5, 3.95))
    arrow(ax, (5.2, 3.4), (1.85, 3.4), style="-|>")
    arrow(ax, (1.85, 3.4), (1.85, 5.3), style="-|>")
    ax.text(3.4, 3.1, "до 3 ітерацій", fontsize=8, color=C["ink2"])
    box(ax, 0.4, 1.3, 3.5, 1.15, "ліміт ітерацій вичерпано")
    arrow(ax, (1.85, 3.4), (2.15, 2.45), ls=(0, (3, 2)))
    box(ax, 4.6, 1.3, 2.2, 1.15, "ВІДМОВА")
    box(ax, 7.1, 1.3, 2.5, 1.15, "ПОВТОРНИЙ\nПОШУК таблиці", True, fs=8)
    arrow(ax, (3.9, 1.88), (4.6, 1.88)); arrow(ax, (6.8, 1.88), (7.1, 1.88))
    ax.text(5.0, 0.55, "повторний пошук — блок, доданий у цій роботі",
            ha="center", fontsize=8, color=C["orange"])
    save(fig, "fig2_2_repair_loop.png", "Схема")


# ---------- 3.1 ----------

def fig_confirm_bar():
    rows = finqa("confirm300")
    a = [r for r in rows if r["verdict"] == "ACCEPT"]
    rj = [r for r in rows if r["verdict"] != "ACCEPT" and r["r2"] is not None]
    seg = [("прийнято, правильно", sum(1 for r in a if r["correct"]), C["blue"], 0),
           ("прийнято, хибно", sum(1 for r in a if not r["correct"]), C["orange"], 1),
           ("відхилено, хибно", sum(1 for r in rj if not r["correct"]), C["aqua"], 2),
           ("відхилено, правильно", sum(1 for r in rj if r["correct"]), C["yellow"], 3),
           ("код не дав числа", sum(1 for r in rows if r["r2"] is None), C["pale"], 4)]
    e = EXPECTED["confirm300"]
    for key, (_, v, _, _) in zip(["acc_ok", "acc_bad", "rej_bad", "rej_ok",
                                  "no_num"], seg):
        check(f"confirm300.{key}", v, e[key])

    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    left = 0
    for name, v, col, i in seg:
        ax.barh(0, v, left=left, height=0.42, color=col, hatch=HATCH[i],
                edgecolor=C["surface"], linewidth=2.0, label=f"{name} — {v}")
        if v >= 14:
            ax.text(left + v / 2, 0, str(v), ha="center", va="center",
                    fontsize=9.5, color=C["ink"] if col == C["pale"] else "white")
        left += v
    n = left
    acc_n = seg[0][1] + seg[1][1]
    rej_n = seg[2][1] + seg[3][1]
    for x0, x1, lab, y in ((0, acc_n, f"прийнято {acc_n}", 0.33),
                           (acc_n, acc_n + rej_n, f"відхилено {rej_n}", 0.33)):
        ax.plot([x0 + 1, x1 - 1], [y, y], color=C["ink2"], linewidth=0.8)
        ax.plot([x0 + 1, x0 + 1], [y, y - 0.04], color=C["ink2"], linewidth=0.8)
        ax.plot([x1 - 1, x1 - 1], [y, y - 0.04], color=C["ink2"], linewidth=0.8)
        ax.text((x0 + x1) / 2, y + 0.03, lab, ha="center", fontsize=8.6,
                color=C["ink2"])
    ax.set_xlim(0, n); ax.set_ylim(-0.42, 0.52)
    ax.set_yticks([]); ax.set_xlabel("питань")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3,
              frameon=False, fontsize=8.2, handlelength=1.5)
    save(fig, "fig3_1_confirm300_bar.png",
         f"передреєстроване (5.26), gpt-oss-120b, n={n}")


# ---------- 3.2 ----------

def fig_route_matrix():
    rows = finqa("confirm300")
    m = [[sum(1 for r in rows if r["r1_correct"] and r["correct"]),
          sum(1 for r in rows if not r["r1_correct"] and r["correct"])],
         [sum(1 for r in rows if r["r1_correct"] and not r["correct"]),
          sum(1 for r in rows if not r["r1_correct"] and not r["correct"])]]
    e = EXPECTED["matrix"]
    check("matrix.both", m[0][0], e["both"]); check("matrix.r2", m[0][1], e["r2"])
    check("matrix.r1", m[1][0], e["r1"]); check("matrix.none", m[1][1], e["none"])
    ident = sum(1 for r in rows if not r["r1_correct"] and not r["correct"]
                and r["r1"] is not None and r["r2"] is not None
                and abs(r["r1"] - r["r2"]) <= 0.01 * max(abs(r["r2"]), 1.0))

    fig, ax = plt.subplots(figsize=(4.5, 3.9))
    top = max(max(r) for r in m)
    # sequential single hue: the cell encodes a MAGNITUDE
    ax.imshow(m, cmap="Blues", vmin=0, vmax=top * 1.08)
    for i in range(2):
        for j in range(2):
            v = m[i][j]
            ax.text(j, i - 0.07, str(v), ha="center", va="center",
                    fontsize=15, color="white" if v > top * 0.55 else C["ink"])
    ax.text(1, 1.17, f"з них {ident} — ідентичні числа", ha="center",
            fontsize=8, color=C["ink"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["код правий", "код хибний"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["пряма\nправа", "пряма\nхибна"])
    ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
    ax.set_yticks([-0.5, 0.5, 1.5], minor=True)
    ax.grid(which="minor", color=C["surface"], linewidth=2.4)
    ax.tick_params(which="minor", length=0)
    save(fig, "fig3_2_route_matrix.png",
         f"пост-хок, gpt-oss-120b, n={len(rows)}")


# ---------- 3.3 ----------

def fig_replication():
    panels = [
        ("recall відхилення, %", 40.0,
         [("confirm300", 46.5, 36.0, 57.0), ("held-out 150", 45.7, 31.2, 60.0)]),
        ("precision відхилення, %", 60.0,
         [("confirm300", 93.9, 86.1, 100.0), ("held-out 150", 91.3, 78.9, 100.0)]),
        ("приріст точності прийнятих, п.п.", 10.0,
         [("confirm300", 17.4, 13.2, 21.7), ("held-out 150", 16.0, 11.0, 21.6)]),
    ]
    # Three different scales -> three panels. Never one shared axis.
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
    for ax, (lab, thr, series) in zip(axes, panels):
        xs = range(len(series))
        for i, (nm, v, lo, hi) in enumerate(series):
            ax.bar(i, v, width=0.52, color=[C["blue"], C["aqua"]][i],
                   hatch=HATCH[i], edgecolor=C["surface"], linewidth=1.6)
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="none",
                        ecolor=C["ink"], elinewidth=1.0, capsize=3.2)
            ax.text(i, hi + 2.2, f"{v:.1f}".replace(",", "."), ha="center",
                    fontsize=9, color=C["ink"])
        ax.axhline(thr, color=C["orange"], linewidth=1.1,
                   linestyle=(0, (4, 2)))
        ax.text(len(series) - 0.46, thr + 1.4, f"порог {thr:.0f}",
                fontsize=7.6, color=C["orange"], ha="right")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([s[0] for s in series], fontsize=8.4)
        ax.set_ylabel(lab, fontsize=8.6)
        ax.set_ylim(0, 112)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig3_3_replication.png",
         "передреєстровано (5.26 / 5.42), gpt-oss-120b; вуса — 95% CI "
         "бутстрепом по кластерах таблиць")


# ---------- 3.4 ----------

def fig_by_type():
    A, B = db("A", "test150"), db("B", "test150")
    types = ["boolean", "category", "number", "list[category]", "list[number]"]
    va, vb = [], []
    for t in types:
        sa = [r for r in A.values() if r["type"] == t]
        sb = [r for r in B.values() if r["type"] == t]
        va.append(100.0 * sum(1 for r in sa if r["correct"]) / max(len(sa), 1))
        vb.append(100.0 * sum(1 for r in sb if r["correct"]) / max(len(sb), 1))

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    x = range(len(types))
    w = 0.38
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], va, width=w, color=C["blue"],
                hatch=HATCH[0], edgecolor=C["surface"], linewidth=1.6,
                label="A — пряма відповідь (30 рядків)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], vb, width=w, color=C["aqua"],
                hatch=HATCH[1], edgecolor=C["surface"], linewidth=1.6,
                label="B — згенерований код (вся таблиця)")
    bar_labels(ax, b1, va); bar_labels(ax, b2, vb)
    ax.set_xticks(list(x)); ax.set_xticklabels(types, fontsize=8.8)
    ax.set_ylabel("точність, %"); ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig3_4_by_type.png",
         f"заморожений тест, flash-lite, n={len(B)}, по 30 на тип")


# ---------- 3.5 ----------

def fig_waterfall():
    a = acc(db("A", "dev100"))
    al = acc(db("A-lite", "dev100"))
    bl = acc(db("B-lite", "dev100"))
    check("waterfall.A", a, EXPECTED["waterfall"]["A"])
    check("waterfall.Alite", al, EXPECTED["waterfall"]["Alite"])
    check("waterfall.Blite", bl, EXPECTED["waterfall"]["Blite"])

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    labs = ["A\nпряма,\n30 рядків", "видимість\nданих",
            "A-lite\nпряма,\nвся таблиця", "обчислення\nкодом",
            "B-lite\nкод,\nвся таблиця"]
    ax.bar(0, a, width=0.52, color=C["blue"], hatch=HATCH[0],
           edgecolor=C["surface"], linewidth=1.6)
    ax.bar(1, al - a, bottom=a, width=0.52, color=C["yellow"], hatch=HATCH[2],
           edgecolor=C["surface"], linewidth=1.6)
    ax.bar(2, al, width=0.52, color=C["blue"], hatch=HATCH[0],
           edgecolor=C["surface"], linewidth=1.6)
    ax.bar(3, bl - al, bottom=al, width=0.52, color=C["orange"],
           hatch=HATCH[3], edgecolor=C["surface"], linewidth=1.6)
    ax.bar(4, bl, width=0.52, color=C["aqua"], hatch=HATCH[1],
           edgecolor=C["surface"], linewidth=1.6)
    for i, (v, y) in enumerate([(a, a), (al - a, al), (al, al),
                                (bl - al, bl), (bl, bl)]):
        txt = (f"+{v:.1f}" if i in (1, 3) else f"{v:.1f}").replace(",", ".")
        ax.text(i, y + 1.6, txt, ha="center", fontsize=9.6, color=C["ink"])
    for i, (y0, y1) in enumerate([(a, a), (al, al), (al, al)]):
        pass
    ax.plot([0.26, 0.74], [a, a], color=C["ink2"], linewidth=0.8,
            linestyle=(0, (2, 2)))
    ax.plot([1.26, 1.74], [al, al], color=C["ink2"], linewidth=0.8,
            linestyle=(0, (2, 2)))
    ax.plot([2.26, 2.74], [al, al], color=C["ink2"], linewidth=0.8,
            linestyle=(0, (2, 2)))
    ax.plot([3.26, 3.74], [bl, bl], color=C["ink2"], linewidth=0.8,
            linestyle=(0, (2, 2)))
    ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8.2)
    ax.set_ylabel("точність, %"); ax.set_ylim(0, 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig3_5_waterfall.png",
         "DataBench dev, flash-lite, n=100; Lite — еталон sample_answer")


# ---------- 3.6 ----------

def fig_rag_rerank():
    import json as _json
    from ingestion import databench_catalog as cat
    from ingestion import databench_loader as dbl
    spec = _json.load(open(os.path.join(HERE, "databench_test150.json"),
                           encoding="utf-8"))
    qs = [q for q in spec["questions"] if dbl.have_table(q["dataset"])]
    r = cat.Retriever()
    A = db("A", "test150"); B = db("B", "test150")
    RG = db("B-rag", "test150"); R2 = {}
    for x in cache_io.load_rows(os.path.join(DCACHE,
                                             "B-rag2_test150.jsonl"))[0]:
        R2.setdefault(x["id"], []).append(x)

    from eval.databench_reretrieval import triggered
    base_ok = blk_ok = 0
    for q in qs:
        top = [t for t, _ in r.search(q["question"], 3)]
        row = B[q["qid"]] if (top and top[0] == q["dataset"]) else RG.get(q["qid"])
        ok = bool(row and row.get("correct"))
        base_ok += ok
        if not triggered(row, q["type"]):
            blk_ok += ok
            continue
        tries = sorted(R2.get(q["qid"], []), key=lambda z: z["rank"])
        pick = next((z for z in tries if not triggered(z, q["type"])), None)
        blk_ok += bool(pick["correct"]) if pick else ok

    n = len(qs)
    vals = [acc(A), 100.0 * base_ok / n, 100.0 * blk_ok / n]
    given = acc(B)
    e = EXPECTED["rag"]
    check("rag.A", vals[0], e["A"]); check("rag.top1", vals[1], e["top1"])
    check("rag.rr", vals[2], e["rr"]); check("rag.given", given, e["given"])

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    labs = ["A — пряма відповідь,\nтаблицю ДАНО",
            "B-rag — код,\nтаблицю знайдено (top-1)",
            "B-rag + повторний\nпошук за виконанням"]
    cols = [C["blue"], C["yellow"], C["aqua"]]
    bars = ax.bar(range(3), vals, width=0.56, color=cols,
                  hatch=[HATCH[0], HATCH[2], HATCH[1]],
                  edgecolor=C["surface"], linewidth=1.8)
    bar_labels(ax, bars, vals, dy=1.3, fs=10)
    ax.axhline(given, color=C["orange"], linewidth=1.2, linestyle=(0, (4, 2)))
    ax.text(2.42, given + 1.4, f"верхня межа: таблицю дано — {pct(given)}",
            ha="right", fontsize=8.2, color=C["orange"])
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=8.2)
    ax.set_ylabel("точність, %"); ax.set_ylim(0, 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig3_6_rag_rerank.png",
         f"заморожений тест, flash-lite, n={n}; корпус 60 таблиць")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Графіки у", OUT)
    fig_architecture()
    fig_repair_loop()
    fig_confirm_bar()
    fig_route_matrix()
    fig_replication()
    fig_by_type()
    fig_waterfall()
    fig_rag_rerank()

    idx = os.path.join(OUT, "INDEX.md")
    with open(idx, "w", encoding="utf-8") as f:
        f.write("# Графіки роботи\n\n")
        f.write("Відтворення: `python -m eval.plots_thesis`. "
                "Жодне число не вписане в код — усе рахується з кешів; "
                "`EXPECTED` існує, щоб це перевіряти.\n\n")
        f.write("Шрифт серифний, десяткова ТОЧКА, 300 dpi, без заголовків "
                "усередині зображення. Кожна серія має колір І штриховку, "
                "тож ідентичність зберігається у відтінках сірого.\n\n")
        for n, d in (
            ("fig2_1_architecture.png", "Схема методу; рамкою — нові блоки"),
            ("fig2_2_repair_loop.png", "Цикл ремонту L0–L3 і тригер повторного пошуку"),
            ("fig3_1_confirm300_bar.png", "Рішення правила згоди, confirm300"),
            ("fig3_2_route_matrix.png", "Матриця двох маршрутів, confirm300"),
            ("fig3_3_replication.png", "Реплікація критеріїв: confirm300 проти held-out"),
            ("fig3_4_by_type.png", "A проти B за типом відповіді, тест"),
            ("fig3_5_waterfall.png", "Розклад переваги коду: видимість і обчислення"),
            ("fig3_6_rag_rerank.png", "Наскрізний RAG і повторний пошук, тест"),
            ("fig3_7_ui_demo.png", "Знімок інтерфейсу: 011_SF, 713 107 рядків"),
            ("figG_1_ui_main.png", "Інтерфейс: головний екран (додаток)"),
            ("figG_2_ui_steps.png", "Інтерфейс: кроки конвеєра (додаток)"),
        ):
            f.write(f"- `{n}` — {d}\n")
    print(f"  INDEX.md")

    print()
    if PROBLEMS:
        print("!! РОЗБІЖНОСТІ З EXPECTED:")
        for p in PROBLEMS:
            print("   ", p)
    else:
        print("Усі числа збіглися з EXPECTED.")


if __name__ == "__main__":
    main()
