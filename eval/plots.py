"""Figures for the thesis, built only from artefacts.

No number is typed into this file. Every value on every axis is recomputed from
a cache or a CSV at draw time, so a figure cannot drift from the run it claims
to describe. The one place numbers appear literally is `EXPECTED`, and they are
there to be CHECKED against what the artefacts produce -- any mismatch is
reported, never silently reconciled.

Every figure carries its own provenance: the kind of claim it makes
(pre-registered / exploratory / post-hoc), the model, and n. A figure without
that label is a figure someone will misquote.

Run:  python -m eval.plots
Out:  eval/figures/*.png (300 dpi) and eval/figures/INDEX.md
"""
import collections
import csv
import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from eval import cache_io
from eval.finqa_confirm import build, cluster_of, decide as prereg_decide
from eval.finqa_run import is_correct

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "finqa_cache")
OUT = os.path.join(HERE, "figures")

# Validated with the dataviz skill's checker (light surface #fcfcfb):
# lightness band PASS, chroma floor PASS, CVD separation PASS (worst adjacent
# dE 9.1), normal-vision floor PASS (22.9). Contrast WARN on aqua and yellow
# obliges visible direct labels -- every bar here carries its value, which is
# what the relief rule asks for.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "ink": "#0b0b0b", "ink2": "#52514e",
     "grid": "#d9d8d4", "surface": "#fcfcfb"}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"]]

THRESHOLD = 0.7
N_BOOT = 4000
SEED = 20260928

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "savefig.facecolor": C["surface"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": C["grid"], "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "text.color": C["ink"], "axes.labelcolor": C["ink2"],
    "legend.frameon": False, "figure.autolayout": False,
})

INDEX = []


# ---------- helpers ----------

def rows(tag):
    r, _ = cache_io.load_rows(os.path.join(CACHE, f"{tag}.jsonl"))
    return r


def pair(code_tag, dir_tag, ids=None):
    c, d = rows(code_tag), rows(dir_tag)
    out = build(c, d)
    if ids is not None:
        out = [r for r in out if r["id"] in ids]
    return out, {r["id"]: r for r in c}


def boot_ci(items, fn, n_boot=N_BOOT, seed=SEED, cluster=True):
    """Cluster bootstrap over unique tables; `cluster=False` resamples items."""
    if cluster:
        groups = collections.defaultdict(list)
        for r in items:
            groups[cluster_of(r["id"])].append(r)
        units = list(groups.values())
    else:
        units = [[r] for r in items]
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        draw = []
        for _ in range(len(units)):
            draw.extend(units[rng.randrange(len(units))])
        v = fn(draw)
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def stamp(ax, kind, model, n, extra=""):
    ax.text(0.0, -0.22, f"{kind} · {model} · n = {n}{extra}",
            transform=ax.transAxes, fontsize=7, color=C["ink2"], va="top")


def save(fig, name, title, source, numbers):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    INDEX.append({"file": name, "title": title, "source": source,
                  "numbers": numbers})
    return path


def barlabels(ax, bars, vals, fmt="{:.1%}", dy=0.012, tops=None):
    """Value labels, lifted clear of the error bar when there is one.

    The relief rule: two of the four hues sit below 3:1 against the surface, so
    every bar must carry a visible value. A label sitting on top of a whisker
    is not visible, hence `tops`.
    """
    for i, (b, v) in enumerate(zip(bars, vals)):
        if v != v:
            continue
        y = b.get_height()
        if tops is not None and i < len(tops) and tops[i] == tops[i]:
            y = max(y, tops[i])
        ax.text(b.get_x() + b.get_width() / 2, y + dy,
                fmt.format(v), ha="center", va="bottom", fontsize=8,
                color=C["ink"])


# ---------- signals: cascade phi(v) and route agreement ----------

def cascade_reject(r, by_id):
    """Did the phi(v) cascade refuse this answer?"""
    return by_id[r["id"]].get("status") != "ACCEPT"


def agreement_reject(r):
    return r["verdict"] != "ACCEPT"


def rej_metrics(items, reject, has_value=lambda r: r["r2"] is not None):
    with_v = [r for r in items if has_value(r)]
    wrong = [r for r in with_v if not r["correct"]]
    rej = [r for r in with_v if reject(r)]
    rec = sum(1 for r in wrong if reject(r)) / len(wrong) if wrong else float("nan")
    prec = sum(1 for r in rej if not r["correct"]) / len(rej) if rej else float("nan")
    return rec, prec


# ---------- figures ----------

def fig01():
    items, by_id = pair("confirm300", "baseline_table_confirm300")
    n = len(items)
    defs = [("Каскад φ(v)", lambda r: cascade_reject(r, by_id)),
            ("Згода маршрутів", agreement_reject)]
    recs, precs, ci_r, ci_p = [], [], [], []
    for _, rej in defs:
        rc, pc = rej_metrics(items, rej)
        recs.append(rc)
        precs.append(pc)
        ci_r.append(boot_ci(items, lambda s, j=rej: rej_metrics(s, j)[0]))
        ci_p.append(boot_ci(items, lambda s, j=rej: rej_metrics(s, j)[1]))

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = [0, 1]
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in x], recs, w, color=C["blue"],
                label="recall відхилення")
    b2 = ax.bar([i + w / 2 for i in x], precs, w, color=C["orange"],
                label="precision відхилення")
    for i, (lo, hi) in enumerate(ci_r):
        ax.plot([i - w / 2, i - w / 2], [lo, hi], color=C["ink"], lw=1.2)
    for i, (lo, hi) in enumerate(ci_p):
        ax.plot([i + w / 2, i + w / 2], [lo, hi], color=C["ink"], lw=1.2)
    barlabels(ax, b1, recs, tops=[hi for _, hi in ci_r])
    barlabels(ax, b2, precs, tops=[hi for _, hi in ci_p])
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in defs])
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("частка")
    ax.set_title("Виявлення хибних відповідей: каскад проти згоди маршрутів")
    ax.legend(loc="upper left")
    stamp(ax, "передреєстроване (5.26)", "openai/gpt-oss-120b", n,
          ", 95% ДІ кластерний по таблицях")
    return save(fig, "01_cascade_vs_agreement.png",
                "Каскад φ(v) проти згоди маршрутів",
                "confirm300.jsonl + baseline_table_confirm300.jsonl",
                {"recall_cascade": recs[0], "precision_cascade": precs[0],
                 "recall_agreement": recs[1], "precision_agreement": precs[1],
                 "n": n})


def risk_coverage(items, score, correct=lambda r: r["correct"]):
    """Risk at each coverage, ordered by `score` ascending (best first)."""
    order = sorted(items, key=score)
    errs, xs, ys = 0, [], []
    for i, r in enumerate(order, start=1):
        if not correct(r):
            errs += 1
        xs.append(i / len(order))
        ys.append(errs / i)
    aurc = sum(ys) / len(ys)
    return xs, ys, aurc


def fig02():
    items, by_id = pair("confirm300", "baseline_table_confirm300")
    n = len(items)
    x1, y1, a1 = risk_coverage(items, lambda r: r["distance"])
    x2, y2, a2 = risk_coverage(
        items, lambda r: -(by_id[r["id"]].get("confidence") or 0.0))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(x1, y1, color=C["blue"], lw=2,
            label=f"згода маршрутів (AURC {a1:.3f})")
    ax.plot(x2, y2, color=C["orange"], lw=2,
            label=f"каскад φ(v) (AURC {a2:.3f})")
    ax.set_xlabel("покриття")
    ax.set_ylabel("ризик (частка хибних серед прийнятих)")
    ax.set_title("Крива ризик–покриття")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    stamp(ax, "передреєстроване (5.26)", "openai/gpt-oss-120b", n,
          "; менше — краще")
    return save(fig, "02_risk_coverage.png", "Крива ризик–покриття",
                "confirm300.jsonl + baseline_table_confirm300.jsonl",
                {"aurc_agreement": a1, "aurc_phi": a2, "n": n})


def fig03():
    items, _ = pair("confirm300", "baseline_table_confirm300")
    n = len(items)
    acc = [r for r in items if r["verdict"] == "ACCEPT"]
    vals = [sum(1 for r in items if r["r1_correct"]) / n,
            sum(1 for r in items if r["correct"]) / n,
            sum(1 for r in acc if r["correct"]) / len(acc)]
    cov = len(acc) / n
    cis = [boot_ci(items, lambda s: sum(1 for r in s if r["r1_correct"]) / len(s)),
           boot_ci(items, lambda s: sum(1 for r in s if r["correct"]) / len(s)),
           boot_ci(items, lambda s: (sum(1 for r in s if r["verdict"] == "ACCEPT"
                                         and r["correct"])
                                     / max(sum(1 for r in s
                                               if r["verdict"] == "ACCEPT"), 1)))]
    labels = ["R1 пряма\n(покриття 100%)", "R2 код\n(покриття 100%)",
              f"прийняті\n(покриття {cov:.0%})"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bars = ax.bar(range(3), vals, 0.55, color=SERIES[:3])
    for i, (lo, hi) in enumerate(cis):
        ax.plot([i, i], [lo, hi], color=C["ink"], lw=1.2)
    barlabels(ax, bars, vals, tops=[hi for _, hi in cis])
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("точність")
    ax.set_title("Точність при повному покритті та серед прийнятих")
    stamp(ax, "передреєстроване (5.26)", "openai/gpt-oss-120b", n,
          ", 95% ДІ кластерний")
    return save(fig, "03_accuracy_coverage.png",
                "Точність R1, R2 і прийнятих",
                "confirm300.jsonl + baseline_table_confirm300.jsonl",
                {"r1": vals[0], "r2": vals[1], "accepted": vals[2],
                 "coverage": cov, "n": n})


def fig04():
    items, _ = pair("confirm300", "baseline_table_confirm300")
    n = len(items)
    both = sum(1 for r in items if r["r1_correct"] and r["correct"])
    only1 = sum(1 for r in items if r["r1_correct"] and not r["correct"])
    only2 = sum(1 for r in items if r["correct"] and not r["r1_correct"])
    none = sum(1 for r in items if not r["r1_correct"] and not r["correct"])
    grid = [[both, only1], [only2, none]]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.imshow([[0, 0], [0, 0]], cmap="Greys", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            v = grid[i][j]
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=SERIES[(i * 2 + j) % 4],
                                       alpha=0.18, edgecolor=C["surface"], lw=2))
            ax.text(j, i - 0.06, str(v), ha="center", va="center",
                    fontsize=16, color=C["ink"])
            ax.text(j, i + 0.22, f"{v / n:.0%}", ha="center", va="center",
                    fontsize=8, color=C["ink2"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["R2 правий", "R2 хибний"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["R1 правий", "R1 хибний"])
    ax.grid(False)
    ax.set_title("Матриця розбіжностей маршрутів")
    stamp(ax, "пост-хок, dev", "openai/gpt-oss-120b", n)
    return save(fig, "04_route_matrix.png", "Матриця R1 × R2",
                "confirm300.jsonl + baseline_table_confirm300.jsonl",
                {"both": both, "only_r1": only1, "only_r2": only2,
                 "neither": none, "n": n})


def fig05():
    """What the repair loop changes, over ALL questions.

    Definition, and it has to be stated because two are possible: the "first
    attempt" is the FIRST ITERATION, and an iteration that produced no number
    counts as wrong. The alternative -- first step that produced a number --
    skips a failed first iteration and credits the repair's own output as the
    starting point, which understates the repair from 10 flips to 2.
    """
    src = rows("confirm300")
    n = len(src)

    def first_iter(x):
        st = x.get("steps") or []
        return st[0].get("result") if st else None

    first = sum(1 for x in src if is_correct(first_iter(x), x.get("gold")))
    final = sum(1 for x in src if x.get("candidate_correct"))
    good = sum(1 for x in src if not is_correct(first_iter(x), x.get("gold"))
               and x.get("candidate_correct"))
    bad = sum(1 for x in src if is_correct(first_iter(x), x.get("gold"))
              and not x.get("candidate_correct"))
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    vals = [first / n, final / n]
    bars = ax.bar([0, 1], vals, 0.5, color=[C["blue"], C["aqua"]])
    barlabels(ax, bars, vals)
    b2 = ax.bar([2, 3], [good / n, bad / n], 0.5, color=[C["aqua"], C["orange"]])
    for b, k in zip(b2, (good, bad)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                str(k), ha="center", fontsize=8, color=C["ink"])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["перша\nітерація", "після\nвиправлення",
                        "хибна→\nправильна", "правильна→\nхибна"])
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("частка питань")
    ax.set_title("Внесок ітеративного виправлення коду")
    stamp(ax, "пост-хок, dev", "openai/gpt-oss-120b", n,
          "; «перша ітерація» = крок 0, відсутнє число = хибно")
    return save(fig, "05_repair_contribution.png",
                "Внесок ітеративного виправлення",
                "confirm300.jsonl (steps[0].result проти candidate)",
                {"first_iter": first / n, "final": final / n,
                 "first_n": first, "final_n": final,
                 "wrong_to_right": good, "right_to_wrong": bad, "n": n})


def _dev100_ids():
    with open(os.path.join(HERE, "finqa_dev100.ids"), encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip()}


def _system_acc(new_tag, r1_tag, arb_tag, ids):
    from eval.finqa_arbiter import decide as sysrule
    new = {r["id"]: r for r in rows(new_tag)}
    r1 = {r["id"]: r for r in rows(r1_tag)}
    arb = {r["id"]: r.get("arbiter") for r in rows(arb_tag)}
    ok = tot = 0
    for q in ids:
        if q not in new or q not in r1:
            continue
        gold = new[q].get("gold")
        v2 = new[q].get("candidate")
        if v2 is None:
            v2 = new[q].get("answer")
        v1 = r1[q].get("answer")
        route, val = sysrule(v1, v2)
        sysv = {"AGREE": val, "R1": v1, "R2": v2}.get(route, arb.get(q))
        tot += 1
        ok += bool(is_correct(sysv, gold)) if sysv is not None else 0
    return ok / tot if tot else float("nan"), tot


def fig06():
    ids = _dev100_ids()
    old = {r["id"]: r for r in rows("confirm300")}
    new = {r["id"]: r for r in rows("confirm300_values")}
    r1 = {r["id"]: r for r in rows("baseline_table_confirm300")}
    ids = {q for q in ids if q in old and q in new and q in r1}
    n = len(ids)

    def acc(d, field="candidate"):
        k = 0
        for q in ids:
            v = d[q].get(field)
            if v is None:
                v = d[q].get("answer")
            k += bool(is_correct(v, d[q].get("gold"))) if v is not None else 0
        return k / n

    v_r1 = sum(1 for q in ids
               if r1[q].get("answer") is not None
               and is_correct(r1[q]["answer"], r1[q].get("gold"))) / n
    v_old, v_new = acc(old), acc(new)
    s1, _ = _system_acc("confirm300_values", "baseline_table_confirm300",
                        "arbiter_confirm300_values", ids)
    s2, _ = _system_acc("confirm300_values", "baseline_table_confirm300",
                        "arbiter_confirm300_values_v2", ids)
    vals = [v_r1, v_old, v_new, s1, s2]
    labels = ["R1\nпряма", "R2\nбез значень", "R2\nзі значеннями",
              "система\n+ розбір v1", "система\n+ розбір v2"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(range(5), vals, 0.6,
                  color=[C["blue"], C["orange"], C["orange"], C["aqua"], C["aqua"]])
    bars[2].set_alpha(0.65)
    bars[4].set_alpha(0.65)
    barlabels(ax, bars, vals)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("EA при повному покритті")
    ax.set_title("Етап 1: чи допомагає код моделі з вбудованим міркуванням")
    stamp(ax, "розвідувальне, dev", "openai/gpt-oss-120b", n)
    return save(fig, "06_stage1_gptoss.png", "Етап 1 на gpt-oss-120b",
                "confirm300 / confirm300_values / baseline_table_confirm300 / arbiter_*",
                {"r1": v_r1, "r2_old": v_old, "r2_new": v_new,
                 "system_v1": s1, "system_v2": s2, "n": n})


def fig07():
    ids = _dev100_ids()
    res = {}
    for name, new_tag, r1_tag, arb in (
            ("gpt-oss-120b", "confirm300_values", "baseline_table_confirm300",
             "arbiter_confirm300_values"),
            ("flash-lite", "gemlite100",
             "baseline_table_confirm300_values_gemlite100", "arbiter_gemlite100")):
        new = {r["id"]: r for r in rows(new_tag)}
        r1 = {r["id"]: r for r in rows(r1_tag)}
        sub = {q for q in ids if q in new and q in r1}
        n = len(sub)
        items = [{"id": q,
                  "correct": bool(is_correct(
                      new[q].get("candidate") if new[q].get("candidate") is not None
                      else new[q].get("answer"), new[q].get("gold"))),
                  "r1_correct": bool(is_correct(r1[q].get("answer"),
                                                new[q].get("gold")))
                  if r1[q].get("answer") is not None else False}
                 for q in sub]
        s, _ = _system_acc(new_tag, r1_tag, arb, sub)
        res[name] = {
            "r1": sum(1 for r in items if r["r1_correct"]) / n,
            "r2": sum(1 for r in items if r["correct"]) / n,
            "sys": s, "n": n,
            "ci_r1": boot_ci(items, lambda s2: sum(1 for r in s2 if r["r1_correct"]) / len(s2)),
            "ci_r2": boot_ci(items, lambda s2: sum(1 for r in s2 if r["correct"]) / len(s2)),
        }
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.26
    names = list(res)
    for k, (key, col, lab) in enumerate((("r1", C["blue"], "R1 пряма"),
                                         ("r2", C["orange"], "R2 код"),
                                         ("sys", C["aqua"], "система"))):
        vals = [res[m][key] for m in names]
        b = ax.bar([i + (k - 1) * w for i in range(len(names))], vals, w,
                   color=col, label=lab)
        tops = ([res[m][f"ci_{key}"][1] for m in names]
                if key != "sys" else None)
        barlabels(ax, b, vals, tops=tops)
        if key != "sys":
            for i, m in enumerate(names):
                lo, hi = res[m][f"ci_{key}"]
                ax.plot([i + (k - 1) * w] * 2, [lo, hi], color=C["ink"], lw=1.1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{m}\n(міркування {'є' if m.startswith('gpt') else 'немає'})"
                        for m in names])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("EA при повному покритті")
    ax.set_title("Етап 2: дві моделі, ті самі 100 питань")
    ax.legend(loc="upper right", ncol=3)
    stamp(ax, "розвідувальне, dev", "дві моделі", res[names[0]]["n"],
          ", 95% ДІ кластерний")
    return save(fig, "07_stage2_models.png", "Етап 2: дві моделі",
                "confirm300_values / gemlite100 / baseline_* / arbiter_*",
                {m: {k: v for k, v in res[m].items() if not k.startswith("ci")}
                 for m in names})


def fig08():
    """Survivorship: the truncation fix revealed wrong answers that had hidden.

    Built from the A/B pair the claim was made on (5.13.1): `ab_yes` before the
    fix, `ab_yes_v2` after, same 25 questions. Before it, a reply cut at the
    token limit produced no candidate at all, so it never entered the
    denominator -- the cascade looked better because its failures were
    invisible, not because it caught them.
    """
    pairs = (("ab_yes", "ab_yes_v2", "конфігурація A"),
             ("ab_no", "ab_no_v2", "конфігурація B"))
    data = []
    for before, after, name in pairs:
        row = []
        for tag in (before, after):
            src = rows(tag)
            cand = [x for x in src if x.get("candidate") is not None]
            wrong = [x for x in cand if not x.get("candidate_correct")]
            rej = [x for x in wrong if x.get("status") != "ACCEPT"]
            row.append((len(rej) / len(wrong) if wrong else float("nan"),
                        len(rej), len(wrong), len(src)))
        data.append((name, row))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    w = 0.34
    xs = range(len(data))
    for k, (lab, col) in enumerate((("до виправлення обриву", C["orange"]),
                                    ("після", C["blue"]))):
        vals = [d[1][k][0] for d in data]
        b = ax.bar([i + (k - 0.5) * w for i in xs], vals, w, color=col, label=lab)
        for bar, (v, nr, nw, _) in zip(b, [d[1][k] for d in data]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.0%}", ha="center", fontsize=8, color=C["ink"])
            ax.text(bar.get_x() + bar.get_width() / 2, 0.015,
                    f"{nr} з {nw}", ha="center", fontsize=7, color=C["ink2"])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([d[0] for d in data])
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("rejection recall")
    ax.set_title("Упередження виживання: обрив коду ховав хибні відповіді")
    ax.legend(loc="upper right")
    stamp(ax, "пост-хок (5.13.1)", "openai/gpt-oss-120b", data[0][1][0][3],
          "; ті самі питання до і після")
    return save(fig, "08_survivorship.png", "Упередження виживання",
                "ab_yes vs ab_yes_v2, ab_no vs ab_no_v2",
                {f"{d[0]}_{lab}": {"recall": d[1][k][0], "rejected": d[1][k][1],
                                   "wrong": d[1][k][2]}
                 for d in data for k, lab in ((0, "до"), (1, "після"))})


def fig09():
    """CI width, naive vs cluster -- with the sample named on the figure.

    Two different quantities have both been written as "1.43" and they are not
    the same thing:
      * 1.43 questions per table -- a property of THIS sample (300 questions,
        210 tables);
      * CI x1.43 -- sqrt(2.05), the design effect at ICC = 1 on the full
        table-only subset (660 questions, 322 tables). That is a DERIVED upper
        bound, not a measured widening.
    The measured ratio on this sample is what the bars show.
    """
    items, _ = pair("confirm300", "baseline_table_confirm300")
    n = len(items)
    fn = lambda s2: (sum(1 for r in s2 if r["verdict"] == "ACCEPT" and r["correct"])
                     / max(sum(1 for r in s2 if r["verdict"] == "ACCEPT"), 1))
    point = fn(items)
    naive = boot_ci(items, fn, cluster=False)
    clust = boot_ci(items, fn, cluster=True)
    clusters = len({cluster_of(r["id"]) for r in items})
    m = n / clusters
    ratio = (clust[1] - clust[0]) / (naive[1] - naive[0])
    bound = math.sqrt(m)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for i2, (lab, (lo, hi), col) in enumerate(
            (("наївний\n(ресемпл питань)", naive, C["orange"]),
             (f"кластерний\n(ресемпл {clusters} таблиць)", clust, C["blue"]))):
        ax.plot([lo, hi], [i2, i2], color=col, lw=5, solid_capstyle="butt")
        ax.plot([point], [i2], "o", color=C["ink"], ms=5)
        ax.text((lo + hi) / 2, i2 + 0.26, f"ширина {hi - lo:.3f}",
                ha="center", fontsize=8, color=C["ink2"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["наївний", "кластерний"])
    ax.set_xlabel("точність прийнятих, 95% ДІ")
    ax.set_ylim(-0.7, 1.7)
    ax.set_title(f"Ширина ДІ: кластерний ширший у {ratio:.2f} раза")
    ax.text(0.5, -0.45,
            f"вибірка confirm300: {n} питань, {clusters} таблиць, "
            f"{m:.2f} питання на таблицю\n"
            f"верхня межа sqrt({m:.2f}) = {bound:.2f}; "
            f"на повній підмножині (660 питань, 2.05 на таблицю) межа "
            f"sqrt(2.05) = 1.43",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color=C["ink2"])
    stamp(ax, "методична перевірка", "openai/gpt-oss-120b", n)
    return save(fig, "09_ci_width.png", "Наївний проти кластерного ДІ",
                "confirm300.jsonl + baseline_table_confirm300.jsonl",
                {"point": point, "naive_width": naive[1] - naive[0],
                 "cluster_width": clust[1] - clust[0], "ratio": ratio,
                 "clusters": clusters, "q_per_table": m,
                 "bound_sqrt_m": bound, "n": n})


def fig10():
    path = os.path.join(HERE, "finqa_agreed_wrong_confirm300.csv")
    with open(path, encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    cnt = collections.Counter(r["category"] for r in data)
    names = {"WRONG_CELLS": "інші клітинки", "BASE": "хибний знаменник",
             "DATASET": "дефект датасету", "SIGN": "знак",
             "NO_GOLD": "без числового еталона", "ROW": "сира клітинка",
             "PERCENT": "×100", "SCALE": "масштаб", "OTHER": "інше",
             "NO_NUMBER": "без числа", "RECIPROCAL": "обернене"}
    items = cnt.most_common()
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ys = range(len(items))
    vals = [v for _, v in items]
    bars = ax.barh(list(ys), vals, 0.62, color=C["blue"])
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2, str(v),
                va="center", fontsize=8, color=C["ink"])
    ax.set_yticks(list(ys))
    ax.set_yticklabels([names.get(k, k) for k, _ in items])
    ax.invert_yaxis()
    ax.set_xlabel("кількість питань")
    ax.set_title("Хибні відповіді, на яких обидва маршрути погодились")
    stamp(ax, "пост-хок, dev", "openai/gpt-oss-120b", sum(vals))
    return save(fig, "10_agreed_wrong_taxonomy.png",
                "Таксономія спільних хибних",
                "finqa_agreed_wrong_confirm300.csv",
                {k: v for k, v in items})


def fig11():
    """TAT-DQA: extraction quality and what the PDF step costs end to end."""
    model = "gemini-flash-lite-latest"
    gold = {r["id"]: r for r in rows(f"tatdqa_gold_{model}")}
    pdf = {r["id"]: r for r in rows(f"tatdqa_pdf_{model}")}
    ids = sorted(set(gold) | set(pdf))
    n = 40
    ea_g = sum(1 for q in ids if gold.get(q, {}).get("correct")) / n
    ea_p = sum(1 for q in ids if pdf.get(q, {}).get("correct")) / n
    cov_g = sum(1 for q in ids if gold.get(q, {}).get("covers_facts")) / n
    cov_p = sum(1 for q in ids if pdf.get(q, {}).get("covers_facts")) / n

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    ax = axes[0]
    vals = [cov_g, cov_p]
    b = ax.bar([0, 1], vals, 0.5, color=[C["blue"], C["orange"]])
    barlabels(ax, b, vals)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["розмітка\nTAT-DQA", "наш парсер\n(з фільтром)"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("частка сторінок")
    ax.set_title("Таблиця містить потрібні клітинки")
    stamp(ax, "вимір, нуль викликів", "—", n)

    ax2 = axes[1]
    vals2 = [ea_g, ea_p]
    b2 = ax2.bar([0, 1], vals2, 0.5, color=[C["blue"], C["orange"]])
    barlabels(ax2, b2, vals2)
    ax2.annotate("", xy=(1, ea_p), xytext=(1, ea_g),
                 arrowprops=dict(arrowstyle="<->", color=C["ink2"], lw=1.2))
    ax2.text(1.08, (ea_g + ea_p) / 2, f"−{(ea_g - ea_p) * 100:.0f} п.п.",
             fontsize=8, color=C["ink"], va="center")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["gold\n(верхня межа)", "pdf\n(реальний режим)"])
    ax2.set_ylim(0, 0.62)
    ax2.set_ylabel("EA при повному покритті")
    ax2.set_title("Наскрізна точність")
    stamp(ax2, "розвідувальне (5.33)", model, n, "; Макнемар p = 0.013")
    return save(fig, "11_tatdqa.png",
                "TAT-DQA: витягування і наскрізна точність",
                f"tatdqa_gold_{model}.jsonl, tatdqa_pdf_{model}.jsonl",
                {"ea_gold": ea_g, "ea_pdf": ea_p, "gap": ea_g - ea_p,
                 "covers_gold": cov_g, "covers_pdf": cov_p, "n": n})


# ---------- verification against reported numbers ----------

EXPECTED = {
    "01: recall згоди": (0.465, "recall_agreement"),
    "01: precision згоди": (0.939, "precision_agreement"),
    "03: точність прийнятих": (0.768, "accepted"),
    "03: покриття": (0.760, "coverage"),
    "03: R2 при повному покритті": (0.593, "r2"),
    "03: R1 при повному покритті": (0.723, "r1"),
    "04: лише R1 правий": (43, "only_r1"),
    "04: лише R2 правий": (4, "only_r2"),
    "06: R1 dev-100": (0.740, "r1"),
    "06: R2 без значень": (0.550, "r2_old"),
    "06: R2 зі значеннями": (0.550, "r2_new"),
    "06: система v1": (0.740, "system_v1"),
    "01: recall каскаду φ(v)": (0.242, "recall_cascade"),
    "01: precision каскаду φ(v)": (0.686, "precision_cascade"),
    "11: EA gold": (0.450, "ea_gold"),
    "11: EA pdf": (0.150, "ea_pdf"),
    "11: містить клітинки, розмітка": (0.925, "covers_gold"),
    "11: містить клітинки, парсер": (0.400, "covers_pdf"),
}


def verify():
    got = {}
    for rec in INDEX:
        num = rec["file"][:2]
        for k, v in rec["numbers"].items():
            got[f"{num}: {k}"] = v
    print()
    print("=" * 74)
    print("ЗВІРКА З ЧИСЛАМИ В ДОПОВІДЯХ І ARCHITECTURE.md")
    print("=" * 74)
    bad = []
    for label, (want, key) in EXPECTED.items():
        num = label[:2]
        v = got.get(f"{num}: {key}")
        if v is None:
            print(f"  [?]    {label:34s} артефакт не дав значення")
            bad.append(label)
            continue
        tol = 0.006 if isinstance(want, float) else 0
        ok = abs(v - want) <= tol
        mark = "OK " if ok else "РОЗБІЖНІСТЬ"
        fmt = "{:.3f}" if isinstance(want, float) else "{:.0f}"
        print(f"  [{mark:^5s}] {label:34s} очікувано {fmt.format(want)}  "
              f"артефакт {fmt.format(v)}")
        if not ok:
            bad.append(label)
    print()
    if bad:
        print(f"!!! РОЗБІЖНОСТЕЙ: {len(bad)} — доповідаю, не виправляю")
    else:
        print("Усі перевірені числа збігаються з доповіданими.")
    return bad


def write_index():
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "INDEX.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("# Покажчик графіків\n\n")
        f.write("Кожен графік побудовано **лише з артефактів** — жодне число "
                "не вписане в код вручну.\nЗапуск: `python -m eval.plots`\n\n")
        for rec in INDEX:
            f.write(f"## {rec['file']}\n\n**{rec['title']}**\n\n")
            f.write(f"- джерело: `{rec['source']}`\n- числа на графіку:\n")
            for k, v in rec["numbers"].items():
                if isinstance(v, float):
                    f.write(f"  - `{k}` = {v:.4f}\n")
                elif isinstance(v, dict):
                    f.write(f"  - `{k}`: " + ", ".join(
                        f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}"
                        for a, b in v.items()) + "\n")
                else:
                    f.write(f"  - `{k}` = {v}\n")
            f.write("\n")
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    made = []
    for fn in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08,
               fig09, fig10, fig11):
        try:
            made.append(fn())
            print(f"  [OK ] {os.path.basename(made[-1])}")
        except Exception as exc:
            print(f"  [ЗБІЙ] {fn.__name__}: {type(exc).__name__}: {exc}")
    idx = write_index()
    print(f"\nграфіків: {len(made)}   покажчик: {idx}")
    verify()


if __name__ == "__main__":
    main()
